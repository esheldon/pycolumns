import pytest


@pytest.mark.parametrize('compression', [False, True])
def test_access(compression):
    """
    cache_mem of 0.01 will force use of mergesort
    """
    import os
    import tempfile
    import numpy as np
    from ..columns import Columns
    from ..schema import TableSchema

    seed = 333
    num = 20

    rng = np.random.RandomState(seed)

    dtype = [('id', 'i8'), ('rand', 'f4'), ('scol', 'U5')]
    data = np.zeros(num, dtype=dtype)
    data['id'] = np.arange(num)
    data['rand'] = rng.uniform(size=num)
    data['scol'] = [
        's' + str(data['id'][i]) for i in range(num)
    ]

    if compression:
        ccols = ['id', 'scol']
    else:
        ccols = None

    schema = TableSchema.from_array(data, compression=ccols)

    print(schema)

    with tempfile.TemporaryDirectory() as tmpdir:

        cdir = os.path.join(tmpdir, 'test.cols')
        cols = Columns.create(cdir, schema, verbose=True)

        cols.append(data)

        assert len(cols.names) == len(data.dtype.names)
        meta = {'version': '0.1', 'seeing': 0.9}
        cols.create_dict('meta')
        cols['meta'].write(meta)

        rmeta = cols['meta'].read()
        assert rmeta == meta

        indata = cols.read()
        for name in data.dtype.names:
            assert np.all(data[name] == indata[name])

        for name in data.dtype.names:
            tdata = cols[name][:]
            assert np.all(data[name] == tdata)

        # sorted rows
        rows = np.arange(0, data.size, 2)
        indata = cols.read(rows=rows)
        for name in data.dtype.names:
            assert np.all(data[name][rows] == indata[name])

        for name in data.dtype.names:
            assert np.all(data[name][rows] == cols[name][rows])

        for name in data.dtype.names:
            assert np.all(data[name][10:15] == cols[name][10:15])

        for name in data.dtype.names:
            assert np.all(data[name][5] == cols[name][5])

        #
        # update data in columns
        #

        dadd = {'new': 'hello'}
        cols['meta'].update(dadd)
        meta.update(dadd)

        rmeta = cols['meta'].read()
        assert rmeta == meta

        if not compression:
            for name in data.dtype.names:
                data[name][10:15] = (
                    np.arange(1000, 1000+5).astype(data[name].dtype)
                )
                cols[name][10:15] = data[name][10:15]
                assert np.all(data[name][:] == cols[name][:])

            # sorted rows
            rows = [5, 15, 17]
            vals = np.arange(len(rows))
            for name in data.dtype.names:
                data[name][rows] = vals.astype(data[name].dtype)
                cols[name][rows] = data[name][rows]
                assert np.all(data[name][:] == cols[name][:])

            # unsorted rows
            rows = [17, 5, 15]
            vals = np.arange(len(rows))
            for name in data.dtype.names:
                data[name][rows] = vals.astype(data[name].dtype)
                cols[name][rows] = data[name][rows]
                assert np.all(data[name][:] == cols[name][:])

            row = 8
            val = 1999
            for name in data.dtype.names:
                data[name][row] = val
                cols[name][row] = val
                assert np.all(data[name][:] == cols[name][:])

            # check negative indices
            assert cols[name][-2] == cols[name][num-2]
            assert np.all(cols[name][[-3, -1]] == cols[name][[num-3, num-1]])
            assert np.all(cols[name][-3:] == cols[name][[num-3, num-2, num-1]])

            with pytest.raises(ValueError):
                # tring to set number from string not representing numbers
                cols['rand'][:] = cols['scol'][:]

            # automatic conversion works num -> string
            cols['scol'][:] = np.arange(cols.nrows)
            assert np.all(
                cols['scol'][:] == np.arange(cols.nrows).astype('U5')
            )

            cols['rand'][:] = cols['id'][:]
            assert np.all(
                cols['rand'][:] == cols['id'][:].astype(cols['rand'].dtype)
            )
