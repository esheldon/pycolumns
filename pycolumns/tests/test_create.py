import pytest


@pytest.mark.parametrize('cache_mem', ['1g', '10k'])
@pytest.mark.parametrize('compression', [False, True])
@pytest.mark.parametrize('verbose', [True, False])
@pytest.mark.parametrize('fromdict', [False, True])
@pytest.mark.parametrize('from_array', [False, True])
def test_create(cache_mem, compression, verbose, fromdict, from_array):
    """
    cache_mem of 0.01 will force use of mergesort
    """
    import os
    import tempfile
    import numpy as np
    from ..columns import Columns
    from ..schema import TableSchema, ColumnSchema

    seed = 333
    num = 20

    rng = np.random.RandomState(seed)

    dtype = [('id', 'i8'), ('rand', 'f4'), ('scol', 'U5')]
    data = np.zeros(num, dtype=dtype)
    data['id'] = np.arange(num)
    data['rand'] = rng.uniform(size=num)
    data['scol'] = [str(val) for val in data['id']]

    if compression:
        ccols = ['id', 'scol']
    else:
        ccols = None

    if fromdict:
        ddict = {}
        for name in data.dtype.names:
            ddict[name] = data[name]
        schema = TableSchema.from_array(ddict, compression=ccols)
    else:
        schema = TableSchema.from_array(data, compression=ccols)

    print(schema)

    with tempfile.TemporaryDirectory() as tmpdir:

        cdir = os.path.join(tmpdir, 'test.cols')
        cols = Columns.create(cdir, cache_mem=cache_mem, verbose=verbose)

        assert cols.dir == cdir
        assert cols.verbose == verbose
        assert cols.cache_mem == cache_mem

        if fromdict:
            append_data = ddict
        else:
            append_data = data

        # created in root
        if from_array:
            cols.from_array(data=append_data, compression=ccols)
        else:
            cols.create_table(schema=schema)
            cols.append(append_data)

        assert len(cols.names) == len(data.dtype.names)
        meta = {'version': '0.1', 'seeing': 0.9}
        cols.create_meta('metadata', meta)

        assert cols.meta['metadata'].read() == meta

        newd = {'x': 5, 'name': 'joe'}
        meta.update(newd)
        cols.meta['metadata'].update(newd)
        assert cols.meta['metadata'].read() == meta

        ncols = Columns(cdir)
        assert ncols.meta['metadata'].read() == meta

        # after appending, verify is run, but let's double check
        for name in data.dtype.names:
            assert cols[name].size == num
            assert cols[name].verbose == verbose
            assert cols[name].cache_mem == cache_mem

        indata = cols.read()
        for name in data.dtype.names:
            assert np.all(data[name] == indata[name])

        for name in data.dtype.names:
            tdata = cols[name][:]
            assert np.all(data[name] == tdata)

        # make sure we can append more data
        cols.append(data)
        assert cols['id'].size == num * 2
        assert cols['rand'].size == num * 2
        assert cols['scol'].size == num * 2

        # don't allow appending with new columns
        with pytest.raises(ValueError):
            bad_data = np.zeros(3, dtype=[('blah', 'f4')])
            cols.append(bad_data)

        with pytest.raises(ValueError):
            bad_data = np.zeros(
                3,
                dtype=dtype + [('extra', 'i2')],
            )
            cols.append(bad_data)

        newschema = ColumnSchema('newcol', dtype='i4')
        cols.create_column(newschema)
        assert cols['newcol'][:].size == cols.size
        assert np.all(cols['newcol'][:] == 0)

        newschema = ColumnSchema('news', dtype='U2')
        cols.create_column(newschema)
        assert cols['news'][:].size == cols.size
        assert np.all(cols['news'][:] == '')

        newschema = ColumnSchema('fcol', dtype='U2', fill_value='-')
        cols.create_column(newschema)
        assert np.all(cols['fcol'][:] == '-')

        newschema = ColumnSchema('x9', dtype='f4', fill_value=9.5)
        cols.create_column(newschema)
        assert np.all(cols['x9'][:] == 9.5)

        newschema = ColumnSchema('ss', dtype='S3', fill_value='yes')
        cols.create_column(newschema)
        assert np.all(cols['ss'][:] == b'yes')

        with pytest.raises(RuntimeError):
            # can't resize compressed cols
            newschema = ColumnSchema('bad', dtype='U2', compression=True)
            cols.create_column(newschema)
