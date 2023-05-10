import pytest


def get_data(rng):
    import numpy as np

    num = 20
    dtype = [('id', 'i8'), ('rand', 'f4'), ('scol', 'U5')]
    data = np.zeros(num, dtype=dtype)
    data['id'] = np.arange(num)
    data['rand'] = rng.uniform(size=num)
    data['scol'] = [
        's' + str(data['id'][i]) for i in range(num)
    ]

    subnum = 25
    sub_data = np.zeros(subnum, dtype=[('ra', 'f8'), ('dec', 'f8')])
    sub_data['ra'] = rng.uniform(size=sub_data.size)
    sub_data['dec'] = rng.uniform(size=sub_data.size)

    sub2num = 15
    sub2_data = np.zeros(sub2num, dtype=[('x', 'f4')])
    sub2_data['x'] = rng.uniform(size=sub2_data.size)

    return data, sub_data, sub2_data


@pytest.mark.parametrize('compression', [False, True])
def test_access(compression):
    """
    cache_mem of 0.01 will force use of mergesort
    """
    import os
    import tempfile
    import numpy as np
    from ..columns import Columns
    from ..indices import Indices

    seed = 333

    rng = np.random.RandomState(seed)

    data, sub_data, sub2_data = get_data(rng)

    if compression:
        ccols = ['id', 'scol']
    else:
        ccols = None

    with tempfile.TemporaryDirectory() as tmpdir:

        cdir = os.path.join(tmpdir, 'test.cols')

        # Create a new Columns store with data in the root
        cols = Columns.create_from_array(
            cdir,
            data=data,
            compression=ccols,
            verbose=True,
        )

        assert len(cols.names) == len(data.dtype.names)

        versions = {'numpy': '1.23.5'}
        cols.create_meta('versions', versions)

        rmeta = cols.meta['versions'].read()
        assert rmeta == versions

        cols.create_meta('list', [3, 4])
        assert cols.meta['list'].read() == [3, 4]

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
            assert np.all(data[name][-5:-2] == cols[name][-5:-2])

        for name in data.dtype.names:
            assert np.all(data[name][5] == cols[name][5])

        #
        # update data in columns and meta
        #

        dadd = {'pycolumns': '2.0.0'}
        cols.meta['versions'].update(dadd)
        versions.update(dadd)

        assert cols.meta['versions'].read() == versions

        dnew = {'replaced': 3}
        cols.meta['versions'].write(dnew)
        assert cols.meta['versions'].read() == dnew

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
        num = cols.size
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

        # filling slice with scalar
        cols['id'][5:10] = 3
        assert np.all(cols['id'][5:10] == 3)

        # filling slice with scalar
        cols['id'][-5:-2] = 9
        assert np.all(cols['id'][-5:-2] == 9)

        cols['scol'][5:10] = 'test'
        assert np.all(cols['scol'][5:10] == 'test')

        # filling rows with a scalar
        ind = Indices([3, 5, 8], is_sorted=True)
        cols['id'][ind] = 9999
        assert np.all(cols['id'][ind] == 9999)

        cols['scol'][ind] = 'test'
        assert np.all(cols['scol'][ind] == 'test')

        # filling rows unsorted
        ind = [8, 1, 3]
        cols['id'][ind] = -8888
        assert np.all(cols['id'][ind] == -8888)

        cols['scol'][ind] = '333'
        assert np.all(cols['scol'][ind] == '333')

        # This only works filling gall items or if data
        # has the right length
        cols['rand'] = 8
        assert np.all(cols['rand'][:] == 8)

        d = rng.uniform(size=cols.nrows).astype('f4')
        cols['rand'] = d
        assert np.all(cols['rand'][:] == d)

        with pytest.raises(IndexError):
            # not long enough
            cols['rand'] = [3, 4]

        #
        # adding more tables
        #

        cols.from_array(name='sub1/', data=sub_data)
        cols['sub1/']
        cols['sub1/dec']

        assert len(cols.names) == len(data.dtype.names) + 1

        cols['sub1/'].from_array(name='sub2/', data=sub2_data)
        cols['sub1/']['sub2/']
        cols['sub1/sub2/']
        cols['sub1/sub2/x']

        # use of "in"
        assert 'sub1/sub2/' in cols

        cols.from_array(name='sub1/sub2/sub3/', data=sub2_data)
        cols['sub1/sub2/sub3/']
        cols['sub1/sub2/sub3/x']

        cols2 = Columns(cdir)
        cols2['sub1/']['dec']
        cols2['sub1/dec']
        cols2['sub1/sub2/']
        cols2['sub1/sub2/x']
        cols2['sub1/sub2/sub3/x']

        with pytest.raises(IndexError):
            cols['sub1/sub2/sub3/sub4/']

        ra = cols['sub1/']['ra'][:]
        assert np.all(ra == sub_data['ra'])
        assert np.all(cols['sub1/ra'][:] == sub_data['ra'])
        with pytest.raises(TypeError):
            cols['sub1/'] = 5

        x = cols['sub1/sub2/']['x'][:]
        assert np.all(x == sub2_data['x'])
        x = cols['sub1/sub2/x'][:]
        assert np.all(x == sub2_data['x'])


def test_set_compressed():
    """
    cache_mem of 0.01 will force use of mergesort
    """
    import os
    import tempfile
    import numpy as np
    from ..columns import Columns

    seed = 333
    num = 100_000
    chunksize = '10000r'

    rng = np.random.RandomState(seed)

    dtype = [('id', 'i8'), ('rand', 'f4'), ('scol', 'U5')]
    data = np.zeros(num, dtype=dtype)
    data['id'] = np.arange(num)
    data['rand'] = rng.uniform(size=num)
    data['scol'] = [
        's' + str(data['id'][i]) for i in range(num)
    ]

    ccols = ['id', 'scol']

    with tempfile.TemporaryDirectory() as tmpdir:

        cdir = os.path.join(tmpdir, 'test.cols')
        cols = Columns.create(cdir, verbose=True)
        cols.from_array(data=data, compression=ccols, chunksize=chunksize)

        assert np.all(cols['id'][:] == data['id'])

        ndata = rng.randint(0, 2**16, size=cols.size)
        cols['id'][:] = ndata
        assert np.all(cols['id'][:] == ndata)

        cols['id'][:] = 999
        assert np.all(cols['id'][:] == 999)

        print(f"nchunks: {cols['id']._col.nchunks}")
        cols.append(data)
        print(f"nchunks: {cols['id']._col.nchunks}")
        assert np.all(cols['id'][data.size:] == data['id'])

        ndata = rng.randint(0, 2**16, size=cols.size)
        cols['id'][data.size:] = ndata
