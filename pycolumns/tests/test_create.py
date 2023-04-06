import pytest


@pytest.mark.parametrize('cache_mem', [1.0, 0.01])
@pytest.mark.parametrize('verbose', [True, False])
def test_create(cache_mem, verbose):
    """
    cache_mem of 0.01 will force use of mergesort
    """
    import os
    import tempfile
    import numpy as np
    from ..columns import Columns

    seed = 333
    num = 20

    rng = np.random.RandomState(seed)

    with tempfile.TemporaryDirectory() as tmpdir:
        cdir = os.path.join(tmpdir, 'test.cols')
        cols = Columns(cdir, cache_mem=cache_mem, verbose=verbose)
        assert len(cols.colnames) == 0

        assert cols.dir == cdir
        assert cols.verbose == verbose
        assert cols.cache_mem == cache_mem

        data = np.zeros(num, dtype=[('id', 'i8'), ('rand', 'f4')])
        data['id'] = np.arange(num)
        data['rand'] = rng.uniform(size=num)

        cols.append(data)

        assert len(cols.colnames) == len(data.dtype.names)
        meta = {'version': '0.1', 'seeing': 0.9}
        cols.create_column('meta', 'dict')
        cols['meta'].write(meta)

        rmeta = cols['meta'].read()
        assert rmeta == meta

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

        # don't allow appending with new columns
        with pytest.raises(ValueError):
            bad_data = np.zeros(3, dtype=[('blah', 'f4')])
            cols.append(bad_data)

        with pytest.raises(ValueError):
            bad_data = np.zeros(
                3,
                dtype=[('id', 'i8'), ('rand', 'f4'), ('extra', 'i2')],
            )
            cols.append(bad_data)

        # can currently update column at a time
        cols['rand'][5] = 35
        assert cols['rand'][5] == 35

        idx = [8, 12]
        vals = [1.0, 2.0]
        cols['rand'][idx] = vals
        assert np.all(cols['rand'][idx] == vals)
