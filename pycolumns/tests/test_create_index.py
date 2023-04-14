import pytest


@pytest.mark.parametrize('cache_mem', [1.0, 0.01])
def test_create_index(cache_mem):
    """
    cache_mem of 0.01 will force use of mergesort
    """
    import os
    import tempfile
    import numpy as np
    from .. import _column
    from ..columns import Columns

    seed = 333
    num = 1_000_000

    with tempfile.TemporaryDirectory() as tmpdir:
        cdir = os.path.join(tmpdir, 'test.cols')
        cols = Columns(cdir, cache_mem=cache_mem, verbose=True)

        rng = np.random.RandomState(seed)
        data = np.zeros(num, dtype=[('rand', 'f8')])
        data['rand'] = rng.uniform(size=num)

        cols.append(data)
        cols['rand'].create_index()
        assert cols['rand'].has_index

        ifile = cols['rand'].index_filename
        sfile = cols['rand'].sorted_filename
        idata = _column.read(ifile)
        sdata = _column.read(sfile)

        s = data['rand'].argsort()
        assert np.all(idata == s)
        assert np.all(sdata == data['rand'][s])


def test_create_index_str():
    """
    cache_mem of 0.01 will force use of mergesort
    """
    import os
    import tempfile
    import numpy as np
    from .. import _column
    from ..columns import Columns

    seed = 55
    num = 20

    with tempfile.TemporaryDirectory() as tmpdir:
        cdir = os.path.join(tmpdir, 'test.cols')
        cols = Columns(cdir, verbose=True)

        rng = np.random.RandomState(seed)
        data = np.zeros(num, dtype=[('scol', 'U5')])
        rand = rng.uniform(size=num)
        data['scol'] = ['%.2g' % v for v in rand]

        cols.append(data)
        cols['scol'].create_index()
        assert cols['scol'].has_index

        ifile = cols['scol'].index_filename
        sfile = cols['scol'].sorted_filename
        idata = _column.read(ifile)
        sdata = _column.read(sfile)

        s = data['scol'].argsort()
        assert np.all(idata == s)
        assert np.all(sdata == data['scol'][s])

        ind = (data['scol'] == data['scol'][5])
        rd = data['scol'][ind]
        assert rd.size == 1
