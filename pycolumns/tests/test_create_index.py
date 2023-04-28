import pytest


@pytest.mark.parametrize('cache_mem', [1.0, 0.01])
@pytest.mark.parametrize('compression', [False, True])
# @pytest.mark.parametrize('cache_mem', [1.0])
def test_create_index(cache_mem, compression):
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
    rng = np.random.RandomState(seed)
    data = np.zeros(num, dtype=[('rand', 'f8')])
    data['rand'] = rng.uniform(size=num)

    if compression:
        ccols = ['id', 'scol']
    else:
        ccols = None

    with tempfile.TemporaryDirectory() as tmpdir:

        cdir = os.path.join(tmpdir, 'test.cols')
        cols = Columns.from_array(
            cdir, data, cache_mem=cache_mem, compression=ccols, verbose=True,
        )

        cols['rand'].create_index()
        assert cols['rand'].has_index

        ifile = cols['rand'].index_filename
        sfile = cols['rand'].sorted_filename
        idata = _column.read(ifile, dtype=cols['rand'].index_dtype)
        sdata = _column.read(sfile, dtype=cols['rand'].dtype)

        s = data['rand'].argsort()
        assert np.all(idata == s)
        assert np.all(sdata == data['rand'][s])


@pytest.mark.parametrize('compression', [False, True])
def test_create_index_str(compression):
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
    rng = np.random.RandomState(seed)

    dt = 'U5'
    data = np.zeros(num, dtype=[('scol', dt)])
    rand = rng.uniform(size=num)
    data['scol'] = rand.astype(dt)

    if compression:
        ccols = ['id', 'scol']
    else:
        ccols = None

    with tempfile.TemporaryDirectory() as tmpdir:

        cdir = os.path.join(tmpdir, 'test.cols')
        cols = Columns.from_array(
            cdir, array=data, compression=ccols, overwrite=True, verbose=True,
        )

        cols['scol'].create_index()
        assert cols['scol'].has_index

        ifile = cols['scol'].index_filename
        sfile = cols['scol'].sorted_filename
        idata = _column.read(ifile, dtype=cols['scol'].index_dtype)
        sdata = _column.read(sfile, dtype=cols['scol'].dtype)

        s = data['scol'].argsort()
        assert np.all(idata == s)
        assert np.all(sdata == data['scol'][s])

        ind = (data['scol'] == data['scol'][5])
        rd = data['scol'][ind]
        assert rd.size == 1
