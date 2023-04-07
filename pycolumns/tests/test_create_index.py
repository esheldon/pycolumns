import pytest


@pytest.mark.parametrize('cache_mem', [1.0, 0.01])
def test_create_index(cache_mem):
    """
    cache_mem of 0.01 will force use of mergesort
    """
    import os
    import tempfile
    import numpy as np
    from .. import sfile
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
        idata = sfile.read(ifile)

        s = data['rand'].argsort()
        assert np.all(idata['value'] == data['rand'][s])
