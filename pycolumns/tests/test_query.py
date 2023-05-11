import pytest


def test_query():
    import os
    import tempfile
    import numpy as np
    from ..columns import Columns

    seed = 333
    num = 20

    rng = np.random.RandomState(seed)

    data = np.zeros(num, dtype=[('id', 'i8'), ('rand', 'f4')])
    data['id'] = np.arange(num)
    data['rand'] = rng.uniform(size=num, high=num)

    with tempfile.TemporaryDirectory() as tmpdir:

        cdir = os.path.join(tmpdir, 'test.cols')
        cols = Columns.create_from_array(cdir, data=data, verbose=True)

        for name in data.dtype.names:
            assert not cols[name].has_index
            with pytest.raises(ValueError):
                cols[name] > 10

        for name in data.dtype.names:
            cols[name].create_index()
            assert cols[name].has_index

            ind = cols[name] > 10
            w, = np.where(data[name] > 10)
            assert w.size > 0
            assert np.all(ind == w)

            ind = cols[name].between(10, 20)
            w, = np.where((data[name] >= 10) & (data[name] <= 20))
            assert w.size > 0
            assert np.all(ind == w)

        # equality
        ind = cols['id'] == 12
        w, = np.where(data['id'] == 12)
        assert w.size == 1
        assert np.all(ind == w)

        # match
        ind = cols['id'].match([9, 15])
        w, = np.where((data['id'] == 9) | (data['id'] == 15))
        assert w.size == 2
        assert np.all(ind == w)

        # tests on two columns with &
        ind = (cols['id'] > 10) & (cols['rand'] > 10)
        w, = np.where((data['id'] > 10) & (data['rand'] > 10))

        assert w.size > 0
        assert np.all(ind == w)

        ind = (cols['id'] >= 10) & (cols['rand'] > 10)
        w, = np.where((data['id'] >= 10) & (data['rand'] > 10))

        assert w.size > 0
        assert np.all(ind == w)

        ind = (cols['id'] <= 10) & (cols['rand'] > 10)
        w, = np.where((data['id'] <= 10) & (data['rand'] > 10))

        assert w.size > 0
        assert np.all(ind == w)

        # test on two columns with |
        ind = (cols['id'] < 10) | (cols['rand'] > 10)
        w, = np.where((data['id'] < 10) | (data['rand'] > 10))

        assert w.size > 0
        assert np.all(ind == w)
