# import pytest


def test_delete():
    import os
    import tempfile
    import numpy as np
    from ..columns import Columns

    seed = 812
    num = 20

    rng = np.random.RandomState(seed)

    dtype = [('id', 'i8'), ('rand', 'f4'), ('scol', 'U5')]
    data = np.zeros(num, dtype=dtype)
    data['id'] = np.arange(num)
    data['rand'] = rng.uniform(size=num)
    data['scol'] = [str(val) for val in data['id']]

    sub_data = np.zeros(num, dtype=[('ra', 'f8'), ('dec', 'f8')])
    sub_data['ra'] = rng.uniform(size=sub_data.size)
    sub_data['dec'] = rng.uniform(size=sub_data.size)

    ccols = ['id', 'scol']

    with tempfile.TemporaryDirectory() as tmpdir:

        cdir = os.path.join(tmpdir, 'test.cols')

        cols = Columns.create_from_array(
            cdir,
            data=data,
            compression=ccols,
            verbose=True,
        )

        cols.create_meta('metadata')
        cols.meta['metadata'].write({'x': 3})
        # cols['metadata'] = {'x': 3}

        cols.from_array(name='sub/', data=sub_data)

        iddir = cols['id'].dir
        assert os.path.exists(iddir)

        cols.delete_entry('id', yes=True)
        assert not os.path.exists(iddir)

        fname = cols.meta['metadata'].filename
        assert os.path.exists(fname)

        cols.delete_meta('metadata', yes=True)
        assert not os.path.exists(fname)

        cols.delete(yes=True)
        assert not os.path.exists(cdir)
