# import pytest


def test_delete():
    import os
    import tempfile
    import numpy as np
    from ..columns import Columns
    from ..schema import TableSchema

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

    schema = TableSchema.from_array(data, compression=ccols)

    print(schema)

    with tempfile.TemporaryDirectory() as tmpdir:

        cdir = os.path.join(tmpdir, 'test.cols')
        cols = Columns.create(cdir, schema, verbose=True)

        cols.append(data)
        cols.create_dict('metadata')
        cols.dicts['metadata'].write({'x': 3})
        # cols['metadata'] = {'x': 3}

        cols.create_sub_from_array(name='/sub', array=sub_data)

        iddir = cols['id'].dir
        assert os.path.exists(iddir)

        cols.delete_entry('id', yes=True)
        assert not os.path.exists(iddir)

        fname = cols.dicts['metadata'].filename
        assert os.path.exists(fname)

        cols.delete_dict('metadata', yes=True)
        assert not os.path.exists(fname)

        cols.delete(yes=True)
        assert not os.path.exists(cdir)
