import pytest


def test_vacuum():
    import os
    import tempfile
    import numpy as np
    from ..columns import Columns

    seed = 9911
    num = 200_000

    rng = np.random.RandomState(seed)

    dtype = [('id', 'i8')]
    data = np.zeros(num, dtype=dtype)

    # this will compress a lot
    data['id'] = np.zeros(num)

    with tempfile.TemporaryDirectory() as tmpdir:

        cdir = os.path.join(tmpdir, 'test.cols')
        cols = Columns.create(cdir, verbose=True)
        cols.from_array(data, compression=['id'])

        # this will expand quite a bit, fragmenting the file
        rind = rng.randint(0, 2**31, size=num)
        cols['id'][:] = rind
        assert np.all(cols['id'][:] == rind)

        cc = cols['id']._col
        w, = np.where(cc.chunk_data['is_external'])
        assert w.size > 0

        for chunk_index in w:
            ext_fname = cc.get_external_filename(chunk_index)
            assert os.path.exists(ext_fname)

        cols.vacuum()

        assert np.all(cols['id'][:] == rind)

        for chunk_index in w:
            ext_fname = cc.get_external_filename(chunk_index)
            assert not os.path.exists(ext_fname)

        # no vaccum in read only mode
        rocols = Columns(cdir)
        with pytest.raises(IOError):
            rocols.vacuum()


def test_vacuum_context():
    import os
    import tempfile
    import numpy as np
    from ..columns import Columns

    seed = 9911
    num = 200_000

    rng = np.random.RandomState(seed)

    dtype = [('id', 'i8')]
    data = np.zeros(num, dtype=dtype)

    # this will compress a lot
    data['id'] = np.zeros(num)

    with tempfile.TemporaryDirectory() as tmpdir:

        cdir = os.path.join(tmpdir, 'test.cols')
        cols = Columns.create(cdir, verbose=True)
        cols.from_array(data, compression=['id'])

        # this will expand quite a bit, fragmenting the file
        with cols['id'].updating(vacuum=True):
            rind = rng.randint(0, 2**31, size=num)
            cols['id'][:] = rind
            assert np.all(cols['id'][:] == rind)

            cc = cols['id']._col
            w, = np.where(cc.chunk_data['is_external'])
            assert w.size > 0

            for chunk_index in w:
                ext_fname = cc.get_external_filename(chunk_index)
                assert os.path.exists(ext_fname)

        assert np.all(cols['id'][:] == rind)

        for chunk_index in w:
            ext_fname = cc.get_external_filename(chunk_index)
            assert not os.path.exists(ext_fname)

        rocols = Columns(cdir)
        with pytest.raises(IOError):
            with rocols.updating():
                pass
