def test(seed=999, num=1_000_000):
    import tempfile
    import os
    import shutil
    import numpy as np
    import esutil as eu
    from .. import sfile
    from ..mergesort import mergesort_index

    valname = 'val'

    rng = np.random.RandomState(seed)
    data = np.zeros(num, dtype=[('index', 'i8'), (valname, 'f8')])
    data['index'] = np.arange(num)
    data['val'] = rng.uniform(size=num)

    with tempfile.TemporaryDirectory() as tmpdir:
        infile = tempfile.mktemp(dir=tmpdir, prefix='infile-', suffix='.sf')
        outfile = tempfile.mktemp(dir=tmpdir, prefix='outfile-', suffix='.sf')

        if os.path.exists(infile):
            os.remove(infile)
        if os.path.exists(outfile):
            os.remove(outfile)

        with sfile.SimpleFile(infile, mode='w+') as sf:
            sf.write(data)

        shutil.copy(infile, outfile)

        with sfile.SimpleFile(infile, mode='r') as sfin:
            with sfile.SimpleFile(outfile, mode='r+') as sfout:

                chunk_size_mbytes = 500
                # chunk_size_mbytes = 50
                # chunk_size_mbytes = 100
                chunk_size_bytes = chunk_size_mbytes*1_000_000

                bytes_per_element = sfin.dtype.itemsize
                chunksize = chunk_size_bytes//bytes_per_element

                mergesort_index(
                    source=sfin._mmap,
                    sink=sfout._mmap,
                    order=valname,
                    chunksize=chunksize,
                    tmpdir=tmpdir,
                )

        sdata = sfile.read(outfile)
        data.sort(order=valname, kind='mergesort')
        assert eu.numpy_util.compare_arrays(sdata, data)
