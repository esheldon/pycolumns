def test_mergesort():
    import tempfile
    import numpy as np
    import esutil as eu
    from .. import sfile
    from ..mergesort import create_mergesort_index

    num = 1_000_000
    chunksize_mbytes = 5

    seed = 555

    rng = np.random.RandomState(seed)
    values = rng.uniform(size=num)

    check_data = np.zeros(num, dtype=[('index', 'i8'), ('value', 'f8')])
    check_data['index'] = np.arange(num)
    check_data['value'] = values

    with tempfile.TemporaryDirectory() as tmpdir:
        infile = tempfile.mktemp(dir=tmpdir, prefix='infile-', suffix='.sf')
        outfile = tempfile.mktemp(dir=tmpdir, prefix='outfile-', suffix='.sf')

        print('writing:', infile)
        sfile.write(infile, values)

        chunksize_bytes = chunksize_mbytes * 1024 * 1024

        bytes_per_element = check_data.dtype.itemsize
        chunksize = chunksize_bytes // bytes_per_element
        print('chunksize:', chunksize)

        create_mergesort_index(
            infile=infile,
            outfile=outfile,
            chunksize=chunksize,
            tmpdir=tmpdir,
        )

        sdata = sfile.read(outfile)
        check_data.sort(order='value', kind='mergesort')
        assert eu.numpy_util.compare_arrays(
            sdata,
            check_data,
            ignore_missing=False,
            verbose=True,
        )
