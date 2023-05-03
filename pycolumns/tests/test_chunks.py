import pytest


@pytest.mark.parametrize('compression', [False, True])
@pytest.mark.parametrize('dtype', ['i8', 'f4', 'U3'])
def test_chunks(compression, dtype):
    import os
    import tempfile
    import numpy as np
    from ..chunks import Chunks

    seed = 812
    num = 100

    rng = np.random.RandomState(seed)

    if dtype[0] == 'i':
        data = np.arange(num, dtype=dtype)
    elif dtype[0] == 'U':
        data = np.zeros(num, dtype=dtype)
        data[:] = [str(i) for i in range(num)]
    else:
        data = rng.uniform(size=num).astype(dtype)

    # small chunksize to guarantee we cross chunks
    row_chunksize = 10
    chunksize = f'{row_chunksize}r'

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.array')
        chunks_fname = os.path.join(tmpdir, 'test.chunks')

        with open(fname, 'w') as fobj:  # noqa
            pass
        with open(chunks_fname, 'w') as fobj:  # noqa
            pass

        with Chunks(
            fname,
            chunks_fname,
            dtype=data.dtype,
            chunksize=chunksize,
            compression=compression,
            mode='w+',
            verbose=True,
        ) as col:

            print('-' * 70)
            print('before append')
            print(col)

            col.append(data)
            assert col.nchunks > 1
            assert col.row_chunksize == row_chunksize

            print('-' * 70)
            print('before append')
            print(col)

            indata = col[:]
            assert np.all(indata == data)

            indata = col[2:8]
            assert np.all(indata == data[2:8])

            indata = col[2:18:2]
            assert np.all(indata == data[2:18:2])

            ind = [3, 5, 7]
            indata = col[ind]
            assert np.all(indata == data[ind])

            # out of order
            ind = np.arange(num)
            rng.shuffle(ind)
            indata = col[ind]
            assert np.all(indata == data[ind])

            ind = 5
            indata = col[ind]
            assert np.all(indata == data[ind])

            col.append(data)
            assert col.nrows == data.size * 2
