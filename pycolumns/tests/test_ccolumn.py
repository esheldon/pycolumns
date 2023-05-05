import pytest


@pytest.mark.parametrize('dtype', ['i8', 'f4', 'U3'])
def test_column(dtype):
    import os
    import tempfile
    import numpy as np
    from .._column import Column

    seed = 333
    num = 20

    rng = np.random.RandomState(seed)

    if dtype[0] == 'i':
        data = np.arange(num, dtype=dtype)
    elif dtype[0] == 'U':
        data = np.zeros(num, dtype=dtype)
        data[:] = [str(i) for i in range(num)]
    else:
        data = rng.uniform(size=num).astype(dtype)

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.col')
        with Column(fname, dtype=data.dtype, mode='w+', verbose=True) as col:
            print('-' * 70)
            print('before append')
            print(col)

            col.append(data)

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
            ind = [7, 3, 5]
            indata = col[ind]
            assert np.all(indata == data[ind])

            ind = 5
            indata = col[ind]
            assert np.all(indata == data[ind])

            s = slice(2, 8)
            indata = np.zeros(s.stop - s.start, dtype=data.dtype)
            col.read_slice_into(indata, s)
            assert np.all(indata == data[s])

            ind = [3, 5, 7]
            indata = np.zeros(len(ind), dtype=data.dtype)
            col.read_rows_into(indata, ind)
            assert np.all(indata == data[ind])

            ind = 6
            indata = np.zeros(1, dtype=data.dtype)
            col.read_row_into(indata, ind)
            assert np.all(indata[0] == data[ind])

            # cannot read into array scalar
            with pytest.raises(ValueError):
                ind = 6
                indata = np.zeros(1, dtype=data.dtype)
                col.read_row_into(indata[0], ind)

            # mismatch in ind size and data size
            with pytest.raises(ValueError):
                ind = [3, 5, 7]
                indata = np.zeros(5, dtype=data.dtype)
                col.read_rows_into(indata, ind)

            # mismatch in ind size and data size
            with pytest.raises(ValueError):
                ind = 3
                indata = np.zeros(5, dtype=data.dtype)
                col.read_row_into(indata, ind)

            col.append(data)
            assert col.nrows == data.size * 2

            #
            # updating column data
            #

            one = data[:1]
            row2update = 3
            col.update_row(row2update, one)
            check = col[3]
            assert check == one[0]

            two = data[3:3+2]
            start = 15
            col.write_at(two, start)
            check = col[start:start+2]
            assert np.all(check == two)

            col[5] = data[0]
            assert np.all(col[5] == data[0])

            col[:data.size] = data
            assert np.all(col[:data.size] == data)

            # sorted rows
            rows = [0, 2, 5]
            wdata = np.array([99, 88, 77], dtype=dtype)
            col[rows] = wdata
            assert np.all(col[rows] == wdata)

            # out of order rows
            rows = [8, 3, 7]
            wdata = np.array([100, 102, 103], dtype=dtype)
            col[rows] = wdata
            assert np.all(col[rows] == wdata)

            rows = slice(0, 6, 2)
            wdata = np.arange(0, 6, 2).astype(dtype)
            col[rows] = wdata
            assert np.all(col[rows] == wdata)

            # bad slice size
            with pytest.raises(IndexError):
                col[0:5] = wdata

            # out of bounds update
            with pytest.raises(IndexError):
                col[np.arange(100)] = np.arange(100)
