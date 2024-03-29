import os
import numpy as np
from . import _column_pywrap
from . import util


class Column(_column_pywrap.Column):
    """
    A column file, with rows of fixed length records
    """
    def __init__(self, filename, dtype, mode='r', verbose=False):
        self._filename = filename
        self._dtype = np.dtype(dtype)
        self._mode = mode
        self._verbose = verbose
        super().__init__(filename, mode, verbose)
        self._set_nrows()

    @property
    def filename(self):
        """
        get the filename
        """
        return self._filename

    @property
    def mode(self):
        """
        get the file open mode
        """
        return self._mode

    @property
    def dtype(self):
        """
        get the numpy dtype
        """
        return self._dtype

    @property
    def size(self):
        """
        get numbe of rows
        """
        return self._nrows

    @property
    def nrows(self):
        """
        get numbe of rows
        """
        return self._nrows

    @property
    def verbose(self):
        """
        get the filename
        """
        return self._verbose

    def resize(self, nrows):
        """
        Expand or truncate the file to num rows, filling with zeros if needed.

        Parameters
        ----------
        nrows: int
            New number of rows
        """
        if nrows < 0:
            raise ValueError(f'cannot reduce rows to {nrows}')

        if nrows != self.nrows:
            nbytes = nrows * self.dtype.itemsize
            self._resize_bytes(nbytes)
            self._nrows = nrows

    def append(self, data):
        """
        append data to the file
        """
        data = util.get_data_with_conversion(data, self.dtype)

        super()._append(data)

        self._nrows += data.size

    def update_row(self, row, data):
        """
        Update the specified row
        """
        data = util.get_data_with_conversion(data, self.dtype)

        if data.size != 1:
            raise ValueError(
                f'attemting to update row with data of size {data.size}'
            )

        super()._write_at(data, row)

    def write_at(self, data, row):
        """
        write data starting at the specified row
        """
        data = util.get_data_with_conversion(data, self.dtype)

        if row > self.nrows - 1:
            raise IndexError(
                f'attempt to write at row {row} > {self.nrows-1}'
            )

        super()._write_at(data, row)

        if row + data.size > self.nrows:
            self._nrows = row + data.size

    def _fill_slice(self, value, s):
        """
        Fill the slice with the indicated value
        """
        data = util.get_data_with_conversion(value, self.dtype)
        if data.size != 1:
            raise IndexError(
                f'cannot fill with length {data.size}'
            )
        super()._fill_slice(data, s.start, s.stop)

    def _fill_rows(self, data, rows, sortind=None):
        if data.size != 1:
            raise IndexError(
                f'cannot fill with length {data.size}'
            )
        if sortind is not None:
            super()._fill_rows_sortind(data, rows, sortind)
        else:
            super()._fill_rows(data, rows)

    def read_into(self, data):
        """
        read rows from the start of the file, storing in the input array

        Returns
        -------
        data: array
            The output data
        """

        self._check_dtype(data)

        nrows = self.nrows
        if data.size > nrows:
            raise ValueError(
                f'input data rows {data.size} > file nrows {nrows}'
            )
        super()._read_slice(data, 0)

    def read_slice_into(self, data, s):
        """
        read rows into the input data

        Parameters
        ----------
        data: array
            The slice.  Must have a start
        s: slice
            The slice.  Must have a start and no step

        Returns
        -------
        None
        """

        self._check_dtype(data)

        s = util.extract_slice(s, self.nrows)
        nrows = s.stop - s.start
        if data.size != nrows:
            raise ValueError(f'data size {data.size} != slice nrows {nrows}')
        super()._read_slice(data, s.start)

    def read_rows_into(self, data, rows):
        """
        read rows into the input data

        Parameters
        ----------
        data: array
            Array in which to store the values
        rows: array
            The rows array

        Returns
        -------
        None
        """
        from .indices import Indices
        self._check_dtype(data)

        rows_use = util.extract_rows(rows, self.nrows)
        if not isinstance(rows_use, Indices):
            raise ValueError(f'git unexpected rows type {type(rows_use)}')

        super()._read_rows(data, rows_use)
        # super()._read_rows_pages(data, rows)

    def read_row_into(self, data, row):
        """
        read rows into the input data

        Parameters
        ----------
        data: array
            Array in which to store the values. Even though reading
            a single row, still must be length one array not scalar
        rows: array
            The rows array

        Returns
        -------
        None
        """
        self._check_dtype(data)
        if data.ndim == 0:
            raise ValueError('data must have ndim > 0')

        super()._read_row(data, row)
        # super()._read_row_pages(data, rows)

    def __getitem__(self, arg):

        # returns either Indices or slice
        # converts slice with step to indices
        rows = util.extract_rows(arg, self.nrows)

        if isinstance(rows, slice):
            nrows = rows.stop - rows.start
            data = np.empty(nrows, dtype=self.dtype)
            super()._read_slice(data, rows.start)
        else:
            data = np.empty(rows.size, dtype=self.dtype)

            if rows.ndim == 0:
                self._read_row(data, rows)
                data = data[0]
            else:
                sortind = rows.sort_index
                if sortind is None:
                    self._read_rows(data, rows)
                else:
                    self._read_rows_sortind(data, rows, sortind)

        return data

    def __setitem__(self, arg, data):
        # returns either Indices or slice
        # converts slice with step to indices
        rows = util.extract_rows(arg, self.nrows, check_slice_stop=True)

        if isinstance(rows, slice):
            nrows = rows.stop - rows.start
            if np.isscalar(data) or (len(data) == 1 and nrows > 1):
                self._fill_slice(data, rows)
            else:
                if nrows != len(data):
                    raise IndexError(
                        f'mismatch slice size {nrows} and data '
                        f'size {len(data)} when writing'
                    )

                self.write_at(data, rows.start)
        else:
            if rows.ndim == 0:
                self.write_at(data, rows)
            else:
                data = util.get_data_with_conversion(data, self.dtype)

                sortind = rows.sort_index

                # rows are sorted, can check first and last
                if sortind is None:
                    first = rows[0]
                    last = rows[-1]
                else:
                    first = rows[sortind[0]]
                    last = rows[sortind[-1]]

                self._check_row(first)
                self._check_row(last)

                if data.size == 1 and rows.size > 1:
                    self._fill_rows(data, rows, sortind=sortind)
                else:
                    if sortind is None:
                        self._write_rows(data, rows)
                    else:
                        self._write_rows_sortind(data, rows, sortind)

    def _check_row(self, row):
        if row < 0 or row > self.nrows - 1:
            raise IndexError(
                f'row {row} out of bounds [0, {self.nrows-1}]'
            )

    def _check_dtype(self, data):
        dtype = data.dtype
        if dtype != self.dtype:
            raise ValueError(
                f'input dtype {dtype} != file dtype {self.dtype}'
            )

    def _set_nrows(self):
        fsize = os.path.getsize(self.filename)
        elsize = self.dtype.itemsize

        if fsize % elsize != 0:
            raise ValueError(
                f'file size {fsize} not a multiple of element size {elsize}'
            )
        self._nrows = fsize // elsize

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

    def __repr__(self):
        indent = '    '

        if self.dtype.names is not None:
            dt = self.dtype.descr
        else:
            dt = self.dtype.str

        rep = [f'filename: {self.filename}']
        rep += [f'mode: {self.mode}']
        rep += [f'dtype: {dt}']
        rep += [f'nrows: {self.nrows}']

        rep = [indent + r for r in rep]

        rep = ['Column:'] + rep
        return '\n'.join(rep)


def read(fname, dtype, rows=None, verbose=False):
    with Column(fname, dtype=dtype, mode='r', verbose=verbose) as col:
        if rows is None:
            data = col[:]
        else:
            data = col[rows]
    return data


def write(fname, data, append=True, verbose=False):
    if os.path.exists(fname) and append:
        mode = 'r+'
    else:
        mode = 'w+'

    with Column(fname, dtype=data.dtype, mode=mode, verbose=verbose) as col:
        if not col.has_header():
            col.init(data.dtype.str)
        col.append(data)


def append(fname, data, verbose=False):
    if not os.path.exists(fname):
        mode = 'w+'
    else:
        mode = 'r+'

    with Column(fname, dtype=data.dtype, mode=mode, verbose=verbose) as col:
        if not col.has_header():
            col.init(data.dtype.str)
        col.append(data)


def test():
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        data = np.arange(20)
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

            # ind = [3, 5, 7]
            # indata = np.zeros(5, dtype=data.dtype)
            # col.read_rows_into(indata, ind)
            # assert np.all(indata == data[ind])
