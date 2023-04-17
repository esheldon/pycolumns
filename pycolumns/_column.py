import os
import numpy as np
from . import _column_pywrap


class Column(_column_pywrap.Column):
    """
    A column file, with rows of fixed length records
    """
    def __init__(self, fname, mode='r', dtype=None, verbose=False):
        super().__init__(fname, mode, verbose)
        if mode[0] == 'w':
            # this is the initial creation of the file, we can go
            # ahead and init if the dtype was sent
            if dtype is not None:
                self.init(dtype)

    def __enter__(self):
        return self

    def read(self):
        """
        read all rows, creating the output

        Returns
        -------
        data: array
            The output data
        """
        data = np.empty(self.get_nrows(), dtype=self.get_dtype())
        super()._read_slice(data, 0)
        return data

    def read_into(self, data):
        """
        read rows, storing in the input array

        Returns
        -------
        data: array
            The output data
        """
        nrows = self.get_nrows()
        if not data.size == nrows:
            raise ValueError(f'data rows {data.size} != file nrows {nrows}')
        super()._read_slice(data, 0)

    def read_rows(self, rows):
        """
        read rows, creating the output

        Parameters
        ----------
        rows: array
            The rows array

        Returns
        -------
        data: array
            The output data
        """
        data = np.empty(rows.size, dtype=self.get_dtype())
        super()._read_rows(data, rows)
        return data

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
        super()._read_rows(data, rows)

    def read_slice(self, s):
        """
        read slice, creating the output

        Parameters
        ----------
        s: slice
            The slice.  Must have a stop and start and no step

        Returns
        -------
        data: array
            The output data
        """
        start, stop = self._extract_slice_start_stop(s)
        nrows = stop - start
        data = np.empty(nrows, dtype=self.get_dtype())
        super()._read_slice(data, start)
        return data

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
        start, stop = self._extract_slice_start_stop(s)
        nrows = stop - start
        if data.size != nrows:
            raise ValueError(f'data size {data.size} != slice nrows {nrows}')
        super()._read_slice(data, start)

    def append(self, data):
        """
        append data to the file
        """
        if not self.has_header():
            self.init(data.dtype)

        super()._append(data)

    def init(self, dtype):
        """
        Initialize the header with the input data type
        """
        dtype = np.dtype(dtype)
        super().init(dtype.str)

    def _extract_slice_start_stop(self, s):
        nrows = self.get_nrows()

        start = s.start
        if start is None:
            start = 0
        stop = s.stop
        if stop is None:
            stop = nrows
        elif stop > nrows:
            stop = nrows
        return start, stop

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()


def read(fname, rows=None, verbose=False):
    with Column(fname, mode='r', verbose=verbose) as col:
        if rows is None:
            data = col.read()
        elif isinstance(rows, slice):
            data = col.read_slice()
        else:
            data = col.read_rows(rows)
    return data


def write(fname, data, append=True, verbose=False):
    if os.path.exists(fname) and append:
        mode = 'r+'
    else:
        mode = 'w+'

    with Column(fname, mode=mode, verbose=verbose) as col:
        if not col.has_header():
            col.init(data.dtype.str)
        col.append(data)


def append(fname, data, verbose=False):
    if not os.path.exists(fname):
        mode = 'w+'
    else:
        mode = 'r+'

    with Column(fname, mode=mode, verbose=verbose) as col:
        if not col.has_header():
            col.init(data.dtype.str)
        col.append(data)
