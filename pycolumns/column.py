"""
TODO

    - allow appending without updating index, which we would do at the end
      of a series of appends

"""
import os
import numpy as np
from .filebase import FileBase
from . import util
from . import _column


class Column(FileBase):
    """
    Represent an array column in a Columns database

    Parameters
    ----------
    filename: str, optional
        Path to the file
    name: str, optional
        Name of the column
    dir: str, optional
        Directory of column
    cache_mem: number, optional
        Memory for cache used when creating index in gigabytes. Default 1.0
    verbose: bool, optional
        If set to True print messages

    Construction
    ------------
    # there are alternative construction methods
    # this method determines all info from the full path

    col = ArrayColumn(filename='/full/path')

    # this one uses directory, column name
    col = ArrayColumn(name='something', dir='/path2o/dbname.cols')

    Slice and item lookup
    ---------------------
    # The Column class supports item lookup access, e.g. slices and
    # arrays representing a subset of rows

    col = Column(fname)
    data = col[25:22]

    rows = np.arange(100)
    data = col[rows]

    Indexes on columns
    ------------------
    # create the index
    >>> col.create_index()

    # get indices for some range.  Can also do
    >>> ind = (col > 25)
    >>> ind = col.between(25,35)
    >>> ind = (col == 25)

    # composite searches over multiple columns with the same number
    # of records
    >>> ind = (col1 == 25) & (col2 < 15.23)
    >>> ind = col1.between(15,25) | (col2 != 66)
    >>> ind = col1.between(15,25) & (col2 != 66) & (col3 > 5)
    """
    def __init__(
        self,
        filename=None,
        name=None,
        dir=None,
        cache_mem=1,
        verbose=False,
    ):
        self._do_init(
            filename=filename,
            name=name,
            dir=dir,
            cache_mem=cache_mem,
            verbose=verbose,
        )

    def _do_init(
        self,
        filename=None,
        name=None,
        dir=None,
        verbose=False,
        cache_mem=1,
    ):
        """
        initialize the meta data, and possibly load the mmap
        """

        self._type = 'array'
        self._ext = 1
        self._index_dtype = np.dtype('i8')

        self._cache_mem_gb = float(cache_mem)

        super()._do_init(
            filename=filename,
            name=name,
            dir=dir,
            verbose=verbose,
        )

        # if self.filename is None or not os.path.exists(self.filename):
        if self.filename is None:
            return

        self._open_file()

        # get info for index if it exists
        self._init_index()

    def _open_file(self):
        if os.path.exists(self.filename):
            mode = 'r+'
        else:
            mode = 'w+'

        self._col = _column.Column(
            fname=self.filename,
            mode=mode,
            verbose=self.verbose,
        )

    def _clear(self):
        """
        Clear out all the metadata for this column.
        """

        super()._clear()

        if self.has_data:
            del self._col
            del self._dtype

        self._has_index = False

    @property
    def has_data(self):
        """
        returns True if this column has a header defined, even if there are no
        rows
        """
        has_data = False

        if hasattr(self, '_col'):
            if self._col.has_header():
                has_data = True
        return has_data

    def ensure_has_data(self):
        """
        raise RuntimeError if no data is present
        """
        if not self.has_data:
            raise ValueError('this column has no associated data')

    @property
    def dtype(self):
        """
        get the data type of the column
        """

        self.ensure_has_data()

        if not hasattr(self, '_dtype'):
            self._dtype = np.dtype(self._col.get_dtype())

        return self._dtype

    @property
    def index_dtype(self):
        return self._index_dtype

    @property
    def nrows(self):
        """
        get the number of rows in the column
        """
        self.ensure_has_data()
        return self._col.get_nrows()

    @property
    def size(self):
        """
        get the size of the columns
        """
        return self.nrows

    @property
    def data_size_bytes(self):
        return self.dtype.itemsize * self.nrows

    @property
    def data_size_gb(self):
        return self.data_size_bytes / 1024**3

    @property
    def index_size_bytes(self):
        return self.index_dtype.itemsize * self.size

    @property
    def index_size_gb(self):
        return self.index_size_bytes / 1024**3

    @property
    def cache_mem(self):
        return self._cache_mem_gb

    def _append(self, data):
        """
        Append data to the column.  Data are appended.  If the file has not
        been initialized, the data type is initialized to that of the input
        data.

        Parameters
        ----------
        data: array
            Data to append to the file.  If the column data already exists,
            the data types must match exactly.
        """

        self._check_data(data)

        if self.has_data:
            self._check_data_dtype(data)

        self._col.append(data)

        if self.has_index:
            self._update_index()

    def _check_data(self, data):
        if data.dtype.names is not None:
            raise ValueError('column data cannot have fields')

        if not isinstance(data, np.ndarray):
            raise ValueError(f'data must be a numpy array, got {type(data)}')

        if data.ndim > 1:
            raise ValueError('data must be one dimensional array')

    def _check_data_dtype(self, data):
        dt = data.dtype
        mydt = self.dtype

        if dt != mydt:
            raise ValueError(f"data dtype '{dt}' != '{mydt}'")

    def _get_rec_view(self, data):
        view_dtype = [('data', data.dtype.descr[0][1])]
        return data.view(view_dtype)

    def __getitem__(self, arg):
        """
        Item lookup method, e.g. col[..] meaning slices or
        sequences, etc.
        """

        self.ensure_has_data()

        # converts to slice or Indices and converts stepped slices to
        # arange
        rows = util.extract_rows(arg, self.nrows, sort=True)

        if isinstance(rows, slice):
            # can ignore step since we convert stepped slices to rows in
            # extract_rows
            data = self._col.read_slice(rows)
        else:
            data = self._col.read_rows(rows)

        return data

    def __setitem__(self, arg, values):
        """
        Item lookup method, e.g. col[..] meaning slices or
        sequences, etc.
        """
        raise RuntimeError('fix writing')

    def read(self, rows=None):
        """
        read data from this column

        Parameters
        ----------
        rows: sequence, slice, Indices or None, optional
            A subset of the rows to read.
        """
        return self[rows]

    def _delete(self):
        """
        Attempt to delete the data file associated with this column
        """

        if hasattr(self, '_col'):
            del self._col

        super()._delete()

        # remove index if it exists
        self.delete_index()

    #
    # index related methods
    #

    @property
    def has_index(self):
        """
        returns True if an index exists for this column
        """
        return self._has_index

    def verify_index_available(self):
        """
        raises an exception of the index isn't available
        """
        if not self.has_index:
            raise ValueError('no index exists for column %s' % self.name)

    def create_index(self):
        """
        Create an index for this column.

        Once the index is created, all data from the column will be put into
        the index.  Also, after creation any data appended to the column are
        automatically added to the index.
        """
        import os
        import tempfile
        import shutil

        if self.has_index:
            print('column %s already has an index')
            return

        if self.verbose:
            print('creating index for column %s' % self.name)
            print('index size gb:', self.index_size_gb)
            print('cache mem gb:', self._cache_mem_gb)

        # total usage for in-memory sorting is index size plus twice the size
        # data since we need to do data[s] to get the sorted this is an
        # optimization to not run sort twice
        size_gb = self.index_size_gb + self.data_size_gb * 2

        with tempfile.TemporaryDirectory(dir=self.dir) as tmpdir:
            ifile = os.path.join(
                tmpdir,
                os.path.basename(self.index_filename),
            )
            sfile = os.path.join(
                tmpdir,
                os.path.basename(self.sorted_filename),
            )

            if size_gb < self._cache_mem_gb:
                self._write_index_memory(ifile, sfile)
            else:
                self._write_index_mergesort(tmpdir, ifile, sfile)

            if self.verbose:
                print(f'{ifile} -> {self.index_filename}')
            shutil.move(ifile, self.index_filename)

            if self.verbose:
                print(f'{sfile} -> {self.index_filename}')
            shutil.move(sfile, self.sorted_filename)

        self._init_index()

    def _write_index_memory(self, index_file, sorted_file):
        if self.verbose:
            print(f'creating index for {self.name} in memory')

        data = self[:]
        sort_index = data.argsort()
        data = data[sort_index]

        with _column.Column(
            index_file, mode='w+', verbose=self.verbose,
        ) as indcol:
            indcol.append(sort_index)

        with _column.Column(
            sorted_file, mode='w+', verbose=self.verbose,
        ) as scol:
            scol.append(data)

    def _write_index_mergesort(self, tmpdir, index_file, sorted_file):
        from .mergesort import create_mergesort_index

        if self.verbose:
            print(f'creating index for {self.name} with mergesort on disk')

        chunksize_bytes = int(self._cache_mem_gb * 1024**3)

        bytes_per_element = self.index_dtype.itemsize + self.dtype.itemsize

        # need factor of two because we keep both the cache and the scratch in
        # mergesort
        chunksize = chunksize_bytes // (bytes_per_element * 2)

        create_mergesort_index(
            source=self,
            ifile=index_file,
            sfile=sorted_file,
            chunksize=chunksize,
            tmpdir=tmpdir,
            verbose=self.verbose,
        )

    def update_index(self):
        """
        re-create the index
        """
        self.delete_index()
        self.create_index()

    def delete_index(self):
        """
        Delete the index for this column if it exists
        """
        import os

        if self.has_index:
            index_fname = self.index_filename
            if os.path.exists(index_fname):
                print("Removing index for column: %s" % self.name)
                os.remove(index_fname)

        self._init_index()

    def _init_index(self):
        """
        If index file exists, load some info
        """
        import os
        import numpy as np

        index_fname = self.index_filename
        if os.path.exists(index_fname):
            sort_fname = self.sorted_filename
            if not os.path.exists(sort_fname):
                raise RuntimeError(f'missing sorted file {sort_fname}')

            self._has_index = True
            self._index = _column.Column(index_fname)
            self._sorted = _column.Column(sort_fname)
            self._arr1 = np.zeros(1, dtype=self.dtype)
        else:
            self._has_index = False
            self._index = None
            self._sorted = None

    @property
    def index_filename(self):
        """
        get the filename for the index of this column
        """
        if self.filename is None:
            return None

        if not hasattr(self, '_index_filename'):
            # remove the final extension
            bname = '.'.join(self.filename.split('.')[0:-1])
            self._index_filename = bname+'.index'

        return self._index_filename

    @property
    def sorted_filename(self):
        """
        get the filename for the index of this column
        """
        if self.filename is None:
            return None

        if not hasattr(self, '_sorted_filename'):
            # remove the final extension
            bname = '.'.join(self.filename.split('.')[0:-1])
            self._sorted_filename = bname+'.sort'

        return self._sorted_filename

    def match(self, values):
        """
        get indices of entries that match the input value or values

        Parameters
        ----------
        values: scalar or array
            Value or values to match.  These entries should be
            unique to avoid unexpected results.

        Returns
        -------
        Indices of matches
        """
        import numpy as np
        from .indices import Indices

        values = np.array(values, ndmin=1, copy=False)

        # query each separately
        ind_list = []
        ntot = 0
        for value in values:
            ind = (self == value)
            ntot += ind.size
            if ind.size > 0:
                ind_list.append(ind)

        if len(ind_list) == 1:
            return ind_list[0]

        if len(ind_list) == 0:
            ind_total = np.zeros(0, dtype='i8')
        else:
            ind_total = np.zeros(ntot, dtype='i8')

            start = 0
            for ind in ind_list:
                ind_total[start:start+ind.size] = ind
                start += ind.size

        return Indices(ind_total)

    def __eq__(self, val):
        """
        get exact equality
        """
        return self.between(val, val)

    def _read_one_from_index(self, index):
        arr1 = self._arr1
        self._sorted.read_row(arr1, index)
        val = arr1[0]
        return val

    def _bisect_right(self, val):
        return _bisect_right(
            func=self._read_one_from_index,
            x=val,
            lo=0,
            hi=self.nrows,
        )

    def _bisect_left(self, val):
        return _bisect_left(
            func=self._read_one_from_index,
            x=val,
            lo=0,
            hi=self.nrows,
        )

    # one-sided range operators
    def __gt__(self, val):
        """
        bisect_right returns i such that data[i:] are all strictly > val
        """
        from .indices import Indices

        self.verify_index_available()
        i = self._bisect_right(val)
        indices = self._index.read_slice(slice(i, None))

        return Indices(indices)

    def __ge__(self, val):
        """
        bisect_left returns i such that data[i:] are all strictly >= val
        """
        from .indices import Indices

        self.verify_index_available()

        i = self._bisect_left(val)
        indices = self._index.read_slice(slice(i, None))

        return Indices(indices)

    def __lt__(self, val):
        """
        bisect_left returns i such that data[:i] are all strictly < val
        """
        from .indices import Indices

        self.verify_index_available()

        i = self._bisect_left(val)
        indices = self._index.read_slice(slice(0, i))

        return Indices(indices)

    def __le__(self, val):
        """
        bisect_right returns i such that data[:i] are all strictly <= val
        """
        from .indices import Indices

        self.verify_index_available()

        i = self._bisect_right(val)
        indices = self._index.read_slice(slice(0, i))

        return Indices(indices)

    def between(self, low, high, interval='[]'):
        """
        Find all entries in the range low,high, inclusive by default.

        Using between() requires that an index was created for this column
        using create_index()

        Parameters
        ----------
        low: number
            the lower end of the range.  Must be convertible to the key data
            type.
        high: number
            the upper end of the range.  Must be convertible to the key data
            type.
        interval: str, optional
            '[]': Closed on both sides
            '[)': Closed on the lower side, open on the high side.
            '(]': Open on the lower side, closed on the high side
            '()': Open on both sides.

        examples
        ---------

        # Extract the indices for values in the given range
        query_index = col.between(low, high)

        # Extract from different types of intervals
        values = db.between(low, high,'[]')
        values = db.between(low, high,'[)')
        values = db.between(low, high,'(]')
        values = db.between(low, high,'()')

        # combine with the results of another query and extract the
        # index array
        ind = columns.where( (col1.between(low,high)) & (col2 == value2) )
        """
        from .indices import Indices

        self.verify_index_available()

        if interval == '[]':
            # bisect_left returns i such that data[i:] are all strictly >= val
            ilow = self._bisect_left(low)

            # bisect_right returns i such that data[:i] are all strictly <= val
            ihigh = self._bisect_right(high)

        elif interval == '(]':
            # bisect_right returns i such that data[i:] are all strictly > val
            ilow = self._bisect_right(low)

            # bisect_right returns i such that data[:i] are all strictly <= val
            ihigh = self._bisect_right(high)

        elif interval == '[)':
            # bisect_left returns i such that data[:i] are all strictly >= val
            ilow = self._bisect_left(low)

            # bisect_left returns i such that data[:i] are all strictly < val
            ihigh = self._bisect_left(high)

        elif interval == '()':
            # bisect_right returns i such that data[i:] are all strictly > val
            ilow = self._bisect_right(low)

            # bisect_left returns i such that data[:i] are all strictly < val
            ihigh = self._bisect_left(high)
        else:
            raise ValueError('bad interval type: %s' % interval)

        indices = self._index.read_slice(slice(ilow, ihigh))

        return Indices(indices)

    def _get_repr_list(self, full=False):
        """

        Get a list of metadat for this column.

        """
        indent = '  '

        if not full:
            s = ''
            if self.name is not None:
                s += 'Column: %-15s' % self.name

            s += ' type: %10s' % self.type
            if self.has_data:

                # if self.shape is not None:
                #     s += ' shape: %12s' % (self.shape,)
                s += ' nrows: %12s' % self.nrows

                if self.name is not None:
                    s += ' has index: %s' % self.has_index

            s = [s]
        else:
            s = []
            if self.name is not None:
                s += ['name: %s' % self.name]

            if self.filename is not None:
                s += ['filename: %s' % self.filename]

            s += ['type: array']

            if self.has_data:
                s += ['has index: %s' % self.has_index]

                c_dtype = self.dtype.descr[0][1]
                s += ['dtype: %s' % c_dtype]

                # if self.shape is not None:
                #     s += ['shape: %s' % (self.shape,)]
                s += ['nrows: %s' % self.nrows]

            s = [indent + tmp for tmp in s]
            s = ['Column: '] + s

        return s


def _bisect_right(func, x, lo, hi):
    """
    bisect right with function call to get value
    """

    while lo < hi:
        mid = (lo + hi) // 2
        if x < func(mid):
            hi = mid
        else:
            lo = mid + 1

    return lo


def _bisect_left(func, x, lo, hi):
    """
    bisect left with function call to get value
    """

    while lo < hi:
        mid = (lo + hi) // 2
        if func(mid) < x:
            lo = mid + 1
        else:
            hi = mid

    return lo
