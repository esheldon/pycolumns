"""
TODO

    - allow appending without updating index, which we would do at the end
      of a series of appends

"""
import os
import bisect
import numpy as np

from . import util
from .sfile import SimpleFile
from .indices import Indices


class ColumnBase(object):
    """
    Represent a column in a Columns database.  Facilitate opening, reading,
    writing of data.  This class can be instantiated alone, but is usually
    accessed through the Columns class.
    """
    def __init__(
        self,
        filename=None,
        name=None,
        dir=None,
        verbose=False,
    ):

        self._do_init(
            filename=filename,
            name=name,
            dir=dir,
            verbose=verbose,
        )

    def _do_init(
        self,
        filename=None,
        name=None,
        dir=None,
        verbose=False,
    ):
        """
        See main docs for the Column class
        """

        self._clear()

        self._filename = filename
        self._name = name
        self._dir = dir
        self._verbose = verbose

        # make sure we have something to work with before continuing
        if not self._init_args_sufficient(filename, dir, name):
            return

        if self.filename is not None:
            self._set_meta_from_filename()

        elif self.name is not None:
            self._set_meta_from_name()

    @property
    def verbose(self):
        return self._verbose

    @property
    def dir(self):
        """
        get the directory holding the file
        """
        return self._dir

    @property
    def filename(self):
        """
        get the file name holding the column data
        """
        return self._filename

    @property
    def name(self):
        """
        get the name type of the column
        """
        return self._name

    @property
    def type(self):
        """
        get the data type of the column
        """
        return self._type

    def reload(self):
        """
        Just reload the metadata for this column
        """
        if self.verbose:
            print('Reloading column metadata for: %s' % self.name)
        self._do_init(filename=self.filename, verbose=self.verbose)

    def _clear(self):
        """
        Clear out all the metadata for this column.
        """

        self._filename = None
        self._name = None
        self._dir = None

    def read(self, *args):
        """
        Read data from a column
        """
        raise NotImplementedError('implement read')

    #
    # setup methods
    #

    def _set_meta_from_filename(self):
        """
        Initiaize this column based on the pull path filename
        """
        if self.filename is not None:
            if self.verbose:
                print("Initializing from file: %s" % self.filename)

            if self.filename is None:
                raise ValueError("You haven't specified a filename yet")

            self._name = util.extract_colname(self.filename)
        else:
            raise ValueError("You must set filename to use this function")

    def _set_meta_from_name(self):
        """
        Initialize this based on name.  The filename is constructed from
        the dir and name
        """

        if self.name is not None and self.dir is not None:

            if self.verbose:
                mess = "Initalizing from \n\tdir: %s \n\tname: %s \n\ttype: %s"
                print(mess % (self.dir, self.name, self.type))

            self._filename = util.create_filename(
                self.dir,
                self.name,
                self.type,
            )
        else:
            raise ValueError("You must set dir,name,type to use this function")

    def _get_repr_list(self, full=False):
        """

        Get a list of metadat for this column.

        """
        raise NotImplementedError('implemente _get_repr_list')

    def __repr__(self):
        """
        Print out some info about this column
        """
        s = self._get_repr_list(full=True)
        s = "\n".join(s)
        return s

    def _init_args_sufficient(self,
                              filename=None,
                              dir=None,
                              name=None):
        """
        Determine if the inputs are enough for initialization
        """
        if (filename is None) and \
                (dir is None or name is None):
            return False
        else:
            return True


class ArrayColumn(ColumnBase):
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

        self._cache_mem_gb = float(cache_mem)

        super()._do_init(
            filename=filename,
            name=name,
            dir=dir,
            verbose=verbose,
        )

        if self.filename is None or not os.path.exists(self.filename):
            return

        self._open_file('r+')

        # get info for index if it exists
        self._init_index()

    def _open_file(self, mode):
        self._sf = SimpleFile(self.filename, mode=mode)

    def _clear(self):
        """
        Clear out all the metadata for this column.
        """

        super()._clear()

        if self.has_data:
            del self._sf

        self._has_index = False

    @property
    def has_data(self):
        """
        returns True if this column has some data
        """
        return hasattr(self, '_sf')

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

        return self._sf.dtype

    @property
    def index_dtype(self):
        return get_index_dtype(self.dtype)

    @property
    def shape(self):
        """
        get the shape of the column
        """
        self.ensure_has_data()

        return self._sf.shape

    @property
    def nrows(self):
        """
        get the shape of the column
        """
        self.ensure_has_data()

        return self._sf.shape[0]

    @property
    def size(self):
        """
        get the size of the columns
        """
        self.ensure_has_data()

        return self._sf.size

    @property
    def data_size_bytes(self):
        return self.dtype.itemsize * self.size

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
        Append data to the column.  Data are appended if the file already
        exists.

        Parameters
        ----------
        data: array
            Data to append to the file.  If the column data already exists,
            the data types must match exactly.
        """

        # make sure the data type of the input equals that of the column
        if not isinstance(data, np.ndarray):
            raise ValueError("For 'array' columns data must be a numpy array")

        if data.dtype.names is not None:
            raise ValueError('do not enter data with fields')

        if not self.has_data:
            self._open_file('w+')

        self._sf.write(data)

        if self.has_index:
            self._update_index()

    def __getitem__(self, arg):
        """
        Item lookup method, e.g. col[..] meaning slices or
        sequences, etc.
        """
        if not hasattr(self, '_sf'):
            raise ValueError('no file loaded yet')

        use_arg = util.extract_rows(arg, sort=True)
        return self._sf[use_arg]

    def __setitem__(self, arg, values):
        """
        Item lookup method, e.g. col[..] meaning slices or
        sequences, etc.
        """
        if not hasattr(self, '_sf'):
            raise ValueError('no file loaded yet')

        self._sf[arg] = values

    def read(self, rows=None):
        """
        read data from this column

        Parameters
        ----------
        rows: sequence, slice, Indices or None, optional
            A subset of the rows to read.
        """
        if not hasattr(self, '_sf'):
            raise ValueError('no file loaded yet')

        return self[rows]
        # if rows is None:
        #     return self._sf[:]
        # else:
        #     use_rows = util.extract_rows(rows, sort=True)
        #     return self._sf[use_rows]

    def _delete(self):
        """
        Attempt to delete the data file associated with this column
        """

        if hasattr(self, '_sf'):
            del self._sf

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
        import tempfile
        import shutil

        if self.has_index:
            print('column %s already has an index')
            return

        if len(self.shape) > 1:
            raise ValueError('cannot index a multi-dimensional column')

        if self.verbose:
            print('creating index for column %s' % self.name)
            print('index size gb:', self.index_size_gb)
            print('cache mem gb:', self._cache_mem_gb)

        with tempfile.TemporaryDirectory(dir=self.dir) as tmpdir:
            tfile = os.path.join(
                tmpdir,
                os.path.basename(self.index_filename),
            )
            if self.index_size_gb < self._cache_mem_gb:
                self._write_index_memory(tfile)
            else:
                self._write_index_mergesort(tmpdir, tfile)

            if self.verbose:
                print(f'{tfile} -> {self.index_filename}')

            shutil.move(tfile, self.index_filename)

        self._init_index()

    def _write_index_memory(self, fname):
        if self.verbose:
            print(f'creating index for {self.name} in memory')

        dt = self.index_dtype
        index_data = np.zeros(self.shape[0], dtype=dt)
        index_data['index'] = np.arange(index_data.size)
        index_data['value'] = self[:]

        index_data.sort(order='value')

        # set up file name and data type info
        with SimpleFile(fname, mode='w+') as sf:
            sf.write(index_data)

    def _write_index_mergesort(self, tmpdir, fname):
        from .mergesort import create_mergesort_index

        if self.verbose:
            print(f'creating index for {self.name} with mergesort on disk')

        chunksize_bytes = int(self._cache_mem_gb * 1024**3)

        bytes_per_element = self.index_dtype.itemsize
        chunksize = chunksize_bytes // bytes_per_element

        create_mergesort_index(
            infile=self.filename,
            outfile=fname,
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
        index_fname = self.index_filename
        if os.path.exists(index_fname):
            self._has_index = True
            self._index = SimpleFile(index_fname, mode='r+')
        else:
            self._has_index = False
            self._index = None

    @property
    def index_filename(self):
        """
        get the filename for the index of this column
        """
        if self.filename is None:
            return None

        # remove the final extension
        index_fname = '.'.join(self.filename.split('.')[0:-1])
        index_fname = index_fname+'__index.sf'
        return index_fname

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

    # one-sided range operators
    def __gt__(self, val):
        """
        bisect_right returns i such that data[i:] are all strictly > val
        """
        self.verify_index_available()

        mmap = self._index.mmap
        i = bisect.bisect_right(mmap['value'], val)
        indices = mmap['index'][i:].copy()

        return Indices(indices)

    def __ge__(self, val):
        """
        bisect_left returns i such that data[i:] are all strictly >= val
        """
        self.verify_index_available()

        mmap = self._index.mmap
        i = bisect.bisect_left(mmap['value'], val)
        indices = mmap['index'][i:].copy()

        return Indices(indices)

    def __lt__(self, val):
        """
        bisect_left returns i such that data[:i] are all strictly < val
        """
        self.verify_index_available()

        mmap = self._index.mmap
        i = bisect.bisect_left(mmap['value'], val)
        indices = mmap['index'][:i].copy()

        return Indices(indices)

    def __le__(self, val):
        """
        bisect_right returns i such that data[:i] are all strictly <= val
        """
        self.verify_index_available()

        mmap = self._index.mmap
        i = bisect.bisect_right(mmap['value'], val)
        indices = mmap['index'][:i].copy()

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

        self.verify_index_available()

        mmap = self._index.mmap
        if interval == '[]':
            # bisect_left returns i such that data[i:] are all strictly >= val
            ilow = bisect.bisect_left(mmap['value'], low)

            # bisect_right returns i such that data[:i] are all strictly <= val
            ihigh = bisect.bisect_right(mmap['value'], high)

        elif interval == '(]':
            # bisect_right returns i such that data[i:] are all strictly > val
            ilow = bisect.bisect_right(mmap['value'], low)

            # bisect_right returns i such that data[:i] are all strictly <= val
            ihigh = bisect.bisect_right(mmap['value'], high)

        elif interval == '[)':
            # bisect_left returns i such that data[:i] are all strictly >= val
            ilow = bisect.bisect_left(mmap['value'], low)

            # bisect_left returns i such that data[:i] are all strictly < val
            ihigh = bisect.bisect_left(mmap['value'], high)

        elif interval == '()':
            # bisect_right returns i such that data[i:] are all strictly > val
            ilow = bisect.bisect_right(mmap['value'], low)

            # bisect_left returns i such that data[:i] are all strictly < val
            ihigh = bisect.bisect_left(mmap['value'], high)
        else:
            raise ValueError('bad interval type: %s' % interval)

        indices = mmap['index'][ilow:ihigh].copy()

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

            if self.shape is not None:
                s += ' shape: %12s' % (self.shape,)

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

            if self.shape is not None:
                s += ['shape: %s' % (self.shape,)]

            if self.name is not None:
                s += ['has index: %s' % self.has_index]

            if self.dtype is not None:
                c_dtype = self.dtype.descr[0][1]
                s += ["dtype: %s" % c_dtype]

            s = [indent + tmp for tmp in s]
            s = ['Column: '] + s

        return s


class DictColumn(ColumnBase):
    def _do_init(
        self,
        filename=None,
        name=None,
        dir=None,
        verbose=False,
    ):

        self._type = 'dict'

        super()._do_init(
            filename=filename,
            name=name,
            dir=dir,
            verbose=verbose,
        )

    def write(self, data):
        """
        Write data to the dict column.

        Parameters
        ----------
        data: dict or json supported object
            The data must be supported by the JSON format.
        """

        util.write_json(data, self.filename)

    def read(self):
        """
        read data from this column
        """

        return util.read_json(self.filename)

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

            s = [s]
        else:
            s = []
            if self.name is not None:
                s += ['name: %s' % self.name]

            if self.filename is not None:
                s += ['filename: %s' % self.filename]

            s += ['type: dict']

            s = [indent + tmp for tmp in s]
            s = ['Column: '] + s

        return s


def get_index_dtype(dtype):
    return np.dtype([
        ('index', 'i8'),
        ('value', dtype.descr[0][1]),
    ])


def _do_test_create_index(tmpdir, cache_mem, seed=999, num=1_000_000):
    import os
    import numpy as np
    from . import sfile
    from .columns import Columns

    cdir = os.path.join(tmpdir, 'test.cols')
    cols = Columns(cdir, cache_mem=cache_mem, verbose=True)

    rng = np.random.RandomState(seed)
    data = np.zeros(num, dtype=[('rand', 'f8')])
    data['rand'] = rng.uniform(size=num)

    cols.append(data)
    cols['rand'].create_index()
    ifile = cols['rand'].index_filename
    idata = sfile.read(ifile)

    s = data['rand'].argsort()
    assert np.all(idata['value'] == data['rand'][s])


def test_create_index(cache_mem=0.01, seed=999, num=1_000_000, keep=False):
    import tempfile
    if keep:
        _do_test_create_index(
            tmpdir='.', cache_mem=cache_mem, seed=seed, num=num,
        )
    else:
        with tempfile.TemporaryDirectory(dir='.') as tmpdir:
            _do_test_create_index(
                tmpdir=tmpdir, cache_mem=cache_mem, seed=seed, num=num,
            )
