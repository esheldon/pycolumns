import numpy as np
from . import util
from ._column import Column as CColumn
from .chunks import Chunks
from .defaults import DEFAULT_CACHE_MEM


class Column(object):
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
    cache_mem: str or number
        Cache memory for index creation, default '1g' or one gigabyte.
        Can be a number in gigabytes or a string
        Strings should be like '{amount}{unit}'
            '1g' = 1 gigabytes
            '100m' = 100 metabytes
            '1000k' = 1000 kilobytes
        Units can be g, m or k, case insenitive
    verbose: bool, optional
        If set to True print messages

    Construction
    ------------
    # there are alternative construction methods
    # this method determines all info from the full path

    col = Column(filename='/full/path')

    # this one uses directory, column name
    col = Column(name='something', dir='/path2o/dbname.cols')

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
        metafile,
        cache_mem=DEFAULT_CACHE_MEM,
        verbose=False,
    ):
        """
        initialize the meta data, and possibly load the mmap
        """

        self._meta_filename = metafile
        self._cache_mem = cache_mem
        self._cache_mem_gb = util.convert_to_gigabytes(cache_mem)
        self._verbose = verbose
        self.reload()

    def reload(self):
        """
        load, or reload all meta data and reopen/open files
        """
        self._meta = util.read_json(self.meta_filename)

        path_info = util.meta_to_colfiles(self.meta_filename)
        self._array_filename = path_info['array']
        self._index_filename = path_info['index']
        self._index1_filename = path_info['index1']
        self._sorted_filename = path_info['sorted']
        self._chunks_filename = path_info['chunks']
        self._name = path_info['name']
        self._dir = path_info['dir']

        self._type = 'col'
        self._ext = 1
        self._dtype = np.dtype(self._meta['dtype'])
        self._index_dtype = np.dtype('i8')
        self._index1_dtype = np.dtype([('index', 'i8'), ('value', self.dtype)])

        self._open()

        # get info for index if it exists
        self._init_index()

    @property
    def verbose(self):
        return self._verbose

    @property
    def name(self):
        """
        get the name type of the column
        """
        return self._name

    @property
    def dir(self):
        """
        get the directory holding the file
        """
        return self._dir

    @property
    def type(self):
        """
        get the data type of the column
        """
        return self._type

    @property
    def filenames(self):
        return [
            self.meta_filename,
            self.array_filename,
            self.index_filename,
            self.index1_filename,
            self.sorted_filename,
            self.chunks_filename,
        ]
    @property
    def meta_filename(self):
        """
        get the filename for the array of this column
        """
        return self._meta_filename

    @property
    def array_filename(self):
        """
        get the filename for the array of this column
        """
        return self._array_filename

    @property
    def chunks_filename(self):
        """
        get the filename for the chunks for this column
        """
        return self._chunks_filename

    @property
    def index_filename(self):
        """
        get the filename for the index of this column
        """
        return self._index_filename

    @property
    def index1_filename(self):
        """
        get the filename for the hierarch 1 index of this column
        """
        return self._index1_filename

    @property
    def sorted_filename(self):
        """
        get the filename for the sorted version of this column
        """
        return self._sorted_filename

    @property
    def meta(self):
        """
        get a copy of the meta data dict
        """
        return self._meta.copy()

    @property
    def dtype(self):
        """
        get the data type of the column
        """
        return self._dtype
        # return self._meta['dtype']

    @property
    def index_dtype(self):
        return self._index_dtype

    @property
    def index1_dtype(self):
        return self._index1_dtype

    @property
    def nrows(self):
        """
        get the number of rows in the column
        """
        return self._col.nrows

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
        return self._cache_mem

    @property
    def cache_mem_gb(self):
        return self._cache_mem_gb

    def _open(self):
        meta = self._meta
        if 'compression' in meta and meta['compression']:
            self._col = Chunks(
                filename=self.array_filename,
                chunks_filename=self.chunks_filename,
                dtype=self._meta['dtype'],
                mode='r+',
                compression=meta['compression'],
                chunksize=meta['chunksize'],
                verbose=self.verbose,
            )
        else:
            self._col = CColumn(
                self.array_filename,
                dtype=self._meta['dtype'],
                mode='r+',
                # verbose=self.verbose,
            )

    def _append(self, data, update_index=True):
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

        # we use data conversion in self._col
        # self._check_data_dtype(data)
        self._col.append(data)

        if update_index:
            self.update_index()

    def _check_data(self, data):
        if data.dtype.names is not None:
            raise ValueError('column data cannot have fields')

        if not isinstance(data, np.ndarray):
            raise ValueError(f'data must be a numpy array, got {type(data)}')

        if data.ndim > 1:
            raise ValueError('data must be one dimensional array')

    # def _check_data_dtype(self, data):
    #     dt = data.dtype
    #     mydt = self.dtype
    #
    #     if dt != mydt:
    #         raise ValueError(f"data dtype '{dt}' != '{mydt}'")

    def _get_rec_view(self, data):
        view_dtype = [('data', data.dtype.descr[0][1])]
        return data.view(view_dtype)

    def __getitem__(self, arg):
        """
        Item lookup method, e.g. col[..] meaning slices or
        sequences, etc.
        """
        return self._col[arg]

    def __setitem__(self, arg, values):
        """
        Item lookup method, e.g. col[..] meaning slices or
        sequences, etc.
        """
        self._col[arg] = values

        # todo, context manager for this so only updates index after leaving
        # context
        self.update_index()

    def read(self, rows=None):
        """
        read data from this column

        Parameters
        ----------
        rows: sequence, slice, Indices or None, optional
            A subset of the rows to read.
        """
        return self._col[rows]

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

    def create_index(self, overwrite=False):
        """
        Create an index for this column.

        Parameters
        ----------
        overwrite: bool, optional
            If set to True, overwrite any existing index.  See also
            update_index()
        """
        import os
        from tempfile import TemporaryDirectory
        import shutil

        if self.has_index and not overwrite:
            raise RuntimeError(
                f'column {self.name} already has an index.  Send '
                f' overwrite=True or use update_index()'
            )

        self.delete_index()

        # total usage for in-memory sorting is index size plus twice the size
        # data since we need to do data[s] to get the sorted this is an
        # optimization to not run sort twice
        size_gb = self.index_size_gb + self.data_size_gb * 2

        if self.verbose:
            print('creating index for column', self.name)
            print(f'cache mem: {self.cache_mem}')
            print(f'required gb: {size_gb:.3g}')

        with TemporaryDirectory(dir=self.dir) as tmpdir:
            ifile = os.path.join(
                tmpdir,
                os.path.basename(self.index_filename),
            )
            sfile = os.path.join(
                tmpdir,
                os.path.basename(self.sorted_filename),
            )
            with CColumn(ifile, mode='w+', dtype=self.index_dtype) as ifobj:
                with CColumn(sfile, mode='w+', dtype=self.dtype) as sfobj:

                    if size_gb < self.cache_mem_gb:
                        self._write_index_memory(ifobj, sfobj)
                    else:
                        self._write_index_mergesort(tmpdir, ifobj, sfobj)

                    if self.verbose:
                        print(f'  {ifile} -> {self.index_filename}')
                    shutil.move(ifile, self.index_filename)

                    if self.verbose:
                        print(f'  {sfile} -> {self.sorted_filename}')
                    shutil.move(sfile, self.sorted_filename)

        self._write_index1()
        self._init_index()

    def _write_index_memory(self, ifobj, sfobj):
        if self.verbose:
            print(f'creating index for {self.name} in memory')

        data = self[:]
        if self.verbose:
            print('  sorting')

        sort_index = data.argsort()

        if self.verbose:
            print('  writing')

        ifobj.append(sort_index)
        sfobj.append(data[sort_index])

    def _write_index_mergesort(self, tmpdir, ifobj, sfobj):
        from .mergesort import create_mergesort_index

        if self.verbose:
            print(f'creating index for {self.name} with mergesort on disk')

        chunksize_bytes = int(self.cache_mem_gb * 1024**3)

        bytes_per_element = self.index_dtype.itemsize + self.dtype.itemsize

        # need factor of two because we keep both the cache and the scratch in
        # mergesort
        chunksize = chunksize_bytes // (bytes_per_element * 2)

        create_mergesort_index(
            source=self,
            isink=ifobj,
            ssink=sfobj,
            chunksize=chunksize,
            tmpdir=tmpdir,
            verbose=self.verbose,
        )

    def _write_index1(self, chunksize_rows=10_000):
        """
        this needs to be worked out much better
        """
        nchunks = self.nrows // chunksize_rows
        if self.nrows % chunksize_rows != 0:
            nchunks += 1

        if nchunks < 10:
            rows = np.arange(self.nrows)
        else:
            rows = np.ones(nchunks+1, dtype='i8')
            rows[0:-1] = np.arange(0, self.nrows, chunksize_rows)
            rows[-1] = self.nrows - 1

        output = np.zeros(rows.size, dtype=self.index1_dtype)
        output['index'] = rows

        if self.verbose:
            print('  reading index1 values from:', self.sorted_filename)
        with CColumn(self.sorted_filename, dtype=self.dtype) as scol:
            output['value'] = scol[rows]

        if self.verbose:
            print('  writing:', self.index1_filename)
        with CColumn(
                self.index1_filename,
                mode='w+',
                dtype=self.index1_dtype) as i1col:
            i1col.append(output)

    def update_index(self):
        """
        Recreate the index for this column.
        """
        if self.has_index:
            self.create_index(overwrite=True)

    def delete_index(self):
        """
        Delete the index for this column if it exists
        """
        import os

        if self.has_index:
            names = [
                self.index_filename,
                self.index1_filename,
                self.sorted_filename,
            ]
            for name in names:
                if os.path.exists(name):
                    print("Removing: %s" % name)
                    os.remove(name)

        self._init_index()

    def _init_index(self):
        """
        If index file exists, load some info
        """
        import os
        import numpy as np

        self._arr1 = np.zeros(1, dtype=self.dtype)

        if os.path.exists(self.index_filename):
            if not os.path.exists(self.sorted_filename):
                raise RuntimeError(
                    f'missing sorted file {self.sorted_filename}'
                )

            if not os.path.exists(self.index_filename):
                raise RuntimeError(
                    f'missing index file {self.index_filename}'
                )

            if not os.path.exists(self.index1_filename):
                raise RuntimeError(
                    f'missing index1 file {self.index1_filename}'
                )

            self._has_index = True
            self._index = CColumn(self.index_filename, dtype=self.index_dtype)
            self._index1 = CColumn(
                self.index1_filename, dtype=self.index1_dtype,
            )
            self._sorted = CColumn(self.sorted_filename, dtype=self.dtype)

            self._index1_data = self._index1[:]
        else:
            self._has_index = False
            self._index = None
            self._index1 = None
            self._index1_data = None
            self._sorted = None

    def match(self, values):
        """
        get indices of entries that match the input value or values

        Parameters
        ----------
        values: scalar or array
            Value or values to match.

        Returns
        -------
        Indices of matches
        """
        import numpy as np
        from .indices import Indices

        # this makes an array
        values = np.unique(values)

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
        self._sorted._read_row(arr1, index)
        val = arr1[0]
        return val

    def _bisect_right(self, val):
        """
        bisect on the hierarch index1 then possibly the full index
        """
        import bisect
        idata = self._index1_data
        i1 = bisect.bisect_right(idata['value'], x=val)

        if i1 == 0 or i1 == idata.size:
            sind = i1
        else:
            start = idata['index'][i1 - 1]
            end = idata['index'][i1]

            sind = _bisect_right_func(
                func=self._read_one_from_index,
                x=val,
                lo=start,
                hi=end,
            )

        return sind

    def _bisect_left(self, val):
        """
        bisect on the hierarch index1 then possibly the full index
        """
        import bisect
        idata = self._index1_data
        i1 = bisect.bisect_left(idata['value'], x=val)

        if i1 == 0 or i1 == idata.size:
            sind = i1
        else:
            start = idata['index'][i1 - 1]
            end = idata['index'][i1]

            sind = _bisect_left_func(
                func=self._read_one_from_index,
                x=val,
                lo=start,
                hi=end,
            )

        return sind

    # one-sided range operators
    def __gt__(self, val):
        """
        bisect_right returns i such that data[i:] are all strictly > val
        """
        from .indices import Indices

        self.verify_index_available()
        i = self._bisect_right(val)
        indices = self._index[i:]

        return Indices(indices)

    def __ge__(self, val):
        """
        bisect_left returns i such that data[i:] are all strictly >= val
        """
        from .indices import Indices

        self.verify_index_available()

        i = self._bisect_left(val)
        indices = self._index[i:]

        return Indices(indices)

    def __lt__(self, val):
        """
        bisect_left returns i such that data[:i] are all strictly < val
        """
        from .indices import Indices

        self.verify_index_available()

        i = self._bisect_left(val)
        indices = self._index[:i]

        return Indices(indices)

    def __le__(self, val):
        """
        bisect_right returns i such that data[:i] are all strictly <= val
        """
        from .indices import Indices

        self.verify_index_available()

        i = self._bisect_right(val)
        indices = self._index[:i]

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

        indices = self._index[ilow:ihigh]

        return Indices(indices)

    def __repr__(self):
        indent = '  '

        s = []
        if self.name is not None:
            s += ['name: %s' % self.name]

        s += ['filename: %s' % self.array_filename]
        s += ['type: array']
        s += ['index: %s' % self.has_index]

        if 'compression' in self.meta and self.meta['compression']:
            c = self.meta['compression']
            s += ['compression:']
            s += ['    cname: %s' % c['cname']]
            s += ['    clevel: %s' % c['clevel']]
            s += ['    shuffle: %s' % c['shuffle']]
            s += ['chunksize: %s' % self._col.chunksize]

        c_dtype = self.dtype.descr[0][1]
        s += ['dtype: %s' % c_dtype]

        s += ['nrows: %s' % self.nrows]

        s = [indent + tmp for tmp in s]
        s = ['Column: '] + s

        s = "\n".join(s)
        return s


def _bisect_right_func(func, x, lo, hi):
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


def _bisect_left_func(func, x, lo, hi):
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
