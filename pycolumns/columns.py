import os
from glob import glob
import pprint
import json
import numpy as np
import shutil
from .sfile import SimpleFile

try:
    import numpydb
    havedb = True
except ImportError:
    havedb = False

ALLOWED_COL_TYPES = ['array', 'json', 'cols']


class Columns(dict):
    """
    Manage a database of "columns" represented by simple flat files.  This
    design is chosen to maximize efficiency of memory and speed.
    Transactional robustness is not yet a priority.

    If numpydb is available, indexes can be created for columns.  This
    facilitates fast searching.

    Construction
    ------------
        >>> coldir='/some/path/mycols.cols
        >>> c=Columns(coldir)

    Examples
    ---------
        # construct a column database from the specified coldir
        >>> coldir='/some/path/mydata.cols'
        >>> c=Columns(coldir)

        # display some info about the columns
        >>> c
        Column Directory:

          dir: /some/path/mydata.cols
          Columns:
            name             type  dtype index  size
            --------------------------------------------------
            ccd               col    <i2 True   64348146
            dec               col    <f8 False  64348146
            exposurename      col   |S20 True   64348146
            id                col    <i4 False  64348146
            imag              col    <f4 False  64348146
            ra                col    <f8 False  64348146
            x                 col    <f4 False  64348146
            y                 col    <f4 False  64348146

          Sub-Column Directories:
            name
            --------------------------------------------------
            psfstars

        # display info about column 'id'
        >>> c['id']
        Column:
          "id"
          filename: ./id.col
          type: array
          size: 64348146
          has index: False
          dtype:
            [('id', '<i4')]


        # get the column names
        >>> c.colnames()
        ['ccd','dec','exposurename','id','imag','ra','x','y']

        # reload all columns or specified column/column list
        >>> c.reload(name=None)

        # read all data from column 'id'
        # alternative syntaxes
        >>> id = c['id'][:]
        >>> id = c['id'].read()
        >>> id = c.read_column('id')

        # read a subset of rows
        # slicing
        >>> id = c['id'][25:125]

        # specifying a set of rows
        >>> rows=[3,225,1235]
        >>> id = c['id'][rows]
        >>> id = c.read_column('id', rows=rows)


        # read multiple columns into a single rec array
        >>> data = c.read(columns=['id','flux'], rows=rows)

        # or put different columns into fields of a dictionary instead of
        # packing them into a single array.  This allows reading from
        # different-length columns and from dict types
        >>> data = c.read(columns=['id','flux'], asdict=True)

        # If numpydb is available, you can create indexes and
        # perform fast searching
        >>> c['x'].create_index()

        # get indices for some range.  Can also do
        >>> ind=(c['x'] > 25)
        >>> ind=c['x'].between(25,35)
        >>> ind=(c['x'] == 25)
        >>> ind=c['x'].match([25,77])

        # composite searches over multiple columns
        >>> ind = (c['col1'] == 25) & (col['col2'] < 15.23)
        >>> ind = c['col1'].between(15,25) | (c['col2'] != 66)
        >>> ind = (
            c['col1'].between(15,25) &
            (c['col2'] != 66) &
            (c['col3'] < 5)
        )

        # create column or append data to a column
        >>> c.write_column(name, data)

        # append to existing column, alternative syntax
        >>> c['id'].write(data)

        # write multiple columns from the fields in a rec array
        # names in the data correspond to column names
        >>> data = np.zeros(num, dtype=[('ra','f8'), ('dec','f8')])
        >>> c.write(data)

        # write/append data from the fields in a .fits file
        >>> c.from_fits(fitsfile_name)
    """

    def __init__(self, dir=None, verbose=False):
        self.verbose = verbose
        self.init(dir=dir)

    def init(self, dir=None):
        """
        Initialize the database.  This will create a new db directory if none
        exists.  If it exists and contains column files their metadata will be
        loaded.
        """
        self._set_dir(dir)
        if self.dir_exists():
            self.load()

    @property
    def dir(self):
        return self._dir

    def _set_dir(self, dir=None):
        """
        Set the database directory, creating if none exists.
        """
        if dir is not None:
            dir = os.path.expandvars(dir)

        self._dir = dir

        # from here on using the property
        if self.verbose and self.dir is not None:
            print('Database directory:', self.dir)
            if not self.dir_exists():
                print('  Directory does not yet exist. Use create()')

    def ready(self, action=None):
        if self.dir_exists():
            return True

    def dir_exists(self):
        """
        returns True of the database directory exists
        """
        if not os.path.exists(self.dir):
            return False
        else:
            if not os.path.isdir(self.dir):
                raise ValueError("Non-directory exists with that "
                                 "name: '%s'" % self.dir)
            return True

    def create(self):
        """
        Create the database directory
        """

        if os.path.exists(self.dir):
            raise RuntimeError("directory '%s' already exists" % self.dir)

        os.makedirs(self.dir)

    def delete(self, yes=False):
        """
        delete all data in the directory

        Parameters
        ----------
        yes: bool
            If True, don't prompt for confirmation
        """
        if not yes:
            answer = input('really delete all data? (y/n) ')
            if answer.lower() == 'y':
                yes = True

        if not yes:
            return

        for colname in self:
            self.delete_column(colname, yes=True)

    def _dirbase(self):
        """
        Return the dir basename minus any extension
        """
        if self.dir is not None:
            bname = os.path.basename(self.dir)
            name = '.'.join(bname.split('.')[0:-1])
            return name
        else:
            return None

    def load(self):
        """

        Load all existing columns in the
        directory.  Column files must have the right extensions to be noticed
        so far these can be

            .col
            .json

        and other column directories can be loaded if they have the extension

            .cols

        """

        if not self.dir_exists():
            raise ValueError("Database dir \n    '%s'\ndoes "
                             "not exist. Use create()" % self.dir)
        # clear out the existing columns and start from scratch
        self.clear()

        for type in ALLOWED_COL_TYPES:
            if self.dir is not None:
                pattern = os.path.join(self.dir, '*.'+type)
                fnames = glob(pattern)

                for f in fnames:
                    if type == 'cols':
                        self.load_coldir(f)
                    else:
                        self.load_column(filename=f)

    def load_column(self, name=None, filename=None, type=None):
        """
        Load the specified column
        """

        if filename is not None:

            name = _extract_colname(filename)
            type = _extract_coltype(filename)

        elif name is not None and type is not None:
            if self.dir is None:
                raise ValueError("no dir is set for Columns db, can't "
                                 "construct names")

            filename = _create_filename(self.dir, name, type)
        else:
            raise ValueError('either send a filename or specify name, type')

        if type not in ALLOWED_COL_TYPES:
            raise ValueError("bad column type: '%s'" % type)

        if type == 'array':
            col = ArrayColumn(
                filename=filename,
                dir=self.dir,
                name=name,
                verbose=self.verbose,
            )
        name = col.name
        self.clear(name)
        self[name] = col

    def load_coldir(self, dir):
        """

        Load a coldir under this coldir

        """
        coldir = Columns(dir, verbose=self.verbose)
        if not os.path.exists(dir):
            raise RuntimeError("coldir does not exists: '%s'" % dir)

        name = coldir._dirbase()
        self.clear(name)
        self[name] = coldir

    def reload(self, name=None):
        """

        Reload an existing column or everything
        Equivalent to self[name].reload()

        """
        if name is not None:
            if isinstance(name, str,):
                name = [name]

            for n in name:
                if name not in self:
                    raise ValueError("Unknown column '%s'" % name)
                self[name].reload()
        else:
            # just reload everything
            for name in self:
                self[name].reload()

    def colnames(self):
        """
        Return a list of all column names
        """
        return list(self.keys())

    def _get_repr_list(self):
        """
        Get a list of metadata for this columns directory and it's
        columns
        """
        indent = '  '
        s = []
        if self.dir is not None:
            dbase = self._dirbase()
            s += [dbase]
            s += ['dir: '+self.dir]
            if not self.dir_exists():
                s += ['    Directory does not yet exist']

        subcols = []
        if len(self) > 0:
            s += ['Columns:']
            cnames = 'name', 'type', 'dtype', 'index', 'shape'
            s += ['  %-15s %5s %6s %-6s %s' % cnames]
            s += ['  '+'-'*(50)]

            subcols = ['Sub-Column Directories:']
            subcols += ['  %-15s' % ('name',)]
            subcols += ['  '+'-'*(50)]

            for name in sorted(self):
                c = self[name]
                if isinstance(c, ColumnBase):

                    name = c.name

                    if len(name) > 15:
                        s += ['  %s' % name]
                        s += ['%23s' % (c.type,)]
                    else:
                        s += ['  %-15s %5s' % (c.name, c.type)]

                    if c.type == 'array':
                        c_dtype = c.dtype.descr[0][1]
                    else:
                        c_dtype = ''
                    s[-1] += ' %6s' % c_dtype
                    s[-1] += ' %-6s' % self[name].has_index
                    s[-1] += ' %s' % (self[name].shape,)

                else:
                    cdir = os.path.basename(c.dir).replace('.cols', '')
                    subcols += ['  %s' % cdir]

        s = [indent + tmp for tmp in s]
        s = ['Column Directory: '] + s

        if len(subcols) > 3:
            s += [indent]
            subcols = [indent + tmp for tmp in subcols]
            s += subcols

        return s

    def __repr__(self):
        """
        The columns representation to the world
        """
        s = self._get_repr_list()
        s = '\n'.join(s)
        return s

    def clear(self, name=None):
        """
        Clear out the dictionary of column info
        """
        if name is not None:
            if isinstance(name, (list, tuple)):
                for n in name:
                    if n in self:
                        del self[n]
        else:
            if name in self:
                del self[name]

    def write_column(self, name, data, type=None):
        """
        Write data to a column.

        If the column does not already exist, it is created.  If the type is
        'array' and the column exists, the data are appended.  For other types,
        the file is always created or overwritten.
        """

        if name not in self:
            if type is None:
                if isinstance(data, np.ndarray):
                    type = 'array'
                elif isinstance(data, dict):
                    type = 'json'
                else:
                    raise ValueError(
                        'only support array and dict types for now'
                    )
            self.load_column(name=name, type=type)

        self[name].write(data)

    def write(self, data):
        """
        Write the fields of a structured array to columns
        """

        names = data.dtype.names
        if names is None:
            raise ValueError('write() takes a structured array as '
                             'input')
        for name in names:
            self.write_column(name, data[name])

    write_columns = write

    def delete_column(self, name, yes=False):
        """
        delete the specified column and reload

        parameters
        ----------
        name: string
            Name of column to delete
        yes: bool
            If True, don't prompt for confirmation
        """
        if name not in self:
            print("cannot delete column '%s', it does not exist" % name)

        if not yes:
            answer = input('really delete all data? (y/n) ')
            if answer.lower() == 'y':
                yes = True

        if not yes:
            return

        self[name].delete()
        self.reload()

    def from_fits(self, filename, ext=1, lower=False):
        """
        Write columns to the database, reading from the input fits file.
        Uses chunks of 100MB

        parameters
        ----------
        filename: string
            Name of the file to read
        ext: extension number, optional
            The FITS extension to read from
        lower: bool, optional
            if True, lower-case all names
        """
        import fitsio

        with fitsio.FITS(filename, lower=lower) as fits:
            hdu = fits[ext]
            self._from_slicer(filename, hdu)

    def _from_slicer(self, filename, slicer):

        one = slicer[1:1+1]
        nrows = slicer.get_nrows()
        rowsize = one.itemsize

        # step size in bytes
        step_bytes = 100*1000000

        # step size in rows
        step = step_bytes//rowsize

        nstep = nrows//step
        nleft = nrows % step
        if nleft > 0:
            nstep += 1

        if self.verbose > 1:
            print('Loading %s rows from file: %s' % (nrows, filename))
            dp = pprint.pformat(one.dtype.descr)
            print('Details of columns: \n%s' % dp)

        for i in range(nstep):

            start = i*step
            stop = (i+1)*step
            if stop > nrows:
                stop = nrows

            if self.verbose > 1:
                print('Writing slice: %s:%s out '
                      'of %s' % (start, stop, nrows))
            data = slicer[start:stop]

            self.write(data)

    def from_columns(self, coldir, create=False, indent=''):
        """
        Load the columns of the input columns database into the current
        database, breaking it up into chunks of ~100 MB
        """

        chunksize = 100  # Mb
        step_bytes = chunksize*1000000

        if create:
            self.create()

        if isinstance(coldir, (list, tuple)):
            print('Processing list of', len(coldir), 'columns dirs')
            for cdir in coldir:
                self.from_columns(cdir, indent=indent+'    ')
                print(indent+'    '+'-'*70)
            return

        print(indent+"Loading columns from:", coldir)
        c = Columns(coldir)

        for colname in c:

            print(indent+'column:', colname)

            if isinstance(c[colname], Columns):
                print(indent+'=> column is a Columns db, recursing')
                dname = colname+'.cols'
                inpath = os.path.join(coldir, dname)
                thispath = os.path.join(self.dir, dname)
                if colname not in self:
                    self[colname] = Columns(thispath)
                    self[colname].create()

                self[colname].from_columns(inpath, indent=indent+'    ')
            else:
                nrows = c[colname].size
                rowsize = c[colname].dtype.itemsize

                # step size in rows
                step = step_bytes/rowsize

                nstep = nrows/step
                nleft = nrows % step
                if nleft > 0:
                    nstep += 1

                if nstep > 1:
                    text = indent+"    Working in %d %d Mb chunks"
                    print(text % (nstep, chunksize), end='')

                for i in range(nstep):
                    if nstep > 1:
                        print('.', end='')
                    start = i*step
                    stop = (i+1)*step
                    if stop > nrows:
                        stop = nrows

                    data = c[colname][start:stop]
                    self.write_column(colname, data)
                if nstep > 1:
                    print()

    def read(self,
             columns=None,
             colnames=None,  # deprecated
             rows=None,
             asdict=False,
             verbose=False):
        """
        read columns and rows

        Parameters
        ----------
        columns: sequence or string
            Can be a scalar string or a sequence of strings.  Defaults to all.
        rows: sequence or scalar
            Sequence of row numbers.  Defaults to all.
        asdict: bool, optional
            If True, read fixed length columns into numpy arrays and store
            each in a dictionary with key=colname.  When supported,
            variable length columns can also be stored in this type of
            container as e.g.  strings.

            if asdict=False, only fixed length columns can be read and they
            are all packed into a single structured array (e.g. recarray).
        verbose: bool, optional
            Be verbose

        Notes
        ------
            If a single column is desired, this can be read into a normal,
            unstructured array using read_column or the [] notation.

        Restrictions
        -------------
        If rows= keyword is sent, this is applied to all columns, and currently
        only supports numpy fixed length types.  If all columns are not the
        same length, an exception is raised.
        """

        if columns is None and colnames is not None:
            columns = colnames

        if columns is None:
            columns = sorted(list(self.keys()))

        dtype = []
        if isinstance(columns, str):
            columns = [columns]

        ncol = len(columns)
        nrows = np.zeros(ncol, dtype='i8')

        i = 0
        for colname in columns:
            if colname not in self:
                raise ValueError("Column '%s' not found" % colname)

            col = self[colname]
            shape = col.shape

            nrows[i] = shape[0]

            descr = col.dtype.descr[0][1]

            dt = (colname, descr)
            if len(shape) > 1:
                dt = dt + shape[1:]

            dtype.append(dt)

            i += 1

        if ncol > 1 and rows is not None:
            # make sure they really align properly
            w, = np.where(nrows != nrows[0])
            if w.size > 0:
                raise ValueError("When using the rows= keyword with multiple "
                                 "columns, the columns must be the same "
                                 "length")

        # if rows not sent, all are read
        if rows is not None:
            if np.isscalar(rows):
                n_rows2read = 1
            else:
                n_rows2read = len(rows)
        else:
            n_rows2read = nrows[0]

        if asdict:
            # Just putting the arrays into a dictionary.
            data = {}

            for colname in columns:

                if self.verbose or verbose:
                    print('\tColumn: %s getting %d '
                          'rows' % (colname, n_rows2read))

                # just read the data and put in dict, simpler than below
                data[colname] = self.read_column(colname, rows=rows)

        else:
            # copying into a single array with fields
            data = np.empty(n_rows2read, dtype=dtype)

            for colname in columns:

                if self.verbose or verbose:
                    print('\tColumn: %s getting %d '
                          'rows' % (colname, n_rows2read))

                data[colname][:] = self.read_column(colname, rows=rows)

        return data

    read_columns = read

    def read_column(self, colname, rows=None):
        """
        Only numpy, fixed length for now.  Eventually allow pickled columns.
        """
        if colname not in self:
            raise ValueError("Column '%s' not found" % colname)

        if self.verbose:
            if rows is not None:
                print("Reading %d rows from "
                      "column: '%s'" % (len(rows), colname))
            else:
                print("Reading column: '%s'" % colname)

        return self[colname].read(rows=rows)


class ColumnBase(object):
    """
    Represent a column in a Columns database.  Facilitate opening, reading,
    writing of data.  This class can be instantiated alone, but is usually
    accessed through the Columns class.

    Construction
    ------------
    >>> col=Column(filename=, name=, dir=, type=, verbose=False)

    # there are alternative construction methods
    # this method determins all info from the full path
    >>> col=Column(filename='/full/path')

    # this one uses directory, column name, type
    >>> col=Column(
            name='something', type='array',
            dir='/some/path/dbname.cols',
        )
    """
    def __init__(self,
                 filename=None,
                 name=None,
                 dir=None,
                 verbose=False):

        self.init(
            filename=filename,
            name=name,
            dir=dir,
            verbose=verbose,
        )

    def init(self,
             filename=None,
             name=None,
             dir=None,
             verbose=False):
        """
        See main docs for the Column class
        """

        self.clear()

        self._filename = filename
        self._name = name
        self._dir = dir
        self.verbose = verbose

        # make sure we have something to work with before continuing
        if not self._init_args_sufficient(filename, dir, name):
            return

        if self.filename is not None:
            self._set_meta_from_filename()

        elif self.name is not None:
            self._set_meta_from_name()

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
        self.init(filename=self.filename, verbose=self.verbose)

    def clear(self):
        """
        Clear out all the metadata for this column.
        """

        self._filename = None
        self._name = None
        self._dir = None
        self.verbose = False

    def write(self, data):
        """
        Write data to a column.
        """
        raise NotImplementedError('implement write')

    def read(self, *args):
        """
        Read data from a column
        """
        raise NotImplementedError('implement read')

    def delete(self):
        """
        Attempt to delete the data file associated with this column
        """
        if self.filename is None:
            return

        if os.path.exists(self.filename):
            print("Removing data for column: %s" % self.name)
            os.remove(self.filename)

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

            self._name = _extract_colname(self.filename)
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

            self._filename = _create_filename(self.dir, self.name, self.type)
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
    represents an array column in a Columns database

    If the numpydb package is available, B-tree indexes can be created for
    any column.  Searching can be done using standard ==, >, >=, <, <=
    operators, as well as the .match() and between() functions.  The
    functional forms are more powerful.

    Slice and item lookup
    ---------------------
    # The Column class supports item lookup access, e.g. slices and
    # arrays representing a subset of rows

    # Note true slices and row subset other than [:] are only supported
    # by the 'array' type

    >>> col=Column(...)
    >>> data = col[25:22]
    >>> data = col[row_list]

    # if the column itself contains multiple fields, you can access subsets
    # of these
    >>> id = col['id'][:]
    >>> data = col[ ['id','flux'] ][ rows ]

    Indexes on columns
    ------------------
    If the numpydb package is available, you can create indexes on
    columns and perform fast searches.

    # create the index
    >>> col.create_index()

    # get indices for some range.  Can also do
    >>> ind = (col > 25)
    >>> ind = col.between(25,35)
    >>> ind = (col == 25)
    >>> ind = col.match([25,77])

    # composite searches over multiple columns with the same number
    # of records
    >>> ind = (col1 == 25) & (col2 < 15.23)
    >>> ind = col1.between(15,25) | (col2 != 66)
    >>> ind = col1.between(15,25) & (col2 != 66) & (col3 > 5)

    """
    def init(self,
             filename=None,
             name=None,
             dir=None,
             verbose=False):
        """
        initialize the meta data, and possibly load the mmap
        """

        self._type = 'array'

        super(ArrayColumn, self).init(
            filename=filename,
            name=name,
            dir=dir,
            verbose=verbose,
        )

        if self.filename is None or not os.path.exists(self.filename):
            return

        self._open_file('r+')

        # get info for index if it exists
        self.init_index()

    def _open_file(self, mode):
        self._sf = SimpleFile(self.filename, mode=mode)

    def clear(self):
        """
        Clear out all the metadata for this column.
        """

        super(ArrayColumn, self).clear()

        if self.has_data:
            del self._sf

        self._has_index = False
        self._index_dtype = None

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
    def shape(self):
        """
        get the shape of the column
        """
        self.ensure_has_data()

        return self._sf.shape

    @property
    def size(self):
        """
        get the size of the columns
        """
        self.ensure_has_data()

        return self._sf.size

    def write(self, data):
        """
        Write data to the column.  Data are appended if the file already
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
            # create the new indices
            new_indices = np.arange(
                self.size-data.size,
                self.size,
                dtype=self.index_dtype,
            )
            self._write_to_index(data, new_indices)

    def __getitem__(self, arg):
        """
        Item lookup method, e.g. col[..] meaning slices or
        sequences, etc.
        """
        if not hasattr(self, '_sf'):
            raise ValueError('no file loaded yet')

        return self._sf[arg]

    def read(self, rows=None):
        """
        read data from this column

        Parameters
        ----------
        rows: sequence, optional
            A subset of the rows to read.
        """
        if not hasattr(self, '_sf'):
            raise ValueError('no file loaded yet')

        if rows is None:
            return self._sf[:]
        else:
            return self._sf[rows]

    def delete(self):
        """
        Attempt to delete the data file associated with this column
        """

        if hasattr(self, '_sf'):
            del self._sf

        super(ArrayColumn, self).delete()

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

    @property
    def index_dtype(self):
        """
        returns True if an index exists for this column
        """
        if not self.has_index:
            raise RuntimeError('no index exists for this column')
        return self._index_dtype

    def init_index(self):
        """
        If index file exists, load some info
        """
        index_fname = self.index_filename()
        if os.path.exists(index_fname):
            self._has_index = True
            db = numpydb.NumpyDB(index_fname)
            # for the numerical indices
            self._index_dtype = db.data_dtype()
            db.close()

    def index_filename(self, tempdir=None):
        """
        get the filename for the index of this column
        """
        if self.filename is None:
            return None

        # remove the final extension
        index_fname = '.'.join(self.filename.split('.')[0:-1])
        index_fname = index_fname+'__index.db'

        if tempdir is not None:
            bname = os.path.basename(index_fname)
            index_fname = os.path.join(tempdir, bname)

        return index_fname

    def create_index(self, index_dtype='i8', force=False,
                     tempdir=None, verbose=False, db_verbose=0):
        """
        Create an index for this column.  The index is created by the
        numpydb package which uses a Berkeley DB B-tree.  The database file
        will be named {columnfile}__index.db

        Once the index is created, all data from the column will be put
        into the index.  Also, after creation any data appended to the
        column are automatically added to the index.

        Parameters
        ----------
        index_dtype: str, optional
            The data type for the index.  The default is 'i8' but can
            also be 'i4'
        force: bool, optional
            If True, any existing index is deleted. If this keyword is not True
            and the index exists, and exception is raised.  Default is False.
        tempdir:
            A temporary directory to write the index.  This is very useful when
            the tempdir is for example a linux "tempfs" which is in memory,
            e.g. /dev/shm.  This can speed up index creation by a large factor

            Note if /dev/shm exists it will be used by default if you don't set
            tempdir yourself.

            After creation, the index will be moved to it's final destination.
        verbose: bool, optional
            This can override the overall verbosity.
        db_verbose: int, optional
            An integer indicating the verbosity of the db code.
        """

        if tempdir is None:
            if os.path.exists('/dev/shm'):
                tempdir = '/dev/shm'

        # delete existing index?
        if force:
            self.delete_index()

        # make sure we can create
        self._verify_db_available('create')

        # set up file name and data type info
        index_fname = self.index_filename(tempdir=tempdir)

        key_dtype = self.dtype.descr[0][1]

        # basic create
        if self.verbose or verbose:
            print("Creating index for column '%s'" % self.name)
            print("    db file: '%s'" % index_fname)

        numpydb.create(
            index_fname,
            key_dtype,
            index_dtype,
            verbosity=db_verbose,
        )

        # this reloads metadata for column, so we know the index file
        # exists (if not using a temp file)
        self.reload()

        # Write the data to the index.  We should do this in chunks of,
        # say, 100 MB or something
        data = self.read()
        indices = np.arange(data.size, dtype=index_dtype)

        # note sending filename= will prevent calling _verify_db_available
        # which is good since we may be using a temp file
        self._write_to_index(
            data,
            indices,
            filename=index_fname,
            verbose=db_verbose,
        )
        del data

        if tempdir is not None:
            # move to the final destination
            final_fname = self.index_filename()
            if self.verbose or verbose:
                print("    Moving to final destination: '%s'" % final_fname)
            shutil.move(index_fname, final_fname)

        self.reload()

    def delete_index(self):
        """
        Delete the index for this column if it exists
        """
        if self.has_index:
            index_fname = self.index_filename()
            if os.path.exists(index_fname):
                print("Removing index for column: %s" % self.name)
                os.remove(index_fname)

        self._has_index = False
        self._index_dtype = None

    def match(self, values, select='values'):
        """
        Find all entries that match the requested value or values and return a
        query index of the result.  The requested values can be a scalar,
        sequence, or array, and must be convertible to the key data type.

        The returned data is by default a columns.Index containing the indices
        (the "values" of the key-value database) for the matches, which can be
        combined with other queries to produce a final result, but this can be
        controlled through the use of keywords.

        Note if you just want to match a single value, you can also use the ==
        operator.

        Using match() requires that an index was created for this column using
        create_index()

        Parameters
        ----------
        values: scalar or sequence
            Value(s) to match.  All entries that match are included in the
            returned query index.  Must be convertible to the key data type.

            Note, these values must be *unique*.

        select: str
            Which data to return.  Can be
            'values':  Return the Index.  This is the values of the key-value
              pairs, which here is a set of indices. (Default)
            'keys': Return the keys of the key-value pairs.
            'both': Return a tuple (keys,values)
            'count': Return the count of all matches.

            Default behaviour is to return a Index of the key-value pairs in
            the database.

        Returns
        --------
        An Index by default, see the select keyword

        Examples
        --------
        # Find the matches and return a Index
        >>> ind = col.match(value)
        >>> ind = col.match([value1,value2,value3])


        # combine with the results of another query
        >>> ind = ( (col1.match(values)) & (col2 == value2) )

        # Instead of indices, extract the key values that match, these will
        # simply equal the requested values
        >>> keys = col.match(values, select='keys')

        # Extract both keys and values for the range of keys.  The data
        # part is not a Index object in this case.
        >>> keys,data = col.match(values,select='both')

        # just return the count
        >>> count = col.match(values,select='count')


        # ways to get the underlying array instead of an Index. The where()
        # function simply returns .array().  Note you can use an Index just
        # like a normal array, but it has different & and | properties
        >>> ind=col.match(value).array()
        >>> ind=columns.where( col.match(values) )
        """

        self._verify_db_available('read')
        db = numpydb.Open(self.index_filename())

        verbosity = 0
        if self.verbose:
            verbosity = 1
        db.set_verbosity(verbosity)

        result = db.match(values, select=select)
        db.close()

        if select == 'both':
            result = (result[0], Index(result[1]))
        elif select != 'count':
            result = Index(result)

        return result

    # one-sided range operators
    def __gt__(self, val):
        self._verify_db_available('read')
        db = numpydb.Open(self.index_filename())
        i = Index(db.range1(val, '>'))
        db.close()
        return i

    def __ge__(self, val):
        self._verify_db_available('read')
        db = numpydb.Open(self.index_filename())
        i = Index(db.range1(val, '>='))
        db.close()
        return i

    def __lt__(self, val):
        self._verify_db_available('read')
        db = numpydb.Open(self.index_filename())
        i = Index(db.range1(val, '<'))
        db.close()
        return i

    def __le__(self, val):
        self._verify_db_available('read')
        db = numpydb.Open(self.index_filename())
        i = Index(db.range1(val, '<='))
        db.close()
        return i

    # equality operators
    def __eq__(self, val):
        return self.match(val)

    def __ne__(self, val):
        self._verify_db_available('read')
        db = numpydb.Open(self.index_filename())
        ind1 = Index(db.range1(val, '<'))
        ind2 = Index(db.range1(val, '>'))
        ind = ind1 | ind2
        db.close()
        return ind

    def between(self, low, high, interval='[]', select='values'):
        """
        Find all entries in the range low,high, inclusive by default.  The
        returned data is by default a columns.Index containing the indices (the
        "values" of the key-value database) for the matches, which can be
        combined with other queries to produce a final result, but this can be
        controlled through the use of keywords.

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

        select: str, optional
            Which data to return.  Can be
                'values': Return the values of the key-value pairs, which
                    here is a set of indices. (Default)
                'keys': Return the keys of the key-value pairs.
                'both': Return a tuple (keys,values)
                'count': Return the count of all matches.

            Default behaviour is to return a Index of the key-value pairs in
            the database.

        examples
        ---------

        result=between(low,high,interval='[]', select='values')

        # Extract the indices for values in the given range
        >>> query_index = col.between(low,high)

        # Extract from different types of intervals
        >>> values = db.between(low, high,'[]')
        >>> values = db.between(low, high,'[)')
        >>> values = db.between(low, high,'(]')
        >>> values = db.between(low, high,'()')

        # combine with the results of another query and extract the
        # index array
        >>> ind = columns.where( (col1.between(low,high)) &
                                 (col2 == value2) )

        # Extract the key values for the range
        >>> keys = col.between(low,high,select='keys')

        # Extract both keys and values for the range of keys
        >>> keys,indices = db.between(low,high,select='both')

        # just return the count
        >>> count = db.between(low,high,select='count')

        # ways to get the underlying index array instead of an Index. The
        # where() function simply returns .array().  Note you can use an
        # Index just like a normal array, but it has different & and |
        # properties

        >>> ind=col.between(low,high).array()
        >>> ind=columns.where( col.between(low,high) )
        """

        self._verify_db_available('read')
        db = numpydb.Open(self.index_filename(), verbose=self.verbose)

        verbosity = 0
        if self.verbose:
            verbosity = 1
        db.set_verbosity(verbosity)

        result = db.between(low, high, interval, select=select)
        db.close()

        if select == 'both':
            result = (result[0], Index(result[1]))
        elif select != 'count':
            result = Index(result)

        return result

    def _write_to_index(self,
                        data,
                        indices,
                        cache=None,
                        filename=None,
                        verbose=False):

        if filename is None:
            self._verify_db_available('write')
            index_fname = self.index_filename()
        else:
            index_fname = filename

        db = numpydb.NumpyDB()
        if cache is not None:
            if not isinstance(cache, (tuple, list)):
                raise ValueError('cache must be a sequence '
                                 '[gbytes,bytes,ncache]')
            gbytes = int(cache[0])
            bytes = int(cache[1])
            ncache = int(cache[2])
            db.set_cachesize(gbytes, bytes, ncache)

        db.open(index_fname, 'r+')

        verbosity = 0
        if self.verbose or verbose:
            verbosity = 1
        db.set_verbosity(verbosity)

        if self.verbose:
            print("Writing data to index for column '%s'" % self.name)
        # if the sent data has names, use the name of
        # this column
        if data.dtype.names is not None:
            if self.name not in data.dtype.names:
                raise ValueError("No field '%s' in data" % self.name)
            db.put(data[self.name], indices)
        else:
            db.put(data, indices)
        db.close()

    def _verify_db_available(self, action=None):
        if not havedb:
            raise ImportError("Could not import numpydb")

        # for reading and writing, the file must already exist
        if action == 'read' or action == 'write':
            if not self.has_index:
                raise RuntimeError(
                    "No index file found for "
                    "column '%s'.  Use create_index()" % self.name
                )
            index_fname = self.index_filename()
            if not os.path.exists(index_fname):
                raise RuntimeError("index file does not "
                                   "exist: '%s'" % index_fname)
        elif action == 'create':
            index_fname = self.index_filename()
            if os.path.exists(index_fname):
                raise RuntimeError("index file already "
                                   "exists: '%s'" % index_fname)

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
                s += ['"'+self.name+'"']

            if self.filename is not None:
                s += ['filename: %s' % self.filename]

            s += ['type: array']

            if self.shape is not None:
                s += ['shape: %s' % (self.shape,)]

            if self.name is not None:
                s += ['has index: %s' % self.has_index]

            if self.dtype is not None:
                drepr = pprint.pformat(self.dtype.descr)
                drepr = drepr.split('\n')
                drepr = ['  '+d for d in drepr]
                s += ["dtype: "] + drepr

            s = [indent + tmp for tmp in s]
            s = ['Column: '] + s

        return s


class JSONColumn(ColumnBase):
    def init(self,
             filename=None,
             name=None,
             dir=None,
             verbose=False):
        """
        initialize the meta data, and possibly load the mmap
        """

        self._type = 'json'

        super(ArrayColumn, self).init(
            filename=filename,
            name=name,
            dir=dir,
            verbose=verbose,
        )

    def write(self, data):
        """
        Write data to the JSON column.

        Parameters
        ----------
        data: array
            The data must be supported by the JSON format.
        """

        _write_json(data, self.filename)

    def read(self):
        """
        read data from this column
        """

        return _read_json(self.filename)

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
                s += ['"'+self.name+'"']

            if self.filename is not None:
                s += ['filename: %s' % self.filename]

            s += ['type: JSON']

            s = [indent + tmp for tmp in s]
            s = ['Column: '] + s

        return s


class Index(np.ndarray):
    """
    Represent an index into a database.  This object inherits from normal
    numpy arrays, but behaves differently under the "&" and "|" operators.
    These return the intersection or union of values in two Index objects.

    Methods:
        The "&" and "|" operators are defined.

        array(): Return an ordinary np.ndarray view of the Index.

    Examples:
        >>> i1=Index([3,4,5])
        >>> i2=Index([4,5,6])
        >>> (i1 & i2)
        Index([4, 5])
        >>> (i1 | i2)
        Index([3, 4, 5, 6])

    """
    def __new__(self, init_data, copy=False):
        arr = np.array(init_data, copy=copy)
        shape = arr.shape

        ret = np.ndarray.__new__(self, shape, arr.dtype,
                                 buffer=arr)
        return ret

    def array(self):
        return self.view(np.ndarray)

    def __and__(self, ind):
        # take the intersection
        if isinstance(ind, Index):
            w = np.intersect1d(self, ind)
        else:
            raise ValueError("comparison index must be an Index object")
        return Index(w)

    def __or__(self, ind):
        # take the unique union
        if isinstance(ind, Index):
            w = np.union1d(self, ind)
        else:
            raise ValueError("comparison index must be an Index object")

        return Index(w)

    def __repr__(self):
        rep = np.ndarray.__repr__(self)
        rep = rep.replace('array', 'Index')
        return rep


def where(query_index):
    """
    Extract results from a query_index object into a normal numpy array.  This
    is not usually necessary, as query objects inherit from numpy arrays.

    Parameters
    ----------
    query_index: Index
        An Index object generated by using operators such as "==" on an indexed
        column object.  The Column methods between and match also return Index
        objects.  Index objects can be combined with the "|" and "&" operators.
        This where functions extracts the underlying index array.

    Returns
    -------
    numpy array of indices

    Example
    --------

        # lets say we have indexes on columns 'type' and 'mag' get the indices
        # where type is 'event' and rate is between 10 and 20.  Note the braces
        # around each operation are required here

        >>> import columns
        >>> c=columns.Columns(column_dir)
        >>> ind=columns.where(  (c['type'] == 'event')
                              & (c['rate'].between(10,20)) )

        # now read some data from a set of columns using these
        # indices.  We can do this with individual columns:
        >>> id = columns['id'][ind]
        >>> x = columns['x'][ind]
        >>> y = columns['y'][ind]

        # we can also extract multiple columns at once
        >>> data = columns.read_columns(['x','y','mag','type'],rows=ind)
    """
    return query_index.array()


def _read_json(fname):
    """
    wrapper to read json
    """

    with open(fname) as fobj:
        data = json.load(fobj)
    return data


def _write_json(obj, fname, pretty=True):
    """
    wrapper for writing json
    """

    with open(fname, 'w') as fobj:
        json.dump(obj, fobj, indent=1, separators=(',', ':'))


def _extract_colname(filename):
    """
    Extract the column name from the file name
    """

    bname = os.path.basename(filename)
    name = '.'.join(bname.split('.')[0:-1])
    return name


def _extract_coltype(filename):
    """
    Extract the type from the file name
    """
    return filename.split('.')[-1]

def _create_filename(dir, name, type):

    if dir is None:
        raise ValueError('Cannot create column filename, dir is None')

    if name is None:
        raise ValueError('Cannot create column filename: name is None')

    if type is None:
        raise ValueError('Cannot create column filename: type is None')

    return os.path.join(dir, name+'.'+type)


