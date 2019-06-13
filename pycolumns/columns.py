"""
todo

    - figure out when to sort the index for reading; this can make a big
      difference in read speeds
    - always use little endian dtype when writing
    - make sure copies are made rather than references, but don't copy twice in
      case where memmap getitem is already returning a copy
    - need to check when appending that all array cols are being updated

    - can we loosen up the requirement of columns being same number of rows?
      - if not, need to hide write_column and also put a check for this other
        than verify
"""
import os
import bisect
from glob import glob
import pprint
import json
import numpy as np
from .sfile import SimpleFile

ALLOWED_COL_TYPES = ['array', 'dict', 'cols']


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
    >>> import pycolumns as pyc

    # instantiate a column database from the specified coldir
    >>> c=pyc.Columns('/some/path/mycols.cols')

    # display some info about the columns
    >>> c
    Column Directory:

      dir: /some/path/mydata.cols
      Columns:
        name             type  dtype index  shape
        --------------------------------------------------
        ccd             array    <i2 True   (64348146,)
        dec             array    <f8 False  (64348146,)
        exposurename    array   |S20 True   (64348146,)
        id              array    <i8 False  (64348146,)
        imag            array    <f4 False  (64348146,)
        ra              array    <f8 False  (64348146,)
        x               array    <f4 False  (64348146,)
        y               array    <f4 False  (64348146,)
        g               array    <f8 False  (64348146, 2)
        meta             dict


      Sub-Column Directories:
        name
        --------------------------------------------------
        psfstars

    # display info about column 'id'
    >>> c['id']
    Column:
      "id"
      filename: ./id.array
      type: col
      shape: (64348146,)
      has index: False
      dtype: <i8

    # get the column names
    >>> c.colnames
    ['ccd', 'dec', 'exposurename', 'id', 'imag', 'ra', 'x', 'y', 'g', 'meta']

    # reload all columns or specified column/column list
    >>> c.reload(name=None)

    # read all data from column 'id'
    # alternative syntaxes
    >>> ind = c['id'][:]
    >>> ind = c['id'].read()
    >>> ind = c.read_column('id')

    # dict columns are read as a dict
    >>> meta = c['meta'].read()

    # read a subset of rows
    # slicing
    >>> ind = c['id'][25:125]

    # specifying a set of rows
    >>> rows=[3, 225, 1235]
    >>> ind = c['id'][rows]
    >>> ind = c.read_column('id', rows=rows)

    # read all columns into a single rec array.  By default the dict
    # columns are not loaded

    >>> data = c.read()

    # using asdict=True puts the data into a dict.  The dict data
    # are loaded in this case
    >>> data = c.read(asdict=True)

    # specify columns
    >>> data = c.read(columns=['id', 'flux'], rows=rows)

    # dict columns can be specified if asdict is True
    >>> data = c.read(columns=['id', 'flux', 'meta'], asdict=True)

    # Create indexes for fast searching
    >>> c['id'].create_index()

    # get indices for some condition
    >>> ind = c['id'] > 25
    >>> ind = c['id'].between(25, 35)
    >>> ind = c['id'] == 25

    # read the corresponding data
    >>> ccd = c['ccd'][ind]
    >>> data = c.read(columns=['ra', 'dec'], rows=ind)

    # composite searches over multiple columns
    >>> ind = (c['id'] == 25) & (col['ra'] < 15.23)
    >>> ind = c['id'].between(15, 25) | (c['id'] == 55)
    >>> ind = c['id'].between(15, 250) & (c['id'] != 66) & (c['ra'] < 100)

    # update values for a column
    >>> c['id'][35] = 10
    >>> c['id'][35:35+3] = [8, 9, 10]
    >>> c['id'][rows] = idvalues

    # write multiple columns from the fields in a rec array
    # names in the data correspond to column names.
    # If columns are not present, they are created
    # but row count consistency must be maintained for all array
    # columns and this is checked.

    >>> c.append(recdata)

    # append data from the fields in a FITS file
    >>> c.from_fits(fitsfile_name)

    # add a dict column
    >>> c.create_column('meta')
    >>> c['meta'].write({'test': 'hello'})
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

        if not os.path.exists(dir):
            os.makedirs(dir)

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
                        self._load_column(f)
        self.verify()

    @property
    def nrows(self):
        """
        number of rows in table
        """
        return self._nrows

    def verify(self):
        """
        verify all array columns have the same length
        """

        first = True
        for c in self.keys():
            col = self[c]
            if col.type == 'array':
                this_nrows = col.nrows
                if first:
                    self._nrows = this_nrows
                    first = False
                else:
                    if this_nrows != self.nrows:
                        raise ValueError(
                            'column size mismatch for %s '
                            'got %d vs %d' % (c, this_nrows, self.nrows)
                        )

    def _load_column(self, filename):
        """
        Load the specified column
        """

        name = _extract_colname(filename)
        type = _extract_coltype(filename)

        if type not in ALLOWED_COL_TYPES:
            raise ValueError("bad column type: '%s'" % type)

        if type == 'array':
            col = ArrayColumn(
                filename=filename,
                dir=self.dir,
                name=name,
                verbose=self.verbose,
            )
        elif type == 'dict':
            col = DictColumn(
                filename=filename,
                dir=self.dir,
                name=name,
                verbose=self.verbose,
            )
        else:
            raise ValueError("bad column type '%s'" % type)

        name = col.name
        self.clear(name)
        self[name] = col

    def create_column(self, name, type, verify=True):
        """
        create the specified column

        You usually don't want to use this directly for array types if columns
        already exist, because consistency will be broken, at least
        temporarily. In fact an exception will be raised.  Better to use the
        append method to ensure row length consistency

        It is fine to use it for dict types

        parameters
        ----------
        name: str
            Column name
        type: str
            Column type, 'dict'
        """

        if name in self:
            raise ValueError("column '%s' already exists" % name)

        if type not in ALLOWED_COL_TYPES:
            raise ValueError("bad column type: '%s'" % type)

        if self.dir is None:
            raise ValueError("no dir is set for Columns db, can't "
                             "construct names")

        filename = _create_filename(self.dir, name, type)

        if type == 'array':
            col = ArrayColumn(
                filename=filename,
                dir=self.dir,
                name=name,
                verbose=self.verbose,
            )
        elif type == 'dict':
            col = DictColumn(
                filename=filename,
                dir=self.dir,
                name=name,
                verbose=self.verbose,
            )
        else:
            raise ValueError("bad column type '%s'" % type)

        name = col.name
        self[name] = col

        if verify:
            self.verify()

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

    def reload(self, columns=None):
        """
        reload the database or a subset of columns

        parameters
        ----------
        columns: str or list of str, optional
            If sent, just reload the given columns.  Can be scalar
        """
        if columns is not None:
            if isinstance(columns, str):
                columns = [columns]
        else:
            columns = self.keys()

        for column in columns:
            if column not in self:
                raise ValueError("Unknown column '%s'" % column)
            self[column].reload()

        self.verify()

    @property
    def colnames(self):
        """
        Return a list of all column names
        """
        return list(self.keys())

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

    def append(self, data, verify=True):
        """
        Append data for the fields of a structured array to columns
        """

        names = data.dtype.names
        if names is None:
            raise ValueError('append() takes a structured array as input')

        for name in names:
            self.append_column(name, data[name], verify=False)

        # make sure the array columns all have the same length
        if verify:
            self.verify()

    def append_column(self, name, data, verify=True):
        """
        Append data to an array column.  The column is created
        if it doesn't exist

        You usually don't want to use this directly in case the
        row count consistency is broken, favor append() to append
        multiple columns
        """

        if name not in self:
            self.create_column(name, 'array', verify=False)

        self[name]._append(data)

        # make sure the array columns all have the same length
        if verify:
            self.verify()

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

            one = hdu[0:0+1]
            nrows = hdu.get_nrows()
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
                data = hdu[start:stop]

                self.append(data, verify=False)

        self.verify()

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
            answer = input("really delete column '%s'? (y/n) " % name)
            if answer.lower() == 'y':
                yes = True

        if not yes:
            return

        self[name].delete()
        self.reload()

    def read(self,
             columns=None,
             rows=None,
             asdict=False):
        """
        read multiple columns from the database

        Parameters
        ----------
        columns: sequence or string
            Can be a scalar string or a sequence of strings.  Defaults to all
            array columns if asdict is False, all columns if asdict is True
        rows: sequence or scalar
            Sequence of row numbers.  Defaults to all.
        asdict: bool, optional
            If True, read the requested columns into a dict.
        """

        columns = self._extract_columns(columns=columns, asdict=asdict)

        if len(columns) == 0:
            return None

        if asdict:
            # Just putting the arrays into a dictionary.
            data = {}

            for colname in columns:

                if self.verbose:
                    print('\treading column: %s' % colname)

                # just read the data and put in dict, simpler than below
                col = self[colname]
                if col.type == 'array':
                    data[colname] = col.read(rows=rows)
                else:
                    data[colname] = col.read()

        else:

            if rows is not None:
                try:
                    n_rows2read = len(rows)
                except TypeError:
                    n_rows2read = 1
            else:
                for c in columns:
                    col = self[c]
                    if col.type == 'array':
                        n_rows2read = col.nrows
                        break

            # copying into a single array with fields
            dtype = self._extract_dtype(columns)

            data = np.empty(n_rows2read, dtype=dtype)

            for colname in columns:
                if self.verbose:
                    print('\treading column: %s' % colname)

                col = self[colname]
                data[colname][:] = col.read(rows=rows)

        return data

    def _extract_dtype(self, columns):
        dtype = []

        for colname in columns:
            col = self[colname]
            shape = col.shape

            descr = col.dtype.descr[0][1]

            dt = (colname, descr)
            if len(shape) > 1:
                dt = dt + shape[1:]

            dtype.append(dt)

        return dtype

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

    def _extract_columns(self, columns=None, asdict=False):
        """
        extract the columns to read.  If no columns are sent then the behavior
        depends on the asdict parameter
            - if asdict is False, read all array columns
            - if asdict is True, read all columns
        """
        if columns is None:

            keys = sorted(self.keys())

            if asdict:
                columns = keys
            else:
                # just get the array columns
                columns = [c for c in keys if self[c].type == 'array']

        else:
            if isinstance(columns, str):
                columns = [columns]

            for c in columns:
                if c not in self:
                    raise ValueError("Column '%s' not found" % c)

                if not asdict and self[c].type != 'array':
                    if not asdict:
                        raise ValueError(
                            "requested non-array column '%s.' "
                            "Use asdict=True to read non-array "
                            "columns with general read() method" % c
                        )

        return columns

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


class ColumnBase(object):
    """
    Represent a column in a Columns database.  Facilitate opening, reading,
    writing of data.  This class can be instantiated alone, but is usually
    accessed through the Columns class.
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

    Construction
    ------------
    col=ArrayColumn(filename=, name=, dir=, verbose=False)

    # there are alternative construction methods
    # this method determins all info from the full path

    col=ArrayColumn(filename='/full/path')

    # this one uses directory, column name
    col=ArrayColumn(name='something', dir='/path2o/dbname.cols')

    Slice and item lookup
    ---------------------
    # The Column class supports item lookup access, e.g. slices and
    # arrays representing a subset of rows

    col=Column(...)
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
        self._init_index()

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

        return self._sf[arg]

    def __setitem__(self, arg, values):
        """
        Item lookup method, e.g. col[..] meaning slices or
        sequences, etc.
        """
        if not hasattr(self, '_sf'):
            raise ValueError('no file loaded yet')

        self._sf._mmap[arg] = values

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

        if self.has_index:
            print('column %s already has an index')
            return

        if len(self.shape) > 1:
            raise ValueError('cannot index a multi-dimensional column')

        if self.verbose:
            print('creating index for column %s' % self.name)

        dt = [
            ('index', 'i8'),
            ('value', self.dtype.descr[0][1]),
        ]
        index_data = np.zeros(self.shape[0], dtype=dt)
        index_data['index'] = np.arange(index_data.size)
        index_data['value'] = self[:]

        index_data.sort(order='value')

        # set up file name and data type info
        index_fname = self.index_filename
        with SimpleFile(index_fname, mode='w+') as sf:
            sf.write(index_data)

        self._init_index()

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
        mmap = self._index.mmap
        i = bisect.bisect_right(mmap['value'], val)
        indices = mmap['index'][i:].copy()

        return Index(indices)

    def __ge__(self, val):
        """
        bisect_left returns i such that data[i:] are all strictly >= val
        """
        mmap = self._index.mmap
        i = bisect.bisect_left(mmap['value'], val)
        indices = mmap['index'][i:].copy()

        return Index(indices)

    def __lt__(self, val):
        """
        bisect_left returns i such that data[:i] are all strictly < val
        """
        mmap = self._index.mmap
        i = bisect.bisect_left(mmap['value'], val)
        indices = mmap['index'][:i].copy()

        return Index(indices)

    def __le__(self, val):
        """
        bisect_right returns i such that data[:i] are all strictly <= val
        """
        mmap = self._index.mmap
        i = bisect.bisect_right(mmap['value'], val)
        indices = mmap['index'][:i].copy()

        return Index(indices)

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

        return Index(indices)

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
                c_dtype = self.dtype.descr[0][1]
                s += ["dtype: %s" % c_dtype]

            s = [indent + tmp for tmp in s]
            s = ['Column: '] + s

        return s


class DictColumn(ColumnBase):
    def init(self,
             filename=None,
             name=None,
             dir=None,
             verbose=False):

        self._type = 'dict'

        super(DictColumn, self).init(
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

            s += ['type: dict']

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
        >>> ind = columns['id'][ind]
        >>> x = columns['x'][ind]
        >>> y = columns['y'][ind]

        # we can also extract multiple columns at once
        >>> data = columns.read(columns=['x','y','mag','type'], rows=ind)
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

    if type == 'dict':
        ext = 'json'
    elif type == 'array':
        ext = 'array'
    else:
        raise ValueError("bad file type: '%s'" % type)
    return os.path.join(dir, name+'.'+ext)
