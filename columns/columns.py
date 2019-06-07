import os
from glob import glob
import pprint
import json
import numpy as np
import fitsio
import shutil

try:
    import numpydb
    havedb = True
except ImportError:
    havedb = False


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
          filename: ./id.fits
          type: col
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

        # Also get metadata
        >>> id, id_meta = c.read_column('id', rows=rows, meta=True)
        >>> meta=c['id'].read_meta()

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
        >>> c['col'].create_index()

        # get indices for some range.  Can also do
        >>> ind=(c['col'] > 25)
        >>> ind=c['col'].between(25,35)
        >>> ind=(c['col'] == 25)
        >>> ind=c['col'].match([25,77])

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
        >>> c.fromfile(fitsfile_name)


    """
    def __init__(self, dir=None, verbose=False):
        self.types = ['fits', 'idx', 'sort', 'cols', 'json']
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

    def _set_dir(self, dir=None):
        """
        Set the database directory, creating if none exists.
        """
        if dir is not None:
            dir = os.path.expandvars(dir)

        self.dir = dir
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

    def create(self, clobber=False):
        """
        Create the database directory
        """
        exists = os.path.exists(self.dir)

        if exists and not clobber:
            raise RuntimeError("directory '%s' already exists" % self.dir)

        elif exists and clobber:
            shutil.rmtree(self.dir)

        os.makedirs(self.dir)

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

        Load the metadata and a memory map for all existing columns in the
        directory.  Column files must have the right extensions to be noticed
        so far these can be

            .rec, .idx, .sort

        and other column directories can be loaded if they have the extension

            .cols

        """

        if not self.dir_exists():
            raise ValueError("Database dir \n    '%s'\ndoes "
                             "not exist. Use create()" % self.dir)
        # clear out the existing columns and start from scratch
        self.clear()

        for type in self.types:
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

        col = Column(
            filename=filename,
            dir=self.dir,
            name=name,
            type=type,
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
            coldir.create()
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
            cnames = 'name', 'type', 'dtype', 'index', 'size'
            s += ['  %-15s %5s %6s %-6s %s' % cnames]
            s += ['  '+'-'*(50)]

            subcols = ['Sub-Column Directories:']
            subcols += ['  %-15s' % ('name',)]
            subcols += ['  '+'-'*(50)]

            for name in sorted(self):
                c = self[name]
                if isinstance(c, Column):
                    name = c.name
                    if len(name) > 15:
                        s += ['  %s' % name]
                        s += ['%23s' % (c.type,)]
                    else:
                        s += ['  %-15s %5s' % (c.name, c.type)]

                    if c.type == 'fits':
                        c_dtype = c.dtype.descr[0][1]
                    else:
                        c_dtype = ''
                    s[-1] += ' %6s' % c_dtype
                    s[-1] += ' %-6s' % self[name].has_index()
                    s[-1] += ' %s' % self[name].size

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

    def write_column(self, name, data, type=None, create=False, meta=None):
        """
        Write data to a column.

        If the column does not already exist, it is created.  If the type is
        'fits' and the column exists, the data are appended unless create=True.
        For other types, the file is always created or overwritten.
        """

        if name in self and create:
            self.delete_column(name)

        if name not in self:
            if type is None:
                if isinstance(data, np.ndarray):
                    type = 'fits'
                elif isinstance(data, dict):
                    type = 'json'
                else:
                    raise ValueError(
                        'only support array and dict types for now'
                    )
            self.load_column(name=name, type=type)

        self[name].write(data, meta=meta)

    def write(self, data, create=False):
        """
        Write the fields of a structured array to columns
        """

        names = data.dtype.names
        if names is None:
            raise ValueError('write() takes a structured array as '
                             'input')
        for name in names:
            self.write_column(name, data[name], create=create)

    write_columns = write

    def delete_column(self, name):
        """
        delete the specified column and reload

        parameters
        ----------
        name: string
            Name of column to delete
        """
        if name not in self:
            print("cannot delete column '%s', it does not exist" % name)

        self[name].delete()
        self.reload()

    def fromfile(self,
                 filename,
                 create=False,
                 ext=1,
                 lower=False):
        """
        Write columns to the database, reading from the input fits file.
        Uses chunks of 100MB

        parameters
        ----------
        filename: string
            Name of the file to read
        create: bool, optional
            If True, over-write any existing columns with the same names as
            those in the file
        ext: extension number, optional
            The FITS extension to read from
        lower: bool, optional
            if True, lower-case all names
        """

        with fitsio.FITS(filename, lower=lower) as fits:
            hdu = fits[ext]
            self._from_slicer(filename, hdu, create)

    def _from_slicer(self, filename, slicer, create):

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
            if i == 0 and create:
                # only create first time
                docreate = True
            else:
                docreate = False

            start = i*step
            stop = (i+1)*step
            if stop > nrows:
                stop = nrows

            if self.verbose > 1:
                print('Writing slice: %s:%s out '
                      'of %s' % (start, stop, nrows))
            data = slicer[start:stop]

            self.write_columns(data, create=docreate)

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
            nrows[i] = col.size
            dtype.append(col.dtype.descr[0])

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

    def read_column(self, colname, rows=None, getmeta=False):
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

        return self[colname].read(rows=rows, getmeta=getmeta)


class Column(object):
    """
    Represent a column in a Columns database.  Facilitate opening, reading,
    writing of data.  This class can be instantiated alone, but is usually
    accessed through the Columns class.

    If the numpydb package is available, B-tree indexes can be created for
    any column.  Searching can be done using standard ==, >, >=, <, <=
    operators, as well as the .match() and between() functions.  The
    functional forms are more powerful.

    Construction
    ------------
        >>> col=Column(filename=, name=, dir=, type=, verbose=False)

        # there are alternative construction methods
        # this method determins all info from the full path
        >>> col=Column(filename='/full/path')

        # this one uses directory, column name, type
        >>> col=Column(name='something', type='fits',
                       dir='/some/path/dbname.cols')

    Slice and item lookup
    ---------------------
        # The Column class supports item lookup access, e.g. slices and
        # arrays representing a subset of rows

        # Note true slices and row subset other than [:] are only supported
        # by the 'fits' type

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
        >>> ind = c['col'].match([25,77])

        # composite searches over multiple columns with the same number
        # of records
        >>> ind = (col1 == 25) & (col2 < 15.23)
        >>> ind = col1.between(15,25) | (col2 != 66)
        >>> ind = col1.between(15,25) & (col2 != 66) & (col3 > 5)

    Methods
    -------

        # see docs for each method for more info

        init(): Same args as construction
        clear(): Clear out all metadata for this column.
        reload(): Reload all metadat for this column.
        read(rows=,columns=,fields=,getmeta=True):
            read data from column
        write(data, create=False, meta=None):
            write data to column, appending unless create=False
        delete():
            Attempt to delete the data file associated with this column


    """
    def __init__(self, filename=None, name=None, dir=None, type=None,
                 verbose=False):

        self.init(
            filename=filename,
            name=name,
            dir=dir,
            type=type,
            verbose=verbose,
        )

    def init(self, filename=None, name=None, dir=None, type=None,
             verbose=False):
        """
        See main docs for the Column class
        """

        self.clear()

        self.filename = filename
        self.name = name
        self.dir = dir
        self.type = type
        self.verbose = verbose

        # make sure we have something to work with before continuing
        if not self._args_sufficient(filename, dir, name, type):
            return

        if self.filename is not None:
            self._init_from_filename()
        elif self.name is not None:
            self._init_from_name()

        if self.type == 'fits':
            self._init_fits()

        if self.verbose:
            print(self.__repr__())

        # get info for index if it exists
        self.init_index()

    def reload(self):
        """
        Just reload the metadata and memory map for this column
        """
        if self.verbose:
            print('Reloading column metadata for: %s' % self.name)
        self.init(filename=self.filename, verbose=self.verbose)

    def clear(self):
        """
        Clear out all the metadata for this column.
        """
        self.filename = None
        self.name = None
        self.dir = None
        self.type = None
        self.verbose = False
        self.size = -1
        self.meta = None
        self.dtype = None

        self.have_index = False
        self.index_dtype = None

    def init_index(self):
        """
        If index file exists, load some info
        """
        index_fname = self.index_filename()
        if os.path.exists(index_fname):
            self.have_index = True
            db = numpydb.NumpyDB(index_fname)
            # for the numerical indices
            self.index_dtype = db.data_dtype()
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

    def has_index(self):
        """
        returns True if the column has an index
        """
        return self.have_index

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

        Restrictions
        ------------
        Currently, the column type must be ordinary 'col' and should be scalar.
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

    def _init_from_filename(self):
        """
        Initiaize this column based on the pull path filename
        """
        if self.filename is not None:
            if self.verbose:
                print("Initializing from file: %s" % self.filename)
            self.name = self._extract_name()
            self.type = self._extract_type()
        else:
            raise ValueError("You must set filename to use this function")

    def _init_from_name(self):
        """
        Initialize this based on name.  The filename is constructed from
        the dir and name
        """

        if (self.name is not None
                and self.type is not None
                and self.dir is not None):

            if self.verbose:
                mess = "Initalizing from \n\tdir: %s \n\tname: %s \n\ttype: %s"
                print(mess % (self.dir, self.name, self.type))

            self.filename = self._create_filename()
        else:
            raise ValueError("You must set dir,name,type to use this function")

    def _init_fits(self):
        """
        Init from a fits file
        """
        if self.filename is None:
            return

        if not os.path.exists(self.filename):
            return

        with fitsio.FITS(self.filename) as fits:
            hdu = fits[1]
            self.meta = hdu.read_header()
            self.size = hdu.get_nrows()
            example = hdu[0:1]
            descr = example.dtype.descr
            self.dtype = np.dtype(descr)

    def __getitem__(self, arg):
        """
        Item lookup method, e.g. col[..].  for fits files this is sent right to
        the __getitem__ of the fits hdu object.

        Slices and sequences are supported for rows.  You can also request a
        subset of fields.
        """

        if self.type == 'fits':
            with fitsio.FITS(self.filename) as fits:
                hdu = fits[1]
                data = hdu[self.name][arg]
            return data
        else:
            raise RuntimeError(
                'Only support indexing and slicing for fits type'
            )

    def read(self, rows=None, getmeta=False):
        """
        read data from this column

        Parameters
        ----------
        rows: sequence, optional
            A subset of the rows to read.
        getmeta: bool, optional
            Return a tuple (data,metadata). Only supported for array types
        """
        if self.type == 'fits':

            with fitsio.FITS(self.filename) as fits:
                hdu = fits[1]
                data = hdu.read(
                    rows=rows,
                    header=getmeta,
                )
                data = data[self.name]
                if getmeta:
                    meta = hdu.read_header()
                    return data, meta
                else:
                    return data

        elif self.type == 'json':
            return _read_json(self.filename)

        else:
            raise RuntimeError("Only support 'fits' and 'json' type")

    def read_meta(self):
        """
        Read the meta data for this column.  Metadata is only supported
        for array types
        """
        if self.type == 'fits':
            with fitsio.FITS(self.filename) as fits:
                hdu = fits[1]
                return hdu.read_header()
        else:
            raise RuntimeError('only fits type has meta data')

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

    def write(self, data, create=False, meta=None):
        """
        Write data to a column.  Append unles create=True.
        The column must be created with mode = 'w' or 'a' or 'r+'.
        (mode checking not yet implemented)

        Inputs:
            data:
                Data to write.  If the column data already exists, data may be
                appended for 'fits' type columns.  The data types in that case
                must match exactly.

        Keywords:
            create: If True, delete the existing data and write a new
                file. Default False.
            meta:
                Add this metadata to the header of the file if this is
                supported by the file type.  Will normally only be written if
                this is the creation of the file.
        """

        if create:
            self.delete()

        if self.type == 'fits':
            self._write_fits(data, meta=meta)
        elif self.type == 'json':
            self._write_json(data)
        else:
            raise RuntimeError("Currently only support rec types")

    def _write_json(self, data):
        """
        write json data
        """
        _write_json(data, self.filename)

    def _write_fits(self, data, create=False, meta=None):
        """
        Write data to a 'col' column in a fits file.  The data must be an array
        without fields.  Append unless create=True.

        Parameters
        ----------
        data: array
            Data to append to the file.  If the column data already exists,
            and create=False, the data types must match exactly.
        create: bool, optional
            If True, delete the existing data and write a new file. Default
            False.
        meta: bool, optional
            Add this metadata to the header of the file.  Will only be written
            if this is the creation of the file. Note the number of rows is
            updated during appending.
        """

        # If forcing create, delete myself.
        if create:
            self.delete()

        # make sure the data type of the input equals that of the column
        if not isinstance(data, np.ndarray):
            raise ValueError("For 'fits' columns data must be a numpy array")

        if data.dtype.names is not None:
            raise ValueError('do not enter data with fields')
        # view the data with a named field for writing
        data = self._get_named_view(data)

        with fitsio.FITS(self.filename, 'rw', clobber=create) as fits:
            if 1 in fits:
                fits[1].append(data)
            else:
                fits.write(data, header=meta)
                self.dtype = np.dtype(data.dtype.descr)

            self.size = fits[1].get_nrows()

        if self.have_index:
            # create the new indices
            new_indices = np.arange(
                self.size-data.size,
                self.size,
                dtype=self.index_dtype,
            )
            self._write_to_index(data, new_indices)

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
        if self.type != 'col':
            raise ValueError("Column type must be 'col' for indexing")

        # for reading and writing, the file must already exist
        if action == 'read' or action == 'write':
            if not self.have_index:
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

    def delete(self):
        """
        Attempt to delete the data file associated with this column
        """
        if self.filename is None:
            return
        if os.path.exists(self.filename):
            print("Removing data for column: %s" % self.name)
            os.remove(self.filename)

        # remove index if it exists
        self.delete_index()

        print("Deleting metadata for column: %s" % self.name)
        self.size = 0
        self.meta = None
        self.dtype = None

    def delete_index(self):
        """
        Delete the index for this column if it exists
        """
        if self.have_index:
            index_fname = self.index_filename()
            if os.path.exists(index_fname):
                print("Removing index for column: %s" % self.name)
                os.remove(index_fname)

        self.have_index = False
        self.index_dtype = None

    def _get_named_view(self, data):
        typestring = data.dtype.descr[0][1]
        shape = data[0].shape
        if len(shape) > 0:
            # I don't know how to do a simple view in this case,
            # I'm going to have to make a copy
            if len(shape) == 1:
                # this is just easier to read
                shape = shape[0]
            dtype_use = [(self.name, typestring, shape)]
            size = data.shape[0]
            newdata = np.zeros(size, dtype=dtype_use)
            newdata[self.name] = data
        else:
            # Can just re-view it
            dtype_use = [(self.name, typestring)]
            newdata = data.view(dtype_use)

        return newdata

    def _get_repr_list(self, full=False):
        """

        Get a list of metadat for this column.

        """
        indent = '  '

        if not full:
            s = ''
            if self.name is not None:
                s += 'Column: %-15s' % self.name
            if self.type is not None:
                s += ' type: %10s' % self.type
            if self.size >= 0:
                s += ' size: %12s' % self.type
            if self.name is not None:
                s += ' has index: %s' % self.have_index

            s = [s]
        else:
            s = []
            if self.name is not None:
                s += ['"'+self.name+'"']

            if self.filename is not None:
                s += ['filename: %s' % self.filename]
            if self.type is not None:
                s += ['type: %s' % self.type]
            if self.size >= 0:
                s += ['size: %s' % self.size]

            if self.name is not None:
                s += ['has index: %s' % self.have_index]

            if self.dtype is not None:
                drepr = pprint.pformat(self.dtype.descr)
                drepr = drepr.split('\n')
                drepr = ['  '+d for d in drepr]
                s += ["dtype: "] + drepr

            if self.meta is not None:
                hs = self._meta2string()
                if hs != '':
                    hs = hs.split('\n')
                    hs = ['  '+h for h in hs]
                    s += ['meta data:'] + hs

            s = [indent + tmp for tmp in s]
            s = ['Column: '] + s

        return s

    def __repr__(self):
        """
        Print out some info about this column
        """
        s = self._get_repr_list(full=True)
        s = "\n".join(s)
        return s

    def _extract_name(self):
        """
        Extract the column name from the file name
        """
        if self.filename is None:
            raise ValueError("You haven't specified a filename yet")
        bname = os.path.basename(self.filename)
        name = '.'.join(bname.split('.')[0:-1])
        return name

    def _extract_type(self):
        """
        Extract the type from the file name
        """
        if self.filename is None:
            raise ValueError("You haven't specified a filename yet")
        return self.filename.split('.')[-1]

    def _create_filename(self):
        if self.dir is None:
            raise ValueError("Cannot create column filename: directory "
                             "has not been set")
        if self.name is None:
            raise ValueError("Cannot create column filename: name "
                             "has not been set")
        if self.type is None:
            raise ValueError("Cannot create column filename: type "
                             "has not been set")
        return os.path.join(self.dir, self.name+'.'+self.type)

    def _meta2string(self, strip=True):
        if self.meta is None:
            return ''

        newd = {}

        """
        skipkeys = [
            '_DTYPE',
            '_SIZE',
            '_NROWS',
            '_HAS_FIELDS',
            '_DELIM',
            '_SHAPE',
            '_VERSION',
        ]
        """
        skipkeys = [
            'BITPIX',
            'GCOUNT',
            'NAXIS',
            'NAXIS1',
            'NAXIS2',
            'PCOUNT',
            'TFIELDS',
            'TFORM1',
            'TTYPE1',
            'XTENSION',
        ]

        for key in self.meta:
            if strip:
                if key not in skipkeys:
                    newd[key] = self.meta[key]
            else:
                newd[key] = self.meta[key]

        if len(newd) == 0:
            return ''
        return pprint.pformat(newd)

    def _args_sufficient(self,
                         filename=None,
                         dir=None,
                         name=None,
                         type=None):
        """
        Determine if the inputs are enough for initialization
        """
        if (filename is None) and \
                (dir is None or name is None or type is None):
            return False
        else:
            return True


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
