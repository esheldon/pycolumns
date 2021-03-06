"""
todo

    - figure out when to sort the index for reading; this can make a big
      difference in read speeds
    - need to check when appending that all array cols are being updated

    - can we loosen up the requirement of columns being same number of rows?
      - if not, need to hide write_column and also put a check for this other
        than verify
"""
import os
from glob import glob
import numpy as np
from .column import ColumnBase, ArrayColumn, DictColumn
from . import util

ALLOWED_COL_TYPES = ['array', 'dict', 'cols']


class Columns(dict):
    """
    Manage a database of "columns" represented by simple flat files.

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

        name = util.extract_colname(filename)
        type = util.extract_coltype(filename)

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

        filename = util.create_filename(self.dir, name, type)

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

            for i in range(nstep):

                start = i*step
                stop = (i+1)*step
                if stop > nrows:
                    stop = nrows

                if self.verbose > 1:
                    print('Writing slice: %s:%s out '
                          'of %s' % (start, stop, nrows))
                data = hdu[start:stop]

                data = util.get_native_data(data)

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


def where(query_index):
    """
    Extract results from a query_index object into a normal numpy array.  This
    is not usually necessary, as query objects inherit from numpy arrays.

    Parameters
    ----------
    query_index: Indices
        An Indices object generated by using operators such as "==" on an
        indexed column object.  The Column methods between and match also
        return Indices objects.  Indices objects can be combined with the "|"
        and "&" operators.  This where functions extracts the underlying
        index array.

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
