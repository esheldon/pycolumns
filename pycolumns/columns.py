"""
todo

    - add read_into so we don't have to make copies, but that would be slower
    when reading slices into rec since it could not do one big fread
    - allow update index without completely redoing it
    - tests for sub columns
    - Maybe add option "unsort" to put indices back in original unsorted order
    - add updating a set of columns/column
    - make it possible to add_column for array if we then eventually verify
    - if reading single row, scalar, doing from column gives a number but
      on columns with read gives length 1 array
    - support update on dict column, which would be like a normal dict
      update
"""
import os
import numpy as np
from .filebase import FileBase
from .column import Column
from .dictfile import Dict
from . import util

ALLOWED_COL_TYPES = ['array', 'dict', 'cols']


class Columns(dict):
    """
    Manage a database of "columns" represented by simple flat files.

    Parameters
    ----------
    dir: str
        Path to database
    cache_mem: number, optional
        Cache memory for index creation in gigabytes.  Default 1.0
    verbose: bool, optional
        If set to True, print messages
    """

    def __init__(self, dir=None, cache_mem=1, verbose=False):
        self._type = 'cols'
        self._verbose = verbose
        self._cache_mem_gb = float(cache_mem)
        self._set_dir(dir)
        self._load()

    @property
    def nrows(self):
        """
        number of rows in table
        """
        return self._nrows

    @property
    def names(self):
        """
        Get a list of all column names
        """
        return list(self.keys())

    @property
    def column_names(self):
        """
        Get a list of all column names
        """
        return [c for c in self if self[c].type == 'array']

    @property
    def dict_names(self):
        """
        Get a list of the array column names
        """
        return [c for c in self if self[c].type == 'dict']

    @property
    def subcols_names(self):
        """
        Get a list of the array column names
        """
        return [c for c in self if self[c].type == 'cols']

    @property
    def type(self):
        """
        Get the type (cols for Columns)
        """
        return self._type

    @property
    def dir(self):
        return self._dir

    @property
    def verbose(self):
        return self._verbose

    @property
    def cache_mem(self):
        return self._cache_mem_gb

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
        delete the entire Columns database

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

        for name in self:
            self.delete_entry(name, yes=True)

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

    def _load(self):
        """

        Load all existing columns in the
        directory.  Column files must have the right extensions to be noticed
        so far these can be

            .col
            .json

        and other column directories can be loaded if they have the extension

            .cols

        """
        from glob import glob

        # clear out the existing columns and start from scratch
        self._clear()

        for type in ALLOWED_COL_TYPES:
            if self.dir is not None:
                pattern = os.path.join(self.dir, '*.'+type)
                fnames = glob(pattern)

                for f in fnames:
                    if type == 'cols':
                        self._load_coldir(f)
                    else:
                        self._load_entry(f)
        self.verify()

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

    def _load_entry(self, filename):
        """
        Load the specified column
        """

        name = util.extract_colname(filename)
        type = util.extract_coltype(filename)

        if type not in ALLOWED_COL_TYPES:
            raise ValueError("bad column type: '%s'" % type)

        col = self._open_entry(
            filename=filename,
            name=name,
            type=type,
        )

        name = col.name
        self._clear(name)
        self[name] = col

    def _open_entry(self, filename, name, type):

        if type == 'array':
            col = Column(
                filename=filename,
                dir=self.dir,
                name=name,
                verbose=self.verbose,
                cache_mem=self.cache_mem,
            )
        elif type == 'dict':
            col = Dict(
                filename=filename,
                dir=self.dir,
                name=name,
                verbose=self.verbose,
            )
        else:
            raise ValueError("bad column type '%s'" % type)

        return col

    def create_dict(self, name):
        """
        create a new dict column

        parameters
        ----------
        name: str
            Column name
        """

        type = 'dict'
        if name in self.dict_names:
            raise ValueError("column '%s' already exists" % name)

        if self.dir is None:
            raise ValueError("no dir is set for Columns db, can't "
                             "construct names")

        filename = util.create_filename(dir=self.dir, name=name, type=type)

        dictfile = self._open_entry(
            filename=filename,
            name=name,
            type=type,
        )

        name = dictfile.name
        self[name] = dictfile

    def _create_column(self, name, verify=True):
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
        """

        type = 'array'

        if name in self:
            raise ValueError("column '%s' already exists" % name)

        if self.dir is None:
            raise ValueError("no dir is set for Columns db, can't "
                             "construct names")

        filename = util.create_filename(dir=self.dir, name=name, type=type)

        col = self._open_entry(
            filename=filename,
            name=name,
            type=type,
        )

        name = col.name
        self[name] = col

        if verify:
            self.verify()

    def _load_coldir(self, dir):
        """
        Load a coldir under this coldir
        """
        if not os.path.exists(dir):
            raise RuntimeError("coldir does not exists: '%s'" % dir)

        cols = Columns(
            dir, cache_mem=self.cache_mem, verbose=self.verbose,
        )

        name = cols._dirbase()
        self._clear(name)
        self[name] = cols

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

    def _clear(self, name=None):
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

        if len(self) > 0:
            # make sure the input data matches the existing column names
            in_names = set(names)
            column_names = set(self.column_names)
            if in_names != column_names:
                raise ValueError(
                    f'input columns {in_names}'
                    f'do not match existing table columns {column_names}'
                )

        for name in names:
            self._append_column(name, data[name])

        # make sure the array columns all have the same length
        if verify:
            self.verify()

    def _append_column(self, name, data):
        """
        Append data to an array column.  The column is created
        if it doesn't exist
        """

        if name not in self:
            self._create_column(name, verify=False)

        self[name]._append(data)

    def from_fits(
        self, filename, ext=1, native=False, little=True, lower=False,
    ):
        """
        Write columns to the database, reading from the input fits file.
        Uses chunks of 100MB

        parameters
        ----------
        filename: string
            Name of the file to read
        ext: extension, optional
            The FITS extension to read, numerical or string. default 1
        native: bool, optional
            FITS files are in big endian byte order.
            If native is True, ensure the outpt is in native byte order.
            Default False.
        little: bool, optional
            FITS files are in big endian byte order.
            If little is True, convert to little endian byte order. Default
            True.
        lower: bool, optional
            if True, lower-case all names.  Default False.
        """
        import fitsio

        if (native and np.little_endian) or little:
            byteswap = True
            if self.verbose:
                print('byteswapping')
        else:
            byteswap = False

        # step size in bytes
        step_bytes = int(self._cache_mem_gb * 1024**3)

        with fitsio.FITS(filename, lower=lower) as fits:
            hdu = fits[ext]

            one = hdu[0:0+1]
            nrows = hdu.get_nrows()
            rowsize = one.itemsize

            # step size in rows
            step = step_bytes // rowsize

            nstep = nrows // step
            nleft = nrows % step

            if nleft > 0:
                nstep += 1

            if self.verbose:
                print('Loading %s rows from file: %s' % (nrows, filename))

            for i in range(nstep):

                start = i * step
                stop = (i + 1) * step

                if stop > nrows:
                    # not needed, but use for printouts
                    stop = nrows

                if self.verbose:
                    print(f'    {start}:{stop} of {nrows}')

                data = hdu[start:stop]

                if byteswap:
                    data.byteswap(inplace=True)
                    data.dtype = data.dtype.newbyteorder()

                self.append(data, verify=False)
                del data

        self.verify()

    def delete_entry(self, name, yes=False):
        """
        delete the specified entry and reload

        parameters
        ----------
        name: string
            Name of entry to delete
        yes: bool
            If True, don't prompt for confirmation
        """
        if name not in self.names:
            print("cannot delete entry '%s', it does not exist" % name)

        if not yes:
            answer = input("really delete entry '%s'? (y/n) " % name)
            if answer.lower() == 'y':
                yes = True

        if not yes:
            return

        entry = self[name]
        if entry.type == 'cols':
            print("Removing data for sub columns: %s" % name)

            entry.delete(yes=True)
            fname = entry.filename
            if os.path.exists(fname):
                os.removedirs(fname)
        else:
            fname = entry.filename
            if os.path.exists(fname):
                print("Removing data for entry: %s" % name)
                os.remove(fname)

        self.reload()

    def read(
        self,
        columns=None,
        rows=None,
        asdict=False,
    ):
        """
        read multiple columns from the database

        Parameters
        ----------
        columns: sequence or string
            Can be a scalar string or a sequence of strings.  Defaults to all
            array columns if asdict is False, but will include
            dicts if asdict is True
        rows: sequence, slice or scalar
            Sequence of row numbers.  Defaults to all.
        asdict: bool, optional
            If set to True, read the requested columns into a dict.
        """

        columns = self._extract_columns(columns=columns, asdict=asdict)

        if len(columns) == 0:
            return None

        # converts to slice or Indices and converts stepped slices to
        # arange
        rows = util.extract_rows(rows=rows, nrows=self.nrows, sort=True)

        if asdict:
            # Just putting the arrays into a dictionary.
            data = {}

            for colname in columns:

                if self.verbose:
                    print('    reading column: %s' % colname)

                # just read the data and put in dict, simpler than below
                col = self[colname]
                if col.type == 'array':
                    data[colname] = col.read(rows=rows)
                else:
                    data[colname] = col.read()

        else:

            if isinstance(rows, slice):
                n_rows2read = rows.stop - rows.start
            else:
                n_rows2read = rows.size

            # copying into a single array with fields
            dtype = self._extract_dtype(columns)

            data = np.empty(n_rows2read, dtype=dtype)

            for colname in columns:
                if self.verbose:
                    print('    reading column: %s' % colname)

                col = self[colname]
                data[colname][:] = col.read(rows=rows)

        return data

    def _extract_dtype(self, columns):
        dtype = []

        for colname in columns:
            col = self[colname]

            descr = col.dtype.str

            dt = (colname, descr)
            dtype.append(dt)

        return dtype

    def _extract_columns(self, columns=None, asdict=False):
        """
        extract the columns to read.  If no columns are sent then the behavior
        depends on the asdict parameter
            - if asdict is False, read all array columns
            - if asdict is True, read all columns
        """
        if columns is None:

            if asdict:
                keys = sorted(self.keys())
                columns = keys
            else:
                columns = self.column_names

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
            if hasattr(self, '_nrows'):
                s += ['nrows: %s' % self.nrows]

        s += ['']
        subcols = []
        if len(self) > 0:
            s += ['Columns:']
            cnames = 'name', 'dtype', 'index'
            s += ['  %-15s %6s %-6s' % cnames]
            s += ['  '+'-'*(28)]

            dicts = ['Dictionaries:']
            dicts += ['  %-15s' % ('name',)]
            dicts += ['  '+'-'*(28)]

            subcols = ['Sub-Columns Directories:']
            subcols += ['  %-15s' % ('name',)]
            subcols += ['  '+'-'*(28)]

            for name in sorted(self):
                c = self[name]
                if isinstance(c, FileBase):

                    name = c.name

                    if len(name) > 15:
                        name_entry = ['  %s' % name]
                        # s += ['%23s' % (c.type,)]
                    else:
                        name_entry = ['  %-15s' % c.name]

                    if c.type == 'array':
                        s += name_entry
                        c_dtype = c.dtype.descr[0][1]
                        s[-1] += ' %6s' % c_dtype
                        s[-1] += ' %-6s' % self[name].has_index
                        # s[-1] += ' %s' % self[name].nrows
                    elif c.type == 'dict':
                        dicts += name_entry
                    else:
                        raise ValueError(f'bad type: {c.type}')
                else:
                    cdir = os.path.basename(c.dir).replace('.cols', '')
                    subcols += ['  %s' % cdir]

        s = [indent + tmp for tmp in s]
        s = ['Columns Directory: '] + s

        if len(dicts) > 3:
            s += [indent]
            dicts = [indent + tmp for tmp in dicts]
            s += dicts

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
