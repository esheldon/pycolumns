"""
TODO

    - Add update of entries for compressed
        - if chunk shrinks, could write in the chunk, if expands
          would need to do something new
            1. push data toward end of file
            2. mark chunk bad and copy to end?  Can vacuum later
    - ability to add a column
    - Maybe don't have dicts and subcols in self as a name
        - get_dict()
        - get_subcols()
    - setters for some things like cache_mem, verbose etc.
    - partitioning of data
"""
import os
import numpy as np
from .column import Column
from .dictfile import Dict
from . import util
from .schema import TableSchema
from .defaults import DEFAULT_CACHE_MEM, DEFAULT_CHUNKSIZE


class Columns(dict):
    """
    Manage a database of "columns" represented by simple flat files.

    Parameters
    ----------
    dir: str
        Path to database
    cache_mem: str or number
        Cache memory for index creation, default '1g' or one gigabyte.
        Can be a number in gigabytes or a string
        Strings should be like '{amount}{unit}'
            '1g' = 1 gigabytes
            '100m' = 100 metabytes
            '1000k' = 1000 kilobytes
        Units can be g, m or k, case insenitive
    verbose: bool, optional
        If set to True, print messages
    """

    def __init__(self, dir, cache_mem=DEFAULT_CACHE_MEM, verbose=False):

        dir = os.path.expandvars(dir)

        if not os.path.exists(dir):
            raise RuntimeError(
                f'dir {dir} does not exist.  Use Columns.create to initialize'
            )

        self._dir = dir
        self._type = 'cols'
        self._verbose = verbose
        self._is_updating = False
        self._cache_mem = cache_mem
        self._cache_mem_bytes = util.convert_to_bytes(cache_mem)
        self._cache_mem_gb = util.convert_to_gigabytes(cache_mem)
        self._load()

    @classmethod
    def create(
        cls, dir, schema={}, cache_mem=DEFAULT_CACHE_MEM, verbose=False,
        overwrite=False,
    ):
        """
        Initialize a new columns database.  The new Columns object
        is returned.

        Parameters
        ----------
        dir: str
            Path to columns directory
        schema: dict, optional
            Dictionary holding information for each column.
        cache_mem: str or number
            Cache memory for index creation, default '1g' or one gigabyte.
            Can be a number in gigabytes or a string
            Strings should be like '{amount}{unit}'
                '1g' = 1 gigabytes
                '100m' = 100 metabytes
                '1000k' = 1000 kilobytes
            Units can be g, m or k, case insenitive

        verbose: bool, optional
            If set to True, display information
        overwrite: bool, optional
            If the directory exists, remove existing data

        Examples
        --------
        import pycolumns as pyc

        # create from column schema
        idcol = pyc.ColumnSchema(dtype='i8', compress=True)
        racol = pyc.ColumnSchema(dtype='f8')
        schema = pyc.TableSchema([idcol, racol])
        cols = pyc.Columns.create(dir, schema=schema)

        # see the TableSchema and ColumnSchema classes for more options
        """
        import shutil

        if dir is not None:
            dir = os.path.expandvars(dir)

        if os.path.exists(dir):
            if not overwrite:
                raise RuntimeError(
                    f'directory {dir} already exists, send '
                    f'overwrite=True to replace'
                )

            if verbose:
                print(f'removing {dir}')
            shutil.rmtree(dir)

        if verbose:
            print(f'creating: {dir}')

        os.makedirs(dir)

        cols = Columns(dir, cache_mem=cache_mem, verbose=verbose)
        cols._add_columns(schema)
        return cols

    @classmethod
    def from_array(
        cls, dir, array,
        compression=None,
        chunksize=DEFAULT_CHUNKSIZE,
        append=True,
        cache_mem=DEFAULT_CACHE_MEM,
        verbose=False,
        overwrite=False,
    ):
        """
        Initialize a new columns database, creating the schema from the input
        array with fields.  The data from array are also written unless
        append=False.

        The new Columns object is returned.

        Parameters
        ----------
        dir: str
            Path to columns directory
        array: numpy array with fields or dict of arrays
            An array with fields defined
        compression: dict, list, bool, or None, optional
            See TableSchema.from_array for a full explanation
        chunksize: dict, str or number
            See TableSchema.from_array for a full explanation
        append: bool, optional
            If set to True, the data are also written to the new
            Default True
        cache_mem: str or number
            Cache memory for index creation, default '1g' or one gigabyte.
            Can be a number in gigabytes or a string
            Strings should be like '{amount}{unit}'
                '1g' = 1 gigabytes
                '100m' = 100 metabytes
                '1000k' = 1000 kilobytes
        Units can be g, m or k, case insenitive

        verbose: bool, optional
            If set to True, display information
        overwrite: bool, optional
            If the directory exists, remove existing data

        Examples
        --------
        array = np.zeros(num, dtype=[('id', 'i8'), ('x', 'f4')])
        cols = Columns.from_array(dir, array)
        """
        schema = TableSchema.from_array(
            array, compression=compression, chunksize=chunksize,
        )
        cols = Columns.create(
            dir,
            schema=schema,
            cache_mem=cache_mem,
            verbose=verbose,
            overwrite=overwrite,
        )
        if append:
            cols.append(array)

        return cols

    @property
    def nrows(self):
        """
        number of rows in table
        """
        return self._nrows

    @property
    def size(self):
        """
        number of rows in table
        """
        return self.nrows

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
        return [c for c in self if self[c].type == 'col']

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
    def is_updating(self):
        return self._is_updating

    @property
    def cache_mem(self):
        return self._cache_mem

    @property
    def cache_mem_bytes(self):
        return self._cache_mem_bytes

    @property
    def cache_mem_gb(self):
        return self._cache_mem_gb

    def _dirbase(self):
        """
        Return the dir basename minus any extension
        """
        bname = os.path.basename(self.dir)
        name = '.'.join(bname.split('.')[0:-1])
        return name

    def _load(self, verify=True):
        """
        Load all entries
        """
        from glob import glob

        # clear out the existing columns and start from scratch
        self._clear_all()

        # load the table columns
        pattern = os.path.join(self.dir, '*')
        fnames = glob(pattern)
        for fname in fnames:
            name = util.extract_colname(fname)
            type = util.extract_coltype(fname)

            if self.verbose:
                if type in ['meta', 'dict']:
                    print(f'    loading column: {name}')
                elif type in ['cols', 'dict']:
                    print(f'    loading {type}: {name}')

            if type == 'meta':
                self[name] = Column(
                    fname, cache_mem=self.cache_mem, verbose=self.verbose,
                )
            elif type == 'dict':
                self[name] = Dict(fname, verbose=self.verbose)
            elif type == 'cols':
                self[name] = Columns(
                    fname, cache_mem=self.cache_mem, verbose=self.verbose,
                )

        if verify:
            self.verify()

    def verify(self):
        """
        verify all array columns have the same length
        """

        first = True
        self._nrows = 0

        for c in self.keys():
            col = self[c]
            if col.type == 'col':
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

    def create_column(self, schema):
        """
        Create a new column, filling with zeros to the right number
        of rows

        Parameters
        ----------
        schema: ColumnSchema
            A ColumnSchema object

            schema = ColumnSchema(name='x', dtype='f4')
            cols.create_column(schema)
        """
        table_schema = TableSchema([schema])
        self._add_columns(table_schema, fill=True)

    def create_dict(self, name, data={}):
        """
        create a new dict column

        parameters
        ----------
        name: str
            Column name
        data: dict, optional
            Optional initial data, default {}
        """

        if name in self.dict_names:
            raise ValueError("column '%s' already exists" % name)

        filename = util.get_filename(dir=self.dir, name=name, type='dict')
        self[name] = Dict(filename, verbose=self.verbose)
        self[name].write(data)

    def create_sub(
        self, name, schema={}, cache_mem=DEFAULT_CACHE_MEM, verbose=False,
        overwrite=False,
    ):
        """
        Create a new sub-columns directory

        parameters
        ----------
        name: str
            Name for sub-columns
        schema: dict, optional
            Dictionary holding information for each column.
        cache_mem: str or number
            Cache memory for index creation, default '1g' or one gigabyte.
            Can be a number in gigabytes or a string
            Strings should be like '{amount}{unit}'
                '1g' = 1 gigabytes
                '100m' = 100 metabytes
                '1000k' = 1000 kilobytes
            Units can be g, m or k, case insenitive

        verbose: bool, optional
            If set to True, display information
        overwrite: bool, optional
            If the directory exists, remove existing data
        """

        if name in self.dict_names:
            raise ValueError("sub Columns '%s' already exists" % name)

        dirname = util.get_filename(dir=self.dir, name=name, type='cols')
        self[name] = Columns.create(
            dirname,
            schema=schema,
            cache_mem=cache_mem,
            verbose=self.verbose,
            overwrite=overwrite,
        )

    def create_sub_from_array(
        self,
        name,
        array,
        compression=None,
        chunksize=DEFAULT_CHUNKSIZE,
        append=True,
        cache_mem=DEFAULT_CACHE_MEM,
        verbose=False,
        overwrite=False,
    ):
        """
        Create a new sub-columns directory

        Parameters
        ----------
        name: str
            Name for sub-columns
        """

        if name in self.dict_names:
            raise ValueError("sub Columns '%s' already exists" % name)

        dirname = util.get_filename(dir=self.dir, name=name, type='cols')
        self[name] = Columns.from_array(
            dirname,
            array=array,
            compression=compression,
            chunksize=chunksize,
            append=append,
            cache_mem=cache_mem,
            verbose=self.verbose,
            overwrite=overwrite,
        )

    def reload(self, columns=None):
        """
        reload the database or a subset of columns.  Note discovery
        of new entries is not performed

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

    def clear(self):
        raise RuntimeError('clear() not supported on Columns')

    def _clear_all(self, name=None):
        """
        Clear out the dictionary of column info
        """
        super().clear()

    def _add_columns(self, schema, fill=False):
        """
        Initialize new columns.  Currently this must be done while all other
        columns are zero size to have consistency

        Parameters
        ----------
        schema: TableSchema or dict
            Schema holding information for each column.
        fill: bool, optional
            If set to True, the new columns are filled to zeros out to
            the current nrows
        """

        # Try to convert to schema
        if not isinstance(schema, TableSchema):
            schema = TableSchema.from_schema(schema)

        for name, this in schema.items():
            if self.verbose:
                print('    creating:', name)

            metafile = util.get_filename(self.dir, name, 'meta')
            if os.path.exists(metafile):
                raise RuntimeError(f'column {name} already exists')

            util.write_json(metafile, this)

            dfile = util.get_filename(self.dir, name, 'array')
            with open(dfile, 'w') as fobj:  # noqa
                # just to create the empty file
                pass

            if 'compression' in this:
                cfile = util.get_filename(self.dir, name, 'chunks')
                with open(cfile, 'w') as fobj:  # noqa
                    # just to create the empty file
                    pass

        if fill and self.nrows > 0:
            self._load(verify=False)
            for name in schema:
                self[name].resize(self.nrows)

        self._load()

    def append(self, data, verify=True):
        """
        Append data to the table columns

        Parameters
        ----------
        data: array or dict
            A structured array or dict with arrays
        verify: bool, optional
            If set to True, verify all the columns have the same number of rows
            after appending.  Default True
        """

        names = util.get_data_names(data)

        if len(self) > 0:
            # make sure the input data matches the existing column names
            in_names = set(names)
            column_names = set(self.column_names)
            if in_names != column_names:
                raise ValueError(
                    f'input columns {in_names} '
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
            raise RuntimeError(f'column {name} does not exist')

        self[name]._append(data, update_index=not self.is_updating)

    def updating(self):
        """
        This enteres the updating context, which delays index
        updates until after exiting the context

        with cols.updating():
            cols.append(data1)
            cols.append(data2)
        """
        self._is_updating = True
        return self

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

        original_names = list(self.keys())
        for name in original_names:
            self.delete_entry(name, yes=True)

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
            print(f'Removing data for sub columns: {name}')

            entry.delete(yes=True)
            fname = entry.filename
            if os.path.exists(fname):
                os.removedirs(fname)
        elif entry.type == 'dict':
            print(f'Removing data for dict: {name}')
            fname = entry.filename
            if os.path.exists(fname):
                os.remove(fname)
        else:
            print(f'Removing data for entry: {name}')
            for fname in entry.filenames:
                if os.path.exists(fname):
                    print(f'    Removing: {fname}')
                    os.remove(fname)

        del self[name]

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
        rows = util.extract_rows(rows=rows, nrows=self.nrows)

        if asdict:
            # Just putting the arrays into a dictionary.
            data = {}

            for colname in columns:

                if self.verbose:
                    print('    reading column: %s' % colname)

                # just read the data and put in dict, simpler than below
                col = self[colname]
                if col.type == 'col':
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

                if not asdict and self[c].type != 'col':
                    if not asdict:
                        raise ValueError(
                            "requested non-array column '%s' "
                            "Use asdict=True to read non-array "
                            "columns with general read() method" % c
                        )

        return columns

    def __enter__(self):
        self._is_updating = True
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self._is_updating = False

        for name in self.column_names:
            self[name].update_index()

    def __repr__(self):
        """
        Get a list of metadata for this columns directory and it's
        columns
        """
        ncols = len(self.column_names)
        indent = '  '
        s = []
        if self.dir is not None:
            # dbase = self._dirbase()
            # s += [dbase]
            s += ['dir: '+self.dir]
            if ncols > 0 and hasattr(self, '_nrows'):
                s += ['nrows: %s' % self.nrows]

        acols = []
        dicts = []
        subcols = []
        if len(self) > 0:
            acols += ['Table Columns:']
            cnames = 'name', 'dtype', 'comp', 'index'
            acols += ['  %-15s %6s %7s %-6s' % cnames]
            acols += ['  '+'-'*(35)]

            dicts += ['Dictionaries:']
            dicts += ['  %-15s' % ('name',)]
            dicts += ['  '+'-'*(28)]

            subcols = ['Sub-Columns Directories:']
            subcols += ['  %-15s' % ('name',)]
            subcols += ['  '+'-'*(28)]

            for name in sorted(self):
                c = self[name]

                if c.type == 'cols':
                    cdir = os.path.basename(c.dir).replace('.cols', '')
                    subcols += ['  %s' % cdir]
                else:

                    name = c.name

                    if len(name) > 15:
                        # name_entry = ['  %s' % name]
                        name_entry = [f'  {name}\n' + ' '*19]
                        # s += ['%23s' % (c.type,)]
                    else:
                        name_entry = ['  %-15s' % c.name]

                    if c.type == 'col':

                        if 'compression' in c.meta:
                            comp = c.meta['compression']['cname']
                        else:
                            comp = 'None'

                        acols += name_entry
                        c_dtype = c.dtype.descr[0][1]
                        acols[-1] += ' %6s' % c_dtype
                        acols[-1] += ' %7s' % comp
                        acols[-1] += ' %-6s' % self[name].has_index
                    elif c.type == 'dict':
                        dicts += name_entry
                    else:
                        raise ValueError(f'bad type: {c.type}')

        s = [indent + tmp for tmp in s]
        s = ['Columns: '] + s

        if len(acols) > 3:
            s += [indent]
            acols = [indent + tmp for tmp in acols]
            s += acols

        if len(dicts) > 3:
            s += [indent]
            dicts = [indent + tmp for tmp in dicts]
            s += dicts

        if len(subcols) > 3:
            s += [indent]
            subcols = [indent + tmp for tmp in subcols]
            s += subcols

        return '\n'.join(s)
