"""
TODO

    - support multiple levels '/sub1/sub2/sub3'
    - setters for some things like cache_mem, verbose etc.
    - auto partitioning of data
        - specify on creation that values in a column will
          be used to automatically partition rows of the data.
          e.g. could specify mdet_step and will automatically
          generate subdirectories and put rows in there.

          The idea is that if one does cols['mdet_step'] == 'noshear'
          then it would automatically limit you to that partition

          Issue is that the current query stuff won't understand that
"""
import os
import numpy as np
from .column import Column
from .metafile import Meta
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
                '200000b' = 200000 bytes
        Units can be g, m, k or b case insenitive

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
    def meta_names(self):
        """
        Get a list of the array column names
        """
        return list(self.meta.keys())

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

    @property
    def meta(self):
        """
        Do this properly to add protections and controls
        """
        return self._meta

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
        # clear out the existing columns and start from scratch
        self._clear_all()

        paths = os.listdir(self.dir)

        # load the table columns
        for i, path in enumerate(paths):
            path = os.path.join(self.dir, path)
            c = None
            if util.is_column(path):
                name = os.path.basename(path)
                c = Column(
                    path, cache_mem=self.cache_mem, verbose=self.verbose,
                )
                super().__setitem__(name, c)
            else:

                # will be /name for name.cols
                # else will be name
                name, ext = util.split_ext(path)
                # name = util.extract_name(path)
                # ext = util.extract_extension(path)

                if self.verbose and ext in ['cols', 'json']:
                    print(f'    loading : {name}')

                # only load .dict or .cols, ignore other stuff
                if ext == 'json':
                    self.meta._load(path)
                elif ext == 'cols':
                    c = Columns(
                        path, cache_mem=self.cache_mem, verbose=self.verbose,
                    )
                    cname = util.get_subname(name)
                    super().__setitem__(cname, c)
                else:
                    pass

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
        Create a new column, filling with zeros or the fill value to the right
        number of rows

        Parameters
        ----------
        schema: ColumnSchema
            A ColumnSchema object

            schema = ColumnSchema(name='x', dtype='f4')
            cols.create_column(schema)
        """
        table_schema = TableSchema([schema])
        self._add_columns(table_schema, fill=True)

    def create_meta(self, name, data={}):
        """
        create a new metadata entry

        parameters
        ----------
        name: str
            Name for this metadata entry
        data: Data that can be stored as JSON
            e.g. a dict, list etc.
        """

        if name in self.meta_names:
            raise ValueError("column '%s' already exists" % name)

        path = util.get_filename(dir=self.dir, name=name, ext='json')
        self.meta._load(path)
        self.meta[name].write(data)

    def create_sub(
        self, name, schema={}, cache_mem=DEFAULT_CACHE_MEM, verbose=False,
        overwrite=False,
    ):
        """
        Create a new sub Columns directory

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
                '200000b' = 200000 bytes
            Units can be g, m, k or b case insenitive

        verbose: bool, optional
            If set to True, display information
        overwrite: bool, optional
            If the directory exists, remove existing data
        """

        if name[0] != '/':
            raise ValueError(
                f'sub-Columns names must have a leading /, got {name}'
            )

        if name in self.subcols_names:
            raise ValueError("sub Columns '%s' already exists" % name)

        dname = name[1:]
        dirname = util.get_filename(dir=self.dir, name=dname, ext='cols')
        c = Columns.create(
            dirname,
            schema=schema,
            cache_mem=cache_mem,
            verbose=self.verbose,
            overwrite=overwrite,
        )
        super().__setitem__(name, c)

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
        Create a new sub Columns directory

        Parameters
        ----------
        name: str
            Name for sub Columns
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
                '200000b' = 200000 bytes
        Units can be g, m, k or b case insenitive

        verbose: bool, optional
            If set to True, display information
        overwrite: bool, optional
            If the directory exists, remove existing data
        """

        if name[0] != '/':
            raise ValueError(
                f'sub-Columns names must have a leading /, got {name}'
            )

        if name in self.subcols_names:
            raise ValueError("sub Columns '%s' already exists" % name)

        dname = name[1:]
        dirname = util.get_filename(dir=self.dir, name=dname, ext='cols')
        c = Columns.from_array(
            dirname,
            array=array,
            compression=compression,
            chunksize=chunksize,
            append=append,
            cache_mem=cache_mem,
            verbose=self.verbose,
            overwrite=overwrite,
        )
        super().__setitem__(name, c)

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
        for col in self.column_names:
            self[col]._close()

        super().clear()
        self._meta = _MetaSet(coldir=self.dir, verbose=self.verbose)

    def _add_columns(self, schema, fill=False):
        """
        Initialize new columns.  Currently this must be done while all other
        columns are zero size to have consistency

        Parameters
        ----------
        schema: TableSchema or dict
            Schema holding information for each column.
        fill: bool, optional
            If set to True, the new columns are filled out to the current nrows
        """

        # Try to convert to schema
        if not isinstance(schema, TableSchema):
            schema = TableSchema.from_schema(schema)

        for name, this in schema.items():
            if self.verbose:
                print('    creating:', name)

            coldir = util.get_column_dir(self.dir, name)
            if not os.path.exists(coldir):
                os.makedirs(coldir)

            metafile = util.get_filename(coldir, name, 'meta')
            if os.path.exists(metafile):
                raise RuntimeError(f'column {name} already exists')

            util.write_json(metafile, this)

            dfile = util.get_filename(coldir, name, 'array')
            with open(dfile, 'w') as fobj:  # noqa
                # just to create the empty file
                pass

            if 'compression' in this:
                cfile = util.get_filename(coldir, name, 'chunks')
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

    def vacuum(self):
        """
        Degragment compressed columns

        When updating data in compressed columns, the new compressed chunks can
        expand beyond their allocated region in the file.  In this case the new
        compressed data is stored temporarily in a separate file.  Running
        vacuum combines all data back together in a single contiguous file.
        """
        for name in self.column_names:
            self[name].vacuum()

    def delete(self, yes=False):
        """
        delete the entire Columns database

        Parameters
        ----------
        yes: bool
            If True, don't prompt for confirmation
        """
        import shutil

        if not yes:
            answer = input('really delete all data? (y/n) ')
            if answer.lower() == 'y':
                yes = True

        if not yes:
            return

        original_names = list(self.keys())
        for name in original_names:
            self.delete_entry(name, yes=True)

        for name in self.meta:
            self.delete_meta(name, yes=True)

        print('removing:', self.dir)
        shutil.rmtree(self.dir)

    def delete_meta(self, name, yes=False):
        """
        delete the specified dict

        parameters
        ----------
        name: string
            Name of entry to delete
        yes: bool
            If True, don't prompt for confirmation
        """

        if name not in self.meta:
            print("cannot delete dict '%s', it does not exist" % name)

        if not yes:
            answer = input("really delete dict '%s'? (y/n) " % name)
            if answer.lower() == 'y':
                yes = True

        if yes:
            print(f'Removing data for dict: {name}')
            fname = self.meta[name].filename
            if os.path.exists(fname):
                os.remove(fname)

            del self.meta[name]

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
        import shutil

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

            # have the sub cols remove data in case the subcols dir is a sym
            # link
            dname = entry.dir

            entry.delete(yes=True)
            if os.path.exists(dname):
                shutil.rmtree(dname)

        else:
            # remove individual files in case it the directory is a sym link
            print(f'Removing data for entry: {name}')
            for fname in entry.filenames:
                if os.path.exists(fname):
                    print(f'    Removing: {fname}')
                    os.remove(fname)

            print(f'Removing: {entry.dir}')
            shutil.rmtree(entry.dir)

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
            array columns.
        rows: sequence, slice or scalar
            Sequence of row numbers.  Defaults to all.
        asdict: bool, optional
            If set to True, read the requested columns into a dict rather
            than structured array
        """

        columns = self._extract_columns(columns=columns)

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

    def _extract_columns(self, columns=None):
        """
        extract the columns to read.  If no columns are sent then all
        columns are read
        """
        if columns is None:
            return self.column_names
        else:
            if isinstance(columns, str):
                columns = [columns]

            for c in columns:
                if c not in self.column_names:
                    raise ValueError("Column '%s' not found" % c)

        return columns

    def __setitem__(self, name, data):
        """
        Only supported for dict.  For Column you need to do
        cols[name][ind] = 3 etc.
        """
        item = self[name]
        if item.type == 'dict':
            item.write(data)
        elif item.type == 'col':
            # let the error handling occur in Column
            self[name][:] = data
        elif item.type == 'cols':
            raise TypeError(
                f'Attempt to replace entire sub Columns "{name}". '
                f'If you are trying to set items for a Column inside '
                f'{name}, use cols[{name}][indices] = data etc.'
            )
        else:
            raise TypeError('Columns object does not support item assignment')

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

        metas = []
        if len(self.meta_names) > 0:
            metas += ['Metadata:']
            metas += ['  %-15s' % ('name',)]
            metas += ['  '+'-'*(28)]
            metas += ['  '+n for n in self.meta_names]

        subcols = []
        if len(self.subcols_names) > 0:
            subcols += ['Sub-Columns Directories:']
            subcols += ['  %-15s' % ('name',)]
            subcols += ['  '+'-'*(28)]
            subcols += ['  '+n for n in self.subcols_names]

        acols = []

        column_names = self.column_names
        if len(column_names) > 0:
            acols += ['Table Columns:']
            cnames = 'name', 'dtype', 'comp', 'index'
            acols += ['  %-15s %6s %7s %-6s' % cnames]
            acols += ['  '+'-'*(35)]

            for name in sorted(column_names):
                c = self[name]

                name = c.name

                if len(name) > 15:
                    # name_entry = ['  %s' % name]
                    name_entry = [f'  {name}\n' + ' '*19]
                    # s += ['%23s' % (c.type,)]
                else:
                    name_entry = ['  %-15s' % c.name]

                if 'compression' in c.meta:
                    comp = c.meta['compression']['cname']
                else:
                    comp = 'None'

                acols += name_entry
                c_dtype = c.dtype.descr[0][1]
                acols[-1] += ' %6s' % c_dtype
                acols[-1] += ' %7s' % comp
                acols[-1] += ' %-6s' % self[name].has_index

        s = [indent + tmp for tmp in s]
        s = ['Columns: '] + s

        if len(acols) > 3:
            s += [indent]
            acols = [indent + tmp for tmp in acols]
            s += acols

        if len(metas) > 0:
            s += [indent]
            metas = [indent + tmp for tmp in metas]
            s += metas

        if len(subcols) > 0:
            s += [indent]
            subcols = [indent + tmp for tmp in subcols]
            s += subcols

        return '\n'.join(s)


class _MetaSet(dict):
    """
    Manage a set of Meta
    """
    def __init__(self, coldir, verbose):
        self._coldir = coldir
        self._verbose = verbose

    def __getitem__(self, name):
        if name not in self:
            raise RuntimeError(f'dict {name} not found')

        return super().__getitem__(name)

    def __setitem__(self, name, data):
        raise TypeError(
            f'To write to dict {name}, use meta[{name}].write(data) '
            f'or meta[{name}].update(data)'
        )

    def _load(self, path):
        name = util.extract_name(path)
        if name in self:
            raise RuntimeError(f'dict {name} already exists')

        c = Meta(path, verbose=self._verbose)
        super().__setitem__(name, c)

    def __repr__(self):
        s = ['Metadata:']
        s += ['  '+n for n in self]
        return '\n'.join(s)
