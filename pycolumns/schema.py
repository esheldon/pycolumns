import numpy as np
from . import util
from .defaults import DEFAULT_COMPRESSION, DEFAULT_CHUNKSIZE


class TableSchema(dict):
    """
    A table Schema

    Parameters
    ----------
    schemas: [ColumnSchema], optional
        A list of ColumnSchema to be added to the table schema.
        You can add more using schema.add_column(schema)

    Examples
    --------

    # Construct from ColumnSchema objects
    cslist = [
        ColumnSchema(name='id', dtype='i8', compression=True),
        ColumnSchema(name='ra', dtype='f8'),
        ColumnSchema(name='name', dtype='U5',
                     compression={'shuffle': 'shuffle'},
                     chunksize='0.5m'),
    ]
    schema = TableSchema(cslist)

    # Or you can add one by one
    schema = TableSchema(cslist)
    for cs in cslist:
        schema.add(cs)

    # One can also construct TableSchema from dicts, other TableSchema and
    # arrays (with fields) using class methods

    schema = TableSchema.from_array(array, compression=c, chunksize=ch)
    schema = TableSchema.from_schema(some_dict)
    schema = TableSchema.from_columns(cslist)  # same as using constructor

    Layout
    --------
    # A TableSchema inherits from a dict. It may look something like this
    {
        'id': {'dtype': '<i8'}
        'ra': {'dtype': '<f8'}
        'dec': {'dtype': '<f8'}
        'name': {'dtype': '<U5'}
    }

    # Compression and chunksize for compressed files can be set
    {
        'id': {
            'dtype': '<i8',
            'compression': {
                'cname': 'zlib',
                'clevel': 5,
                'shuffle': 'bitshuffle',
            }
            'chunksize': '1m',
        },
        'ra': {'dtype': '<f8'}
    }
    """
    def __init__(self, schemas=None):
        if schemas is None:
            return

        for schema in schemas:
            self.add_column(schema)

    def add_column(self, schema):
        """
        add a column schema
        """
        self[schema.name] = schema.copy()

    @classmethod
    def from_array(cls, array, compression=None, chunksize=DEFAULT_CHUNKSIZE):
        """
        Convert an array with fields to a TableSchema

        Parameters
        ----------
        array: numpy array with fields or dict of arrays
            An array with fields defined, or a dict holding arrays
        compression: dict, list, bool, or None, optional
            - If None or False, do not set compression
            - If True, use default compression
            - If a list, return use default compression if the name is in the
              list
            - If a dict, and name is an entry
                 - if value is True, use default compression
                 - if value is a dict, return the dict with defaults set for
                   non specified parameters
        chunksize: dict, str or number
            A dict or str or number

            - if str or number, it is applied to all compressed columns.
            - a dict,  keyed by name, it gives chunksize for certain columns.

        Returns
        -------
        TableSchema
        """
        names = util.get_data_names(array)

        table_schema = TableSchema()
        for name in names:
            keys = {}
            comp = _get_column_compression(compression, name)
            if comp:
                # Note, if comp is a dict, it gets defaults set for
                # non specified compression entries in ColumnSchema
                keys['compression'] = comp
                keys['chunksize'] = _get_column_chunksize(chunksize, name)

            schema = ColumnSchema(
                name=name,
                dtype=array[name].dtype,
                **keys
            )
            table_schema.add_column(schema)

        return table_schema

    @classmethod
    def from_schema(cls, sdict):
        """
        Convert a TableSchema or dict to a TableSchema

        Parameters
        ----------
        sdict: dict or TableSchema
            e.g.
            {
              'id': {
                'dtype': 'f8',
                'compression': True,
              },
              'ra': {'dtype': 'f8'},
            }

        Returns
        -------
        TableSchema
        """

        table_schema = TableSchema()
        for name, dschema in sdict.items():

            schema = ColumnSchema(name=name, **dschema)
            table_schema.add_column(schema)

        return table_schema

    @classmethod
    def from_columns(cls, schemas):
        """
        Construct a new table Schema from a list of ColumnSchema objects

        Parameters
        ----------
        schemas: [ColumnSchema], optional
            A list of ColumnSchema to be added to the table schema.
            You can add more using schema.add_column(schema)
        """
        return TableSchema(schemas)


class ColumnSchema(dict):
    """
    A schema for a column

    Parameters
    ----------
    dtype: numpy dtype
        A numpy dtype for the column
    compression: dict, bool, or None, optional
        If None (the default) or False, no compression is used.
        If True, the default compression settings are used.

        Otherwise a dict describing the compression.  Any missing
        entries are filled in with defaults from pycolumns.DEFAULT_COMPRESSION
        compression = {
            'cname': 'zstd',
            'clevel': 5,
            'shuffle': 'bitshuffle',
        }
    chunksize: str or number
        The chunksize for compressed columns.  Default is
        pycolumns.DEFAULT_CHUNKSIZE.

        This describes the size of the uncompressed chunk.  Can be a string
        with units or a number, in which case it is interpreted as bytes

            '1m': 1 megabyte
            '2k': 2 kilobytes
            '0.5g': 0.5 gigabytes
            '200b': 200 bytes
            '1000r': 1000 rows
    """
    def __init__(
        self,
        name,
        dtype,
        compression=None,
        chunksize=DEFAULT_CHUNKSIZE,
    ):
        self._name = name
        schema = _make_array_schema_dict(
            dtype=dtype,
            compression=compression,
            chunksize=chunksize,
        )
        self.update(schema)

    @property
    def name(self):
        """
        get the column name
        """
        return self._name


def _make_array_schema_dict(
    dtype,
    compression=None,
    chunksize=DEFAULT_CHUNKSIZE,
):
    schema = {
        'dtype': np.dtype(dtype).str,
    }
    if compression:
        schema['chunksize'] = chunksize
        schema['compression'] = DEFAULT_COMPRESSION.copy()
        if compression is not True:
            schema['compression'].update(compression)

    return schema


def _get_column_compression(compression, name):
    """
    extract compression info for a specified column

    Parameters
    ----------
    compression: bool, dict or list
        If a bool, return None for False or defaults for True
        If a list, return defaults if the name is in the list
        If a dict, return the entry or None
    name: str
        Column name

    Returns
    -------
    None or a dict
    """
    if hasattr(compression, 'keys'):
        tcomp = compression.get(name)
        if tcomp:
            comp = tcomp
        else:
            comp = None
    elif _has_len(compression):
        if name in compression:
            comp = True
        else:
            comp = None
    else:
        if not compression:
            comp = None
        else:
            comp = True

    return comp


def _get_column_chunksize(chunksize, name):
    """
    extract chunksize for a specified column

    Parameters
    ----------
    chunksize: dict or str or number
        If an str or number, just return
        If a dict, return the entry if found or None
    name: str
        Column name

    Returns
    -------
    str or number
    """
    if hasattr(chunksize, 'keys'):
        return chunksize.get(name, DEFAULT_CHUNKSIZE)
    else:
        # catch a bug
        if hasattr(chunksize, 'append') or hasattr(chunksize, 'discard'):
            raise ValueError(
                'expected dict, str, or number, got {type(chunksize)}'
            )
        return chunksize


def _has_len(x):
    try:
        len(x)
        ret = True
    except TypeError:
        ret = False

    return ret
