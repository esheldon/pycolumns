import numpy as np
from .defaults import DEFAULT_COMPRESSION, DEFAULT_CHUNKSIZE


class TableSchema(dict):
    """
    A table Schema

    Parameters
    ----------
    schemas: [ColumnSchema], optional
        A list of ColumnSchema to be added to the table schema.
        You can add more using schema.add_column(schema)
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
        compression: dict, bool, or None, optional
            If None (the default) or False, no compression is used for any
            column.

            If True, the default compression settings are used for all columns.

            Otherwise a dict keyed by name describing the compression for
            each column to be compressed.  See ColumnSchema for the the options

        chunksize: dict, str or number
            A dit or str or number

            if str or number, it is applied to all compressed columns.

            Otherwise a dict keyed by name giving chunksize for each column.
            See ColumnSchema for the the options

        Returns
        -------
        TableSchema
        """
        if array.dtype.names is None:
            raise ValueError('array must have fields')

        if compression is True:
            is_dict = False
        else:
            is_dict = True

        table_schema = TableSchema()
        for name in array.dtype.names:

            keys = {}
            if compression and is_dict:
                if is_dict and name in compression:
                    keys['compression'] = compression[name]
            elif compression and not is_dict:
                keys['compression'] = compression

            if 'compression' in keys:
                if name in chunksize:
                    keys['chunksize'] = chunksize[name]
                else:
                    keys['chunksize'] = chunksize

            schema = ColumnSchema(
                name=name,
                dtype=array[name].dtype,
                **keys
            )
            table_schema.add_column(schema)

        return table_schema

    @classmethod
    def from_dict(cls, sdict):
        """
        Convert a dict to a TableSchema

        Parameters
        ----------
        sdict: dict
            A dict representing a table schema, e.g.
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

            schema = ColumnSchema(
                name=name,
                **dschema
            )
            table_schema.add_column(schema)

        return table_schema


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
