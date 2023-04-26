import os
import numpy as np
from . import defaults


def extract_rows(rows, nrows, sort=True):
    """
    extract rows for reading

    Parameters
    ----------
    rows: sequence, Indices, slice or None
        Possible rows to extract
    sort: bool, optional
        Whether to sort when converted to Indices

    Returns
    -------
    Indices, possibly sorted.  Note this does not make a copy of the data
    """
    from .indices import Indices

    if isinstance(rows, slice):
        s = extract_slice(rows, nrows)
        if s.step is not None:
            ind = np.arange(s.start, s.stop, s.step)
            output = Indices(ind, is_sorted=True)
        else:
            output = s
    elif rows is None:
        output = slice(0, nrows)
    elif isinstance(rows, Indices):
        output = rows
    else:
        output = Indices(rows)

    if isinstance(output, Indices) and sort:
        output.sort()

    return output


def extract_slice(s, nrows):
    start = s.start
    stop = s.stop

    if start is None:
        start = 0

    if stop is None or stop > nrows:
        stop = nrows

    return slice(start, stop, s.step)


def extract_colname(filename):
    """
    Extract the column name from the file name
    """

    bname = os.path.basename(filename)
    name = '.'.join(bname.split('.')[0:-1])
    return name


def extract_coltype(filename):
    """
    Extract the type from the file name
    """
    return filename.split('.')[-1]


def get_filename(dir, name, type):
    """
    genearte a file name from dir, column name and column type
    """
    if type not in defaults.ALLOWED_COL_TYPES:
        raise ValueError(f'unknown file type {type}')

    return os.path.join(dir, f'{name}.{type}')


def meta_to_colfiles(metafile):
    dir, bname = os.path.split(metafile)
    name = extract_colname(metafile)
    return {
        'dir': dir,
        'name': name,
        'array': get_filename(dir, name, 'array'),
        'index': get_filename(dir, name, 'index'),
        'index1': get_filename(dir, name, 'index1'),
        'sorted': get_filename(dir, name, 'sorted'),
        'chunks': get_filename(dir, name, 'chunks'),
    }


def read_json(fname):
    """
    wrapper to read json
    """
    import json

    with open(fname) as fobj:
        data = json.load(fobj)
    return data


def write_json(fname, obj):
    """
    wrapper for writing json
    """
    import json

    with open(fname, 'w') as fobj:
        json.dump(obj, fobj, indent=1, separators=(',', ':'))


def get_native_data(data):
    """
    get version of the structured array with native
    byte ordering

    This version works even when the byte ordering is
    mixed
    """
    newdt = []
    for n in data.dtype.names:
        col = data[n]
        dtstr = col.dtype.descr[0][1][1:]
        shape = col.shape
        if len(shape) > 1:
            descr = (n, dtstr, shape[1:])
        else:
            descr = (n, dtstr)

        newdt.append(descr)

    new_data = np.zeros(data.size, dtype=newdt)
    for n in data.dtype.names:
        new_data[n] = data[n]

    return new_data


def convert_to_gigabytes(s):
    try:
        gigs = float(s)
    except ValueError:
        slow = s.lower()

        units = slow[-1]
        amount = slow[:-1]
        if units == 'g':
            gigs = float(amount)
        elif units == 'm':
            gigs = float(amount) / 1000
        elif units == 'k':
            gigs = float(amount) / 1000 / 1000
        else:
            raise ValueError(f'band unit in {s}')

    return gigs


def array_to_schema(array, compression=None):
    """
    Create a schema from the fields in the input array

    Parameters
    ----------
    array: numpy array
        An array with fields
    compression: list or dict, optional
        An optional list or dictionary with compression information for
        specific columns.

        If the input is a list of names, then default compression values
        are used (see pycolumns.DEFAULT_COMPRESSION, pycolumns.DEFAULT_CLEVEL)

        e.g. compression=['id', 'name']

        If the input is a dict, then each dict entry can specify
        the compression and clevel.

        e.g.
        compression={
            'id': {'compressor': 'zstd':, 'clevel': 5}
            'name': {'compressor': 'zstd':},
            'x': {},  # means use defaults
        }

        If the entry is itself an empty dict {}, then defaults are filled in
        from pycolumns.DEFAULT_COMPRESSION, pycolumns.DEFAULT_CLEVEL,
        pycolumns.DEFAULT_SHUFFLE

        You can also add compression to some columns after the fact

        import pycolumns as pyc
        schema = pyc.array_to_schema(array)
        schema['id']['compressor'] = 'zstd'
        schema['id']['clevel'] = 5

    Returns
    -------
    A schema
    """
    if array.dtype.names is None:
        raise ValueError('array must have fields')

    schema = {}

    for name in array.dtype.names:
        schema[name] = {'dtype': array[name].dtype.str}

    if compression is not None:
        schema = add_schema_compression(schema, compression)

    return schema


def add_schema_compression(schema, compression):
    """
    Convenience function to get a new schema with compression settings added,
    falling back to defaults as needed

    Parameters
    ----------
    schema: dict
        A schema
    compression: list or dict, optional
        An optional list or dictionary with compression information for
        specific columns.

        If the input is a list of names, then default compression values
        are used (see pycolumns.DEFAULT_COMPRESSION, pycolumns.DEFAULT_CLEVEL)

            e.g. compression=['id', 'name']

        If the input is a dict, then each dict entry can specify
        the compression and clevel.

            e.g.
            compression={
                'id': {'compressor': 'zstd':, 'clevel': 5}
                'name': {'compressor': 'zstd':},
                'x': {},  # means use defaults
            }

        If the entry is itself an empty dict {}, then defaults are filled in.
        See pycolumns.defaults.DEFAULT_COMPRESSION

    Returns
    -------
    A schema with compression possibly set for some columns
    """

    new_schema = schema.copy()

    if hasattr(compression, 'keys'):
        isdict = True
    else:
        isdict = False

    for name in compression:

        if name in new_schema:
            this = new_schema[name]

            # start with defaults and then update if detailed settings
            # were entered
            this.update(defaults.DEFAULT_COMPRESSION)

            if isdict:
                # now update to the input
                this.update(compression[name])

    return new_schema


def get_compression_with_defaults(compression=None, convert=False):
    """
    get compression with defaults set

    Parameters
    ----------
    compression: dict, optional
        If not sent, the default compression is returned
        If sent, defaults are filled in as needed
    convert: bool, optional
        If set to True, convert the shuffle to the blosc integer
        value

    Returns
    -------
    dict with compression set
    """
    comp = defaults.DEFAULT_COMPRESSION.copy()
    if compression is not None:
        comp.update(compression)
    if convert:
        comp['shuffle'] = convert_shuffle(comp['shuffle'])
    return comp


def convert_shuffle(shuffle):
    """
    convert shuffle to the integer value

    Parameters
    ----------
    shuffle: str or int
        'shuffle', blosc.SHUFFLE, 'bitshuffle' or blosc.BITSHUFFLE
        String is case insensitive

    Returns
    -------
    The integer value, e.g. blosc.SHUFFLE
    """
    import blosc

    shuffle = shuffle.upper()

    if shuffle in ('SHUFFLE', blosc.SHUFFLE):
        new_shuf = blosc.SHUFFLE
    elif shuffle in ('BITSHUFFLE', blosc.BITSHUFFLE):
        new_shuf = blosc.BITSHUFFLE
    else:
        raise ValueError(f'bad shuffle: {shuffle}')
    return new_shuf


def schema_to_dtype(schema):
    """
    convert a schema into a numpy dtype

    Parameters
    ----------
    schema: dict
        Dict names for fields and dtype
    """
    descr = []

    for name in schema:
        descr.append((name, schema[name]['dtype']))

    return np.dtype(descr)
