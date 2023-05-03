import os
import numpy as np
from . import defaults


def extract_rows(rows, nrows, sort=True, check_slice_stop=False):
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
        s = extract_slice(rows, nrows, check_slice_stop=check_slice_stop)
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


def extract_slice(s, nrows, check_slice_stop=False):
    start = s.start
    stop = s.stop
    if stop is not None and check_slice_stop:
        if stop > nrows:
            raise ValueError(f'slice stop {stop} > nrows {nrows}')

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
        json.dump(obj, fobj, indent=4, separators=(',', ':'))


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
            gigs = float(amount) / 1024
        elif units == 'k':
            gigs = float(amount) / 1024 ** 2
        elif units == 'b':
            gigs = float(amount) / 1024 ** 3
        else:
            raise ValueError(f'band unit in {s}')

    return gigs


def convert_to_bytes(s):
    try:
        bts = float(s)
    except ValueError:
        slow = s.lower()

        units = slow[-1]
        amount = slow[:-1]
        if units == 'g':
            bts = float(amount) * 1024 ** 3
        elif units == 'm':
            bts = float(amount) * 1024 ** 2
        elif units == 'k':
            bts = float(amount) * 1024
        elif units == 'b':
            bts = float(amount)
        else:
            raise ValueError(f'band unit in {s}')

    return bts


def get_compression_with_defaults(compression=None, convert=False):
    """
    get compression with defaults set

    Parameters
    ----------
    compression: dict, optional
        If a dict is sent, defaults are filled in as needed, otherwise defaults
        are returned
    convert: bool, optional
        If set to True, convert the shuffle to the blosc integer
        value

    Returns
    -------
    dict with compression set
    """

    comp = defaults.DEFAULT_COMPRESSION.copy()
    if hasattr(compression, 'keys'):
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
        'noshuffle', blosc.NOSHUFFLE
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
    elif shuffle in ('NOSHUFFLE', blosc.NOSHUFFLE):
        new_shuf = blosc.NOSHUFFLE
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


def get_chunks(chunkrows_sorted, rows):
    """
    Get chunk assignments for the input rows

    Parameters
    ----------
    chunkrows_sorted: array
        Sorted array of chunk row start positions
    rows: array
        Rows to assign

    Returns
    -------
    chunk_indices: array
        chunk index for each row
    """
    s = np.searchsorted(chunkrows_sorted, rows, side='right')
    s -= 1
    s.clip(min=0, max=chunkrows_sorted.size-1, out=s)
    return s


def get_data_names(data):
    """
    Get names for the data, either keys for a dict of arrays or names
    from a structured array

    Parameters
    ----------
    data: array or dict
        A structured array or dict with arrays

    Returns
    -------
    List of names
    """
    if hasattr(data, 'keys'):
        names = list(data.keys())
    else:
        names = data.dtype.names
        if names is None:
            raise ValueError('array must have fields')

    return names


def byteswap_inplace(data):
    """
    Byte swap the data in place and with new dtype
    """
    data.byteswap(inplace=True)
    data.dtype = data.dtype.newbyteorder()


def get_data_with_conversion(data, dtype, ndmin=1):
    """
    returns the data, possibly converted to the specified type,
    otherwise just a new ref
    """
    return np.array(data, ndmin=ndmin, dtype=dtype, copy=False)
