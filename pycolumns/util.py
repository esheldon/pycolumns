import os
import numpy as np
from . import defaults


def extract_rows(rows, nrows, check_slice_stop=False):
    """
    extract rows for reading

    Parameters
    ----------
    rows: sequence, Indices, slice or None
        Possible rows to extract
    nrows: int
        Total number of rows
    check_slice_stop: bool, optional
        this is for doing row updates and when we need
        the slice to be exact, not go beyond nrows

    Returns
    -------
    Indices, or slice.  possibly sorted.
    """
    from .indices import Indices

    if isinstance(rows, Indices) and rows.is_checked:
        return rows

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

    if isinstance(output, Indices):
        if output.ndim == 0 and output < 0:
            output = Indices(nrows + output, is_checked=True)
        else:
            w, = np.where(output < 0)
            if w.size > 0:
                # make a copy since we don't want to modify underlying
                # input data
                output = output.copy()
                output[w] += nrows
                output.is_checked = True
                assert output.is_checked

    return output


def extract_slice(s, nrows, check_slice_stop=False):
    start = s.start
    stop = s.stop

    if stop is not None and check_slice_stop:
        # this is for doing row updates and when we need
        # the slice to be exact, not go beyond nrows
        if stop > nrows:
            raise IndexError(f'slice stop {stop} > nrows {nrows}')

    if start is None:
        start = 0
    if stop is None:
        stop = nrows

    if start < 0:
        start = nrows + start
        if start < 0:
            raise IndexError("Index out of bounds")

    if stop < 0:
        stop = nrows + stop

    if stop < start:
        # will return an empty struct
        stop = start

    if stop is None or stop > nrows:
        stop = nrows

    return slice(start, stop, s.step)


def get_subname(name):
    """
    sub-cols get a leading slash
    """
    return f'/{name}'


def extract_name(filename):
    """
    Extract the column name from the file name
    """

    name, _ = split_ext(filename)
    return name
    # n, ext = split_ext(filename)
    # if ext == 'cols':
    #     return f'/{n}'
    # else:
    #     return n
    # bname = os.path.basename(filename)
    # name = '.'.join(bname.split('.')[0:-1])
    # return name


def extract_type(filename):
    """
    Extract the type from the file name
    """
    return extract_extension(filename)


def extract_extension(filename):
    """
    Extract the extension
    """
    _, ext = split_ext(filename)
    return ext


def split_ext(filename):
    """
    For /path/to/blah.txt returns ('blah', 'txt')
    """
    bname = os.path.basename(filename)
    s = bname.split('.')
    if len(s) == 1:
        name = s[0]
        ext = ''
    else:
        name = '.'.join(s[0:-1])
        ext = s[-1]
    return name, ext


def get_meta_filename(path):
    """
    Get a path to a .meta file assuming path is a column directory
    """
    bname = os.path.basename(path)
    return f'{path}/{bname}.meta'


def is_column(path):
    """
    Returns True if path is a directory (or link) containing
    a file named basename(path).meta
    """
    metapath = get_meta_filename(path)
    if os.path.exists(metapath):
        return True
    else:
        return False


def get_column_dir(dir, name):
    """
    get path to column directory
    """
    return os.path.join(dir, name)


def get_filename(dir, name, ext):
    """
    genearte a file name from dir, column name and column type
    """
    if ext not in defaults.ALLOWED_EXTENSIONS:
        raise ValueError(f'unsuported extension {ext}')

    return os.path.join(dir, f'{name}.{ext}')


def get_colfiles(coldir):
    name = os.path.basename(coldir)
    return {
        'dir': coldir,
        'name': name,
        'meta': get_filename(coldir, name, 'meta'),
        'array': get_filename(coldir, name, 'array'),
        'index': get_filename(coldir, name, 'index'),
        'index1': get_filename(coldir, name, 'index1'),
        'sorted': get_filename(coldir, name, 'sorted'),
        'chunks': get_filename(coldir, name, 'chunks'),
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
    try:
        ndata = np.array(data, ndmin=ndmin, dtype=dtype, copy=False)
    except ValueError as err:
        if isinstance(data, np.ndarray):
            raise ValueError(
                f'Could not convert data of type {data.dtype} to {dtype}: '
                f'{err}'
            )
        else:
            raise

    return ndata
