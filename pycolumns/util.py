import os
import json
import numpy as np


def extract_rows(rows, sort=True):
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

    if (
        rows is not None
        and not isinstance(rows, slice)
        and not isinstance(rows, Indices)
    ):
        output = Indices(rows)
    else:
        output = rows

    if isinstance(rows, Indices) and sort:
        output.sort()

    return output


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


def create_filename(dir, name, type):
    """
    genearte a file name from dir, column name and column type
    """
    if dir is None:
        raise ValueError('Cannot create column filename, dir is None')

    if name is None:
        raise ValueError('Cannot create column filename: name is None')

    if type is None:
        raise ValueError('Cannot create column filename: type is None')

    if type == 'dict':
        ext = 'json'
    elif type == 'array':
        ext = 'array'
    else:
        raise ValueError("bad file type: '%s'" % type)

    return os.path.join(dir, name+'.'+ext)


def read_json(fname):
    """
    wrapper to read json
    """

    with open(fname) as fobj:
        data = json.load(fobj)
    return data


def write_json(obj, fname, pretty=True):
    """
    wrapper for writing json
    """

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


def maybe_decode_fits_ascii_strings_to_unicode_py3(array):
    new_dtype, do_conversion = (
        maybe_convert_ascii_dtype_to_unicode(array.dtype)
    )
    if do_conversion:
        array = array.astype(new_dtype, copy=False)
    return array


def maybe_convert_ascii_dtype_to_unicode(dtype):

    do_conversion = False
    new_dt = []
    for dt in dtype.descr:
        if 'S' in dt[1]:
            do_conversion = True
            if len(dt) == 3:
                new_dt.append((
                    dt[0],
                    dt[1].replace('S', 'U').replace('|', ''),
                    dt[2]))
            else:
                new_dt.append((
                    dt[0],
                    dt[1].replace('S', 'U').replace('|', '')))
        else:
            new_dt.append(dt)

    new_dtype = np.dtype(new_dt)
    return new_dtype, do_conversion
