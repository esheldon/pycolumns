import os
import json


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
