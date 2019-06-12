"""
Read and write numpy arrays to a simple file format.  The format is a
simple ascii header followed by data in binary format.
"""
import os
import pprint

import numpy as np

SIMPLEFILE_VERSION = '0.1'


class SimpleFile(object):
    """
    This class implements a simple file format for holding numerical python
    arrays.  The format is a simple ascii header followed by data in
    binary form

    The format is designed to represent a column of data, which can itself be
    multi-dimensional.  This is not designed to represent a full rec array, for
    that see e.g. fitsio or esutil.sfile

    Examples
    ---------

    # Open and read from a .sf file
    # the
    with SimpleFile('test.sf') as sf:

        # see what's in the file.
        print(sf)
        filename: 'test.sf'
        mode: 'r'
        size: 64526192
        dtype: '<f4'
        hdr:
          {'dataset': 'dc4',
           'pyvers':'v2.6.2'}

        #
        # A few ways to read all the data
        #

        data = sf[:]
        data = sf.read()

        # read a subset of rows.  Can use slices, single numbers for rows,
        # or list/array of row numbers.

        data = sf[35]
        data = sf[35:100]
        data = sf[ [35,66,22,23432] ]
        data = sf[ row_list ]

    with SimpleFile('test.sf','w+') as sf:
        sf.write(data)

        # this appends data
        sf.write(more_data)

        # check the first row of the data we have written
        sf[0]

        sf.write(more_data)
    """
    def __init__(self, filename=None, mode='r'):
        self.open(filename=filename, mode=mode)

    def open(self, filename=None, mode='r'):

        self.close()

        self._mode = mode
        self._filename = filename

        if filename is None:
            return

        # expand shortcut variables
        fpath = os.path.expanduser(filename)
        fpath = os.path.expandvars(fpath)
        self._filename = fpath

        if mode == 'r+' and not os.path.exists(self._filename):
            # path doesn't exist but we want to append.  Change the
            # mode to write
            mode = 'w+'

        self._fobj = open(self._filename, _fix_mode(self.mode))

        if self._mode[0] == 'w':
            self._is_empty = True
        else:
            self._is_empty = False
            self._load_metadata()

    def _load_metadata(self):
        self._hdr = self._read_header()

        self._shape = self._hdr['shape']
        self._descr = self._hdr['dtype']
        self._dtype = np.dtype(self._descr)

        self._mmap = np.memmap(
            self._filename,
            dtype=self._dtype,
            mode=self._mode,
            offset=self._data_start,
            shape=self._shape,
        )

    def close(self):
        """
        Close the file and reset the metadata to None
        """

        if hasattr(self, '_mmap'):
            # official way to close the file according to docs
            del self._mmap

        if hasattr(self, '_fobj') and self._fobj is not None:
            self._fobj.close()

        self._filename = None
        self._mode = None
        self._fobj = None
        self._mmap = None
        self._hdr = None
        self._data_start = None
        self._shape = None
        self._descr = None
        self._dtype = None

    @property
    def mmap(self):
        """
        get a reference to the mmap
        """
        return self._mmap

    @property
    def size(self):
        """
        get the total number of elements
        """
        if self._mmap is None:
            raise RuntimeError("no file has been opened for reading")

        return self._mmap.size

    @property
    def shape(self):
        """
        get the shape of the data
        """
        if self._mmap is None:
            raise RuntimeError("no file has been opened for reading")

        return self._mmap.shape

    @property
    def dtype(self):
        """
        get the number of rows in the file
        """
        if self._mmap is None:
            raise RuntimeError("no file has been opened for reading")
        return self._mmap.dtype

    @property
    def header(self):
        """
        get a copy of the header
        """
        if self._hdr is None:
            raise RuntimeError("no file has been opened for reading")
        hdr = {}
        hdr.update(self._hdr)
        return hdr

    @property
    def mode(self):
        """
        get the file open mode
        """
        return self._mode

    @property
    def filename(self):
        """
        get the file name
        """
        return self._filename

    def write(self, data):
        """
        write data to the file, appending if the file is not empty

        paramters
        ---------
        data: array
            A structured numerical python array.  If data already
            exists in the file, this data must have compatible
            data type.  For binary files this includes the byte
            order.
        header: dict, optional
            Optional dictionary to write into the header.  This
            can only be written the first time.
        """

        self._ensure_open_for_writing()

        fobj = self._fobj

        if not self._is_empty:
            # check compatible, in case there is already data in the file
            self._ensure_compatible_dtype(data)
        else:
            self._write_initial_header(data)

        # zero bytes from the end of the file
        fobj.seek(0, 2)
        data.tofile(fobj)

        if not self._is_empty:
            self._update_shape(data.shape)

        self.open(
            filename=self._filename,
            mode='r+',
        )

    def __getitem__(self, arg):
        """

        # read subsets of columns and/or rows from the file.  This only works
        # for record types
        sf = SimpleFile(....)


        # read subsets of rows
        data = sf[35]
        data = sf[35:88]
        data = sf[35:88, 3, 5]
        data = sf[[3, 234, 5551]]
        """

        return self._mmap[arg]

    def _ensure_open(self):
        """
        check if a file is open, if not raise a RuntimeError
        """
        if not hasattr(self, '_fobj') or self._fobj is None:
            raise RuntimeError("no file is open")

    def _ensure_open_for_writing(self):
        """
        check if a file is open for writing, if not raise a RuntimeError
        """
        self._ensure_open()
        if self._fobj.mode[0] != 'w' and '+' not in self._mode:
            raise ValueError("You must open with 'w*' or 'r+' to write")

    def _ensure_open_for_reading(self):
        """
        check if a file is open for reading, if not raise a RuntimeError
        """
        self._ensure_open()

        if self._fobj.mode[0] != 'r' and '+' not in self._mode:
            raise ValueError("You must open with 'w+' or 'r*' to read")

    def _ensure_compatible_dtype(self, data):
        """
        if we are writing binary we demand exact match.

        For text we just make sure everything matches if the
        byte order is ignored
        """

        # if self._dtype is not None, there was data in the file
        if self._dtype is not None:
            if self._dtype != data.dtype:
                m = (
                    "attempt to write an incompatible "
                    "data type: expected '%s' got '%s'"
                )
                m = m % (self._descr, data.dtype.descr)
                raise ValueError(m)

    def _write_initial_header(self, data):
        """
        write the intial header for a new file
        """
        fobj = self._fobj
        fobj.seek(0)

        shape_string = _get_shape_string(data.shape)
        dtype_string = _get_dtype_string(data.dtype)
        version_string = _get_version_string()
        header_end = _get_header_end()

        fobj.write(shape_string)
        fobj.write(dtype_string)
        fobj.write(version_string)
        fobj.write(header_end)
        self._data_start = fobj.tell()

    def _update_shape(self, shape_add):
        """
        update the shape entry in the file
        """

        fobj = self._fobj
        fobj.seek(0)

        shape_current = self.shape
        if shape_current is None:
            raise RuntimeError('Attempting to update shape but '
                               'not found in header')

        ndim = len(shape_current)
        assert ndim == len(shape_add)
        if ndim > 1:
            # multi-dimensional array
            for i in range(1, ndim):
                assert shape_add[i] == shape_current[i]

        nrows_new = shape_current[0] + shape_add[0]
        shape_new = (nrows_new,)
        if ndim > 1:
            shape_new = shape_new + shape_current[1:]

        shape_string = _get_shape_string(shape_new)
        fobj.write(shape_string)
        self._hdr['shape'] = shape_new

    def _read_header(self):
        """
        Read the header from a simple self-describing file format with an
        ascii header.  See the write() function for information about reading
        this file format, and read() for reading.

        The file format:
          First line:
              SHAPE = --------------
        where the shape can be, e.g. (125,) for a 1-d array, or (25,8,15) for
        a 3-d array.  The -- pad it out to 40 characters

        This exact formatting is required so SHAPE can be updated *in place*
        when appending rows to a file.  Note the file can always be read as
        long as the first line reads 'SHAPE = some_number' but appending
        requires the exact format.

        Last two lines of the header region must be:
                END
                blank line
        case does not matter.

        In between the SHAPE and END lines is the header data.  This is a
        string that must eval() to a dictionary.  It must contain the
        following entry:

              DTYPE = 'f8'

        Where any valid scalar dtype is ok.  There should also be a VERSION
        tag.

              VERSION = '0.1'

        The rest of the keywords can by any variable can be used as long as it
        can be eval()d.

        An example header:
            SHAPE = (1032, 2)
            VERSION = 0.1
            DTYPE = '>f8'
            END

            -- data begins --
        """

        fobj = self._fobj

        if fobj is None:
            raise ValueError('there is no open file')

        fobj.seek(0)

        lines = []
        for line in fobj:
            line = str(line, 'ascii').strip()

            if line == 'END':
                break

            lines.append(line)

        assert line == 'END', 'expected END line'

        # we read one more blank line
        tmpline = str(fobj.readline(), 'ascii').strip()
        assert tmpline == '', 'expected blank line after END'

        self._data_start = fobj.tell()
        hdr = {
            'shape': _extract_shape_from_string(lines[0]),
            'dtype': _extract_dtype_from_string(lines[1]),
            'version': _extract_version_from_string(lines[2]),
        }

        return hdr

    def __repr__(self):

        top = ["filename: '%s'" % self._filename]

        s = ["mode: '%s'" % self._mode]

        s += ["shape: %s" % (self._shape,)]

        if self._descr is not None:
            drepr = pprint.pformat(self._descr).split('\n')
            drepr = ['  '+d for d in drepr]

            s += ["dtype:"]
            s += drepr

        slist = []
        for tmp in s:
            slist.append('    '+tmp)

        slist = top + slist
        rep = "\n".join(slist)
        return rep

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()


def write(outfile, data, append=False):
    """
    Write a numpy array into a simple self-describing file format with an ascii
    header.

    Parameters
    ----------
    outfile: string
        The filename to write
    data: array
        Numerical python array, a structured array with fields
    append: bool, optional
    """

    if append:
        mode = 'r+'
    else:
        mode = 'w'

    with SimpleFile(outfile, mode=mode) as sf:
        sf.write(data)


def _get_shape_string(shape):
    """
    Specially formatted fixed-length for updating later
    """
    s = 'SHAPE = %40s\n' % (shape,)
    return bytes(s, 'ascii')


def _get_dtype_string(dtype):
    """
    dtype string for header
    """
    if dtype.names is not None:
        dstr = str(dtype.descr)
        s = "DTYPE = %s\n" % dstr
    else:
        dstr = dtype.descr[0][1]
        s = "DTYPE = '%s'\n" % dstr
    return bytes(s, 'ascii')


def _get_version_string():
    """
    version string for header
    """
    s = "VERSION = '%s'\n" % SIMPLEFILE_VERSION
    return bytes(s, 'ascii')


def _get_header_end():
    """
    version string for header
    """
    s = "END\n\n"
    return bytes(s, 'ascii')


def _extract_shape_from_string(line):
    lsplit = line.split('=')
    if len(lsplit) != 2:
        raise ValueError("First line of header must be SHAPE = %40d")
    fname = lsplit[0].strip()

    # also allow old NROWS word for compatibility
    if fname.upper() != 'SHAPE':
        raise ValueError("First line of header must be SHAPE = %40d")

    shape = eval(lsplit[1])

    return shape


def _extract_dtype_from_string(line):
    lsplit = line.split('=')
    if len(lsplit) != 2:
        raise ValueError("Second line of header must be DTYPE = ...")
    fname = lsplit[0].strip()

    # also allow old NROWS word for compatibility
    if fname.upper() != 'DTYPE':
        raise ValueError("Second line of header must be DTYPE = ...")

    dtype = eval(lsplit[1])

    return dtype


def _extract_version_from_string(line):
    lsplit = line.split('=')
    if len(lsplit) != 2:
        raise ValueError("Third line of header must be VERSION = ...")
    fname = lsplit[0].strip()

    # also allow old NROWS word for compatibility
    if fname.upper() != 'VERSION':
        raise ValueError("Third line of header must be VERSION = ...")

    version = eval(lsplit[1])

    return version


_PYMODES = ['rb', 'r+b', 'wb', 'w+b']


def _fix_mode(mode):
    if mode in _PYMODES:
        return mode

    if mode == 'r':
        mode = 'rb'
    elif mode == 'r+':
        mode = 'r+b'
    elif mode == 'w':
        mode = 'wb'
    elif mode == 'w+':
        mode = 'w+b'
    else:
        raise ValueError("bad open mode: '%s'" % mode)

    return mode


def test():
    """
    very simple test
    """
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'test.sf')
        data = np.array([[1.0, 3.5],
                         [4.5, 2.6],
                         [8.7, 1.5]])

        with SimpleFile(fname, mode='w+') as sf:
            sf.write(data)
            sf.write(data)

            rdata = sf[:]
            assert np.all(data == rdata[0:3])
            assert np.all(data == rdata[3:])

        with SimpleFile(fname) as sf:
            rdata = sf[:]
            assert np.all(data == rdata[0:3])
            assert np.all(data == rdata[3:])
