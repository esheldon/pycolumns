"""
Read and write numpy arrays to a simple file format.  The format is a
simple ascii header followed by data in binary format.
"""
# vim: set filetype=python :
import sys
import os
import pprint
import copy

import np
from np import array

SIMPLEFILE_VERSION='0.1'

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
        data =sf.read(rows=row_list)


    with SimpleFile('test.rec','w+') as sf:
        sf.write(data)

        # this appends data
        sf.write(more_data)

        # check the first row of the data we have written
        sf[0]

        sf.write(more_data)

    """
    def __init__(self, filename=None, mode='r', dtype=None, shape=None):
        self.close()
        self.open(filename=filename, mode=mode, dtype=dtype, shape=shape):

    def open(self, filename=None, mode='r', dtype=None, shape=None):
        self._mode = mode
        self._filename = filename

        if filename is None:
            return

        # expand shortcut variables
        fpath = os.path.expanduser(filename)
        fpath = os.path.expandvars(fpath)
        self._filename=fpath

        if mode == 'r+' and not os.path.exists(self._filename):
            # path doesn't exist but we want to append.  Change the
            # mode to write
            mode = 'w+'

        self._fobj = open(self._filename, self.mode)


        if self._mode[0] == 'w':
            # we are starting from scratch, so we can't read a header, instead
            # we need the user to input the dtype and shape

            if dtype is None:
                raise RuntimeError('send dtype= when creating a file')
            if shape is None:
                raise RuntimeError('send shape= when creating a file')

            self._write_header(
                shape,
                dtype,
            )

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
        Close any open file object.  Make sure fobj, _hdr, are None
        """

        if hasattr(self,'_robj'):
            if self._robj is not None:
                self._robj.close()

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
    def shape(self):
        """
        get the number of rows in the file
        """
        if self._hdr is None:
            raise RuntimeError("no file has been opened for reading")

        return self._shape

    @property
    def dtype(self):
        """
        get the number of rows in the file
        """
        return self._dtype

    @property
    def header(self):
        """
        get a copy of the header
        """
        hdr = {}
        hdr.update(self._hdr)
        return hdr

    @property
    def _mode(self):
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

    def _ensure_open(self):
        """
        check if a file is open, if not raise a RuntimeError
        """
        if self._robj is None:
            raise RuntimeError("no file is open")

    def _ensure_open_for_writing(self):
        """
        check if a file is open for writing, if not raise a RuntimeError
        """
        self._ensure_open()
        if self._robj.mode[0] != 'w' and '+' not in self._mode:
            raise ValueError("You must open with 'w*' or 'r+' to write")

    def _ensure_open_for_reading(self):
        """
        check if a file is open for reading, if not raise a RuntimeError
        """
        self._ensure_open()

        if self._robj.mode[0] != 'r' and '+' not in self._mode:
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
                raise ValueError("attempt to write an incompatible "
                                 "data type: "+mess)

    def write(self, data, header=None):
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

        # check compatible, in case there is already data in the file
        self._ensure_compatible_dtype(data)

        # this will make self._dtype if it is the first write
        self._write_header(data, header=header)

        self._robj.write(data)

    def read(self,
             rows=None,
             fields=None,
             columns=None,
             header=False, 
             view=None, # ignored
             split=False,
             reduce=False):
        """
        Read data from the file.

        parameters
        -----------
        rows: sequence or scalar, optional
            A scalar, array, list or tuple of row numbers to read.  
            Default is None, meaning read all rows.

        columns: sequence or scalar
            A scalar, list or tuple of strings naming columns to read 
            from the file.  Default is None or read all columns.
        fields:  Same as sending columns=.

        header: bool, optional
            If True, return both the array and the header dict in
            a tuple.
        split: bool, optional
            If True, return a list of arrays for each column rather
            than a structured array.
        reduce: bool, optional
            If True, and there is only one field requested, reduce
            it to a plain array. This is equivalent to sending
            columns=(scalar column name)

        returns
        -------

        A structured array with fields.

        If the columns= is a scalar column name (rather than list of names or
        None), then the data is a plain array holding the column data
        """
        
        self._ensure_open_for_reading()

        result = self._do_read(rows=rows, 
                               fields=fields,
                               columns=columns)

        if split:
            result = split_fields(result)
        elif reduce:
            result = reduce_array(result)

        if header:
            return result, copy.deepcopy(self._hdr)
        else:
            return result

    def __getitem__(self, arg):
        """

        # read subsets of columns and/or rows from the file.  This only works
        # for record types
        sf = SFile(....)


        # read subsets of rows
        data = sf[ 35 ]
        data = sf[ 35:88 ]
        data = sf[ [3,234,5551,.. ] ]

        # read subsets of columns
        data = sf['fieldname'][:]
        data = sf[ ['field1','field2',...] ][:]

        # read subset of rows *and* columns.
        data = sf['fieldname'][3:58]
        data = sf[fieldlist][rowlist]
        """

        return self._robj[arg]

    def _do_read(self, rows=None, fields=None, columns=None):
        """
        use the recfile object to read the data
        """

        if columns is None:
            columns = fields

        return self._robj.read(rows=rows, columns=columns)

    def _make_header(self, data, header=None):
        if header is None:
            head={}
        else:
            head=copy.deepcopy(header)

        for key in ['_shape','_nrows','_shape','_has_fields']:
            if key in head: del head[key]
            if key.upper() in head: del head[key.upper()]

        descr = data.dtype.descr

        head['_DTYPE'] = descr
        head['_VERSION'] = SFILE_VERSION

        return head

    def _write_header(self, data, header=None):

        if self._hdr is not None:
            # we are appending data.
            # Just update the nrows and move to the end

            self._update_shape(data.size)
        else:

            # this is a dict of variable size
            self._hdr = self._make_header(data, header=header)

            # store some of the info
            self._descr = self._hdr['_DTYPE']
            self._dtype = np.dtype(self._descr)

            shape_string = self._get_shape_string(data.size)
            self._size = data.size

            # As long as the dict contains types that can be represented as
            # constants, this pretty printing can be eval()d.

            hdr_dict_string = pprint.pformat(self._hdr)

            lines = [
                shape_string,
                hdr_dict_string,
                'END',
                '',  # to add a new line
                '',  # to add a blank line
            ]

            total_str = '\n'.join( lines )

            self._robj.robj.write_header_and_update_offset(total_str)

    def _update_shape(self, shape_add):
        """
        update the size in the file

        TODO: update for new recfile where file object
        is maintained within the C++ code
        """
        shape_current = self.shape
        if shape_current is None:
            raise RuntimeError('Attempting to update shape but not found in header')

        ndim = len(shape_current)
        assert ndim == len(shape_add)
        if ndim > 1:
            # multi-dimensional array
            for i in range(1,ndim):
                assert shape_add[i] == shape_current[i]

        nrows_new  shape_current[0] + shape_add[0]
        shape_new = (nrows_new,)
        if ndim > 1:
            shape_new = shape_new + shape_current[1:]

        self._robj.robj.update_row_count(size_new)
        self._size = size_new
        self._hdr['_SHAPE'] = shape_new

    def _get_shape_string(self, shape):
        # Specially formatted fixed-length for updating later
        s = 'SHAPE = %20d' % (shape,)
        return s

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

        Where any valid scalar dtype is ok.  There should also be a VERSION tag.

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
            line = line.strip()

            if line == 'END':
                break

            lines.append(line)

        assert line == 'END', 'expected END line'

        # we read one more blank line
        tmpline = fobj.readline().strip()
        assert tmpline == '', 'expected blank line after END'

        self._data_start = fobj.tell()
        hdr['shape'] = _extract_shape_from_string(lines[0])
        hdr['dtype'] = _extract_dtype_from_string(lines[0])
        hdr['version'] = _extract_version_from_string(lines[0])

        return hdr

    def __repr__(self):

        top = ["filename: '%s'" % self._filename]

        s = ["mode: '%s'" % self._mode]

        s += ["shape: %s" % (self._shape,)]

        if self._descr is not None:
            drepr=pprint.pformat(self._descr).split('\n')
            drepr = ['  '+d for d in drepr]
            #drepr = '  '+drepr.replace('\n','\n  ')
            s += ["dtype:"]
            s += drepr

        slist=[]
        for tmp in s:
            slist.append('    '+tmp )

        slist = top + slist
        rep = "\n".join(slist)
        return rep

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()


def write(outfile, data, **keys):
    """
    Name:
        sfile.write()

    Calling Sequence:
        sfile.write(data, outfile, header=None, append=False)

    Write a numpy array into a simple self-describing file format with an ascii
    header.  See the read() function for information about reading this file
    format.  See the docs for the SFile class for an idea of the full
    functionality wrapped by this covenience function.


    Inputs:
        outfile: string
            The filename to write
        data: array
            Numerical python array, a structured array with fields

    Optional Inputs:
        header=: A dictionary containing keyword-value pairs to be added to
            the header.

        append=False: Append to the file. Default is False. If set to True,
            then what happens is situation dependent:
                1) if the input is a file object then it is assumed there is
                    an existing header.  The header is updated to reflect the
                    new appended rows after writing.
                2) if the input is a string and the file exists, then the file
                    is opened with mode "r+", it is assumed the header
                    exists, and the header is updated to reflext the new
                    appended rows after writing.
                3) if the input is a string and the file does *not* exist,
                    then the file is opened with mode "w" and the request
                    to append is ignored.

    Examples:
        import sfile
        hdr={'date': '2007-05-12','age': 33}
        sfile.write(data1, 'test.rec', header=hdr)

        sfile.write(data2, 'test.rec', append=True)

        If this is part of the esutil package, use
            import esutil
            esutil.sfile.write(...)

    File Format:
        The file format is an ascii header followed by data in binary or rows
        of ascii.  The data columns must be fixed length in order to map onto
        numpy arrays.  See the documentation for _read_header() for details
        about the header format.

    Modification History:
        Created: 2007-05-25, Erin Sheldon, NYU
        Ignore append=True when file does not yet exist.
        Allow continuation characters "\\" to continue header keywords
        onto the next line.  2009-05-05

        Moved to object-oriented approach using the SFile class.
            2009-11-16, ESS, BNL

    """

    if isinstance(outfile,np.ndarray):
        outfile,data = data,outfile

    header = keys.get('header',None)
    append = keys.get('append',False)

    if append:
        # if file doesn't yet exist, this will be changed to 'w+' internally.
        mode = 'r+'
    else:
        mode = 'w'

    with SimpleFile(outfile, mode=mode) as sf:
        sf.write(data, header=header)


def read(filename, **keys):
    """
    sfile.read()

    Read a numpy array from a simple self-describing file format with an ascii
    header.  See the write() function for information about this file format.

    parameters
    -----------
    filename: string
        Filename from which to read

    rows: sequence or scalar, optional
        A scalar, array, list or tuple of row numbers to read.  
        Default is None, meaning read all rows.

    columns: sequence or scalar
        A scalar, list or tuple of strings naming columns to read 
        from the file.  Default is None or read all columns.
    fields:  Same as sending columns=.

    header: bool
        If True, return both the array and the header dict in
        a tuple.

    split: bool, optional
        If True, return a list of arrays for each column rather
        than a structured array.
    reduce: bool, optional
        If True, and there is only one field requested, reduce
        it to a plain array. This is equivalent to sending
        columns=(scalar column name)

    returns
    -------

    A structured array with fields.

    If the columns= is a scalar column name (rather than list of names or
    None), then the data is a plain array holding the column data
    """

    with SFile(filename) as sf:
        data = sf.read(**keys)

    return data

def split_fields(data, fields=None, getnames=False):
    """
    Name:
        split_fields

    Calling Sequence:
        The standard calling sequence is:
            field_tuple = split_fields(data, fields=)
            f1,f2,f3,.. = split_fields(data, fields=)

        You can also return a list of the extracted names
            field_tuple, names = split_fields(data, fields=, getnames=True)

    Purpose:
        Get a tuple of references to the individual fields in a structured
        array (aka recarray).  If fields= is sent, just return those
        fields.  If getnames=True, return a tuple of the names extracted
        also.

        If you want to extract a set of fields into a new structured array
        by copying the data, see esutil.numpy_util.extract_fields

    Inputs:
        data: An array with fields.  Can be a normal numpy array with fields
            or the recarray or another subclass.
    Optional Inputs:
        fields: A list of fields to extract. Default is to extract all.
        getnames:  If True, return a tuple of (field_tuple, names)

    """

    outlist = []
    allfields = data.dtype.fields

    if allfields is None:
        if fields is not None:
            raise ValueError("Could not extract fields: data has "
                             "no fields")
        return (data,)

    if fields is None:
        fields = allfields
    else:
        if isinstance(fields, (str,unicode)):
            fields=[fields]

    for field in fields:
        if field not in allfields:
            raise ValueError("Field not found: '%s'" % field)
        outlist.append( data[field] )

    output = tuple(outlist)
    if getnames:
        return output, fields
    else:
        return output

def reduce_array(data):
    # if this is a structured array with fields, and only has a single
    # field, return a simple array view of that field, e.g. data[fieldname]
    if hasattr(data, 'dtype'):
        if data.dtype.names is not None:
            if len(data.dtype.names) == 1:
                # get a simpler view
                return data[data.dtype.names[0]]
    else:
        return data

def _match_key(d, key, require=False):
    """
    Match the key in a case-insensitive way and return the value. Return None
    if not found or raise an error if require=True
    """
    if not isinstance(d,dict):
        raise RuntimeError('Input object must be a dict, got %s' % d)

    keys = list( d.keys() )

    keyslow = [k.lower() for k in keys]
    keylow = key.lower()

    if keylow in keyslow:
        ind = keyslow.index(keylow)
        return d[keys[ind]]
    else:
        if not require:
            return None
        else:
            raise RuntimeError("Could not find required key: '%s'" % key)
 

_major_pyvers = int( sys.version_info[0] )
def isstring(obj):
    if _major_pyvers >= 3:
        string_types=(str, np.string_)
    else:
        string_types=(str, unicode, np.string_)

    if isinstance(obj, string_types):
        return True
    else:
        return False

def _fix_range(i, maxval):
    if i < 0:
        i=maxval-i
    if i > maxval:
        i=maxval
    return i

def _extract_shape_from_string(line):
    lsplit = line.split('=')
    if len(lsplit) != 2:
        raise ValueError("First line of header must be SHAPE = %40d")
    fname=lsplit[0].strip()

    # also allow old NROWS word for compatibility
    if fname.upper() != 'SHAPE':
        raise ValueError("First line of header must be SHAPE = %40d")

    shape = eval(lsplit[1])

    return shape

def _extract_dtype_from_string(line):
    lsplit = line.split('=')
    if len(lsplit) != 2:
        raise ValueError("Second line of header must be DTYPE = ...")
    fname=lsplit[0].strip()

    # also allow old NROWS word for compatibility
    if fname.upper() != 'DTYPE':
        raise ValueError("Second line of header must be DTYPE = ...")

    dtype = eval(lsplit[1])

    return dtype

def _extract_version_from_string(line):
    lsplit = line.split('=')
    if len(lsplit) != 2:
        raise ValueError("Third line of header must be VERSION = ...")
    fname=lsplit[0].strip()

    # also allow old NROWS word for compatibility
    if fname.upper() != 'VERSION':
        raise ValueError("Third line of header must be VERSION = ...")

    version = eval(lsplit[1])

    return version

# deprecated
def Open(filename, mode='r', **keys):
    sf = SFile(filename, mode=mode, **keys)
    return sf

def test():
    """
    very simple test
    """
    import tempfile
    tmpfile = tempfile.mktemp(suffix='.rec')
    data=np.array([(1.0, 3),(4.5,2)], dtype=[('fcol','f4'),('icol','i4')])

    write(data, tmpfile)
    read(tmpfile)
