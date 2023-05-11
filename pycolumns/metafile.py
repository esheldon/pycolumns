import os
from . import util


class Meta(object):
    """
    Represent metadata that can be stored in JSON

    Parameters
    ----------
    filename: str
        The path to the file
    mode: str, optional
        'r' for read only
        'r+' for appending and modifying
    verbose: bool, optional
        If set to True, print messages
    """
    def __init__(self, filename, mode='r', verbose=False):
        self._type = 'meta'
        self._mode = mode
        self._verbose = verbose
        self._filename = filename
        self._name = util.extract_name(filename)
        self._dir = os.path.dirname(filename)

    @property
    def verbose(self):
        return self._verbose

    @property
    def name(self):
        """
        get the name of this object
        """
        return self._name

    @property
    def dir(self):
        """
        get the directory holding the file
        """
        return self._dir

    @property
    def mode(self):
        """
        get open mode
        """
        return self._mode

    @property
    def type(self):
        """
        get the type of this object
        """
        return self._type

    @property
    def filename(self):
        """
        get the file name holding the data
        """
        return self._filename

    def write(self, data):
        """
        Write data

        Parameters
        ----------
        data: Any json supported object
            The data must be supported by the JSON format.
        """
        self._check_mode_is_write('write to metadata')

        util.write_json(self.filename, data)

    def update(self, data):
        """
        Update the data.  This only works for dictionaries

        Parameters
        ----------
        data: dict
            Update the data.  The stored data and the input must be dict or
            dict like
        """
        self._check_mode_is_write('update metadata')

        odata = self.read()
        odata.update(data)
        util.write_json(self.filename, odata)

    def read(self):
        """
        read the data
        """
        return util.read_json(self.filename)

    def _check_mode_is_write(self, action):
        if self.mode != 'r+':
            raise IOError('cannot {action} in read only mode')

    def __repr__(self):
        """
        Get a list of metadat for this column.
        """
        indent = '  '

        s = []
        if self.name is not None:
            s += ['name: %s' % self.name]

        if self.filename is not None:
            s += ['filename: %s' % self.filename]

        s += ['type: meta']

        s = [indent + tmp for tmp in s]
        s = ['Meta: '] + s

        return '\n'.join(s)
