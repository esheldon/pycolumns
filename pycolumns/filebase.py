from . import util


class FileBase(object):
    """
    Represent a column in a Columns database.  Facilitate opening, reading,
    writing of data.  This class can be instantiated alone, but is usually
    accessed through the Columns class.
    """

    @property
    def verbose(self):
        return self._verbose

    @property
    def name(self):
        """
        get the name type of the column
        """
        return self._name

    @property
    def dir(self):
        """
        get the directory holding the file
        """
        return self._dir

    @property
    def type(self):
        """
        get the data type of the column
        """
        return self._type

    def __repr__(self):
        """
        Print out some info about this column
        """
        s = self._get_repr_list(full=True)
        s = "\n".join(s)
        return s
