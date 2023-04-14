from .filebase import FileBase
from . import util

class Dict(FileBase):
    def _do_init(
        self,
        filename=None,
        name=None,
        dir=None,
        verbose=False,
    ):

        self._type = 'dict'

        super()._do_init(
            filename=filename,
            name=name,
            dir=dir,
            verbose=verbose,
        )

    def write(self, data):
        """
        Write data to the dict column.

        Parameters
        ----------
        data: dict or json supported object
            The data must be supported by the JSON format.
        """

        util.write_json(data, self.filename)

    def read(self):
        """
        read data from this column
        """
        return util.read_json(self.filename)

    def _get_repr_list(self, full=False):
        """

        Get a list of metadat for this column.

        """
        indent = '  '

        if not full:
            s = ''
            if self.name is not None:
                s += 'Column: %-15s' % self.name
            s += ' type: %10s' % self.type

            s = [s]
        else:
            s = []
            if self.name is not None:
                s += ['name: %s' % self.name]

            if self.filename is not None:
                s += ['filename: %s' % self.filename]

            s += ['type: dict']

            s = [indent + tmp for tmp in s]
            s = ['Column: '] + s

        return s
