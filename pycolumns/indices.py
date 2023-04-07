"""
this is not the index on a column, but set of indices
"""
import numpy as np


class Indices(np.ndarray):
    """
    Represent indices returned by querying a column index.  This object
    inherits from normal numpy arrays, but behaves differently under the "&"
    and "|" operators.  These return the intersection or union of values in two
    Indices objects.

    Methods:
        The "&" and "|" operators are defined.

        array(): Return an ordinary np.ndarray view of the Indices.

    Examples:
        >>> i1=Indices([3,4,5])
        >>> i2=Indices([4,5,6])
        >>> (i1 & i2)
        Indices([4, 5])
        >>> (i1 | i2)
        Indices([3, 4, 5, 6])

    """
    def __new__(self, init_data, copy=False):
        self._is_sorted = False
        arr = np.array(init_data, copy=copy)
        shape = arr.shape

        ret = np.ndarray.__new__(self, shape, arr.dtype,
                                 buffer=arr)
        return ret

    @property
    def is_sorted(self):
        """
        returns True if sort has been run
        """
        return self._is_sorted

    def sort(self):
        """
        sort and set the is_sorted flag
        """
        if not self.is_sorted:
            if self.ndim > 0:
                super(Indices, self).sort()
            self._is_sorted = True

    def array(self):
        return self.view(np.ndarray)

    def __and__(self, ind):
        # take the intersection
        if isinstance(ind, Indices):
            w = np.intersect1d(self, ind)
        else:
            raise ValueError("comparison index must be an Indices object")
        return Indices(w)

    def __or__(self, ind):
        # take the unique union
        if isinstance(ind, Indices):
            w = np.union1d(self, ind)
        else:
            raise ValueError("comparison index must be an Indices object")

        return Indices(w)

    def __repr__(self):
        rep = np.ndarray.__repr__(self)
        rep = rep.replace('array', 'Indices')
        return rep
