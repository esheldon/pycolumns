"""
this is not the index on a column, but set of indices
"""
import numpy as np


class Index(np.ndarray):
    """
    Represent an index into a database.  This object inherits from normal
    numpy arrays, but behaves differently under the "&" and "|" operators.
    These return the intersection or union of values in two Index objects.

    Methods:
        The "&" and "|" operators are defined.

        array(): Return an ordinary np.ndarray view of the Index.

    Examples:
        >>> i1=Index([3,4,5])
        >>> i2=Index([4,5,6])
        >>> (i1 & i2)
        Index([4, 5])
        >>> (i1 | i2)
        Index([3, 4, 5, 6])

    """
    def __new__(self, init_data, copy=False):
        arr = np.array(init_data, copy=copy)
        shape = arr.shape

        ret = np.ndarray.__new__(self, shape, arr.dtype,
                                 buffer=arr)
        return ret

    def array(self):
        return self.view(np.ndarray)

    def __and__(self, ind):
        # take the intersection
        if isinstance(ind, Index):
            w = np.intersect1d(self, ind)
        else:
            raise ValueError("comparison index must be an Index object")
        return Index(w)

    def __or__(self, ind):
        # take the unique union
        if isinstance(ind, Index):
            w = np.union1d(self, ind)
        else:
            raise ValueError("comparison index must be an Index object")

        return Index(w)

    def __repr__(self):
        rep = np.ndarray.__repr__(self)
        rep = rep.replace('array', 'Index')
        return rep
