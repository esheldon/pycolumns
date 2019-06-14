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

    def set_sorted(self):
        """
        set sorted indices
        """
        if not self.has_sort:
            self._sort_indices = self.argsort()
            self._sorted = self[self._sort_indices]

    @property
    def has_sort(self):
        """
        check if the unsort indices are present
        """
        if hasattr(self, '_sorted'):
            return True
        else:
            return False

    @property
    def has_unsort(self):
        """
        check if the unsort indices are present
        """
        if hasattr(self, '_unsort_indices'):
            return True
        else:
            return False

    @property
    def sorted(self):
        """
        indices that sort the array.  Only generated when running the sort()
        method.  Equivalent to argsort() before sorting
        """
        if not self.has_sort:
            self.set_sorted()

        return self._sorted

    @property
    def unsort_indices(self):
        """
        indices that unsort.  Only available after running the sort() method
        """
        if not self.has_unsort:
            if not self.has_sort:
                self.set_sorted()

            si = self._sort_indices
            tind = np.arange(self.size)
            self._unsort_indices = tind[si].argsort()

            # we don't need these any more
            del self._sort_indices

        return self._unsort_indices

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
