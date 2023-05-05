# flake8: noqa
import numpy as np

ALLOWED_COL_TYPES = [
    'array', 'meta', 'index', 'index1', 'sorted', 'dict', 'cols',
    'chunks',
]


DEFAULT_CACHE_MEM = '1g'

DEFAULT_CNAME = 'zstd'
DEFAULT_CLEVEL = 5
DEFAULT_SHUFFLE = 'bitshuffle'

DEFAULT_COMPRESSION = {
    'cname': DEFAULT_CNAME,
    'clevel': DEFAULT_CLEVEL,
    'shuffle': DEFAULT_SHUFFLE,
}

# 1 megabyte
DEFAULT_CHUNKSIZE = '1m'

CHUNKS_DTYPE = np.dtype(
    [
        ('offset', 'i8'),
        ('nbytes', 'i8'),
        ('rowstart', 'i8'),
        ('nrows', 'i8'),
        ('is_external', bool),
    ]
)
