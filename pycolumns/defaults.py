# flake8: noqa

ALLOWED_COL_TYPES = [
    'array', 'meta', 'index', 'index1', 'sorted', 'dict', 'cols', 'chunks',
]


DEFAULT_CACHE_MEM = '1g'

DEFAULT_COMPRESSION = 'zstd'
DEFAULT_CLEVEL = 5

# for the mergesort
DEFAULT_MERGESORT_CHUNKSIZE_GB = 0.1
