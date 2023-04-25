# flake8: noqa

ALLOWED_COL_TYPES = [
    'array', 'meta', 'index', 'index1', 'sorted', 'dict', 'cols', 'chunks',
]


DEFAULT_CACHE_MEM = '1g'

DEFAULT_COMPRESSOR = 'zstd'
DEFAULT_CLEVEL = 5
DEFAULT_SHUFFLE = 'bitshuffle'

DEFAULT_COMPRESSION = {
    'compressor': DEFAULT_COMPRESSOR,
    'clevel': DEFAULT_CLEVEL,
    'shuffle': DEFAULT_SHUFFLE,
}

# for the mergesort
DEFAULT_MERGESORT_CHUNKSIZE_GB = 0.1
