import pytest


def test_column_schema():
    import numpy as np
    from ..schema import ColumnSchema
    from ..defaults import DEFAULT_COMPRESSION, DEFAULT_CHUNKSIZE

    # simple
    s = {'dtype': np.dtype('f8').str}
    cs = ColumnSchema('blah', dtype='f8')
    assert cs.name == 'blah'
    assert cs == s

    # with default compression
    s = {
        'dtype': np.dtype('f8').str,
        'compression': DEFAULT_COMPRESSION.copy(),
        'chunksize': DEFAULT_CHUNKSIZE,
    }
    cs = ColumnSchema('blah', dtype='f8', compression=True)
    assert cs.name == 'blah'
    assert cs == s

    # with more specific settings
    s = {
        'dtype': np.dtype('f8').str,
        'compression': {'cname': 'zlib'},
        'chunksize': '1g',
    }
    cs = ColumnSchema(
        'blah',
        dtype='f8',
        compression=s['compression'],
        chunksize=s['chunksize'],
    )
    se = s.copy()
    se['compression'] = DEFAULT_COMPRESSION.copy()
    se['compression'].update(s['compression'])
    assert cs.name == 'blah'
    assert cs == se


def test_table_schema_from_column_schema():
    from ..schema import ColumnSchema, TableSchema
    from ..defaults import DEFAULT_COMPRESSION, DEFAULT_CHUNKSIZE

    cslist = [
        ColumnSchema(name='id', dtype='i8', compression=True),
        ColumnSchema(name='ra', dtype='f8'),
        ColumnSchema(name='name', dtype='U5',
                     compression={'shuffle': 'shuffle'}),
    ]
    ts = TableSchema(cslist)

    for cs in cslist:
        assert cs.name in ts
        assert ts[cs.name] == cs

    assert ts['id']['compression'] == DEFAULT_COMPRESSION
    assert ts['id']['chunksize'] == DEFAULT_CHUNKSIZE
    assert ts['name']['compression']['shuffle'] == 'shuffle'


def test_table_schema_from_array():
    import numpy as np
    from ..schema import ColumnSchema, TableSchema
    from ..defaults import DEFAULT_CHUNKSIZE

    arr = np.zeros(1, dtype=[('id', 'i8'), ('ra', 'f8'), ('name', 'U5')])

    # simplest
    ts = TableSchema.from_array(arr)

    cslist = []
    for name in arr.dtype.names:
        cs = ColumnSchema(name=name, dtype=arr[name].dtype)
        cslist.append(cs)

    for cs in cslist:
        assert cs.name in ts
        assert ts[cs.name] == cs

    # some defaults
    compression = ['id', 'name']
    ts = TableSchema.from_array(arr, compression=compression)

    cslist = []
    for name in arr.dtype.names:
        cs = ColumnSchema(
            name=name, dtype=arr[name].dtype,
            compression=name in compression,
        )
        cslist.append(cs)

    for cs in cslist:
        assert cs.name in ts
        assert ts[cs.name] == cs

    # get specific
    compression = {
        'id': {
            'cname': 'zlib',
            'clevel': 9,
            'shuffle': 'shuffle',
        },
    }
    chunksize = '1g'
    ts = TableSchema.from_array(arr, compression=compression, chunksize='1g')

    cslist = []
    for name in arr.dtype.names:
        cs = ColumnSchema(
            name=name, dtype=arr[name].dtype,
            compression=compression.get(name),
            chunksize=chunksize,
        )
        cslist.append(cs)

    for cs in cslist:
        assert cs.name in ts
        assert ts[cs.name] == cs

    # get specific with chunksize too
    compression = {
        'id': {
            'cname': 'zlib',
            'clevel': 9,
            'shuffle': 'shuffle',
        },
    }
    chunksize = {'id': '1g'}

    ts = TableSchema.from_array(
        arr, compression=compression, chunksize=chunksize,
    )

    cslist = []
    for name in arr.dtype.names:
        cs = ColumnSchema(
            name=name, dtype=arr[name].dtype,
            compression=compression.get(name),
            chunksize=chunksize.get(name, DEFAULT_CHUNKSIZE),
        )
        cslist.append(cs)

    for cs in cslist:
        assert cs.name in ts
        assert ts[cs.name] == cs


def test_column_compression():
    from ..schema import _get_column_compression

    assert _get_column_compression(False, 'blah') is None
    assert _get_column_compression(True, 'blah') is True

    assert _get_column_compression(['blah'], 'blah') is True
    assert _get_column_compression(['hey'], 'blah') is None

    assert _get_column_compression({'blah': True}, 'blah') is True
    assert _get_column_compression({'blah': False}, 'blah') is None
    assert _get_column_compression({'id': True}, 'blah') is None

    comp = {'cname': 'zlib'}
    assert _get_column_compression({'blah': comp}, 'blah') == comp


def test_column_chunksize():
    from ..schema import _get_column_chunksize
    from ..defaults import DEFAULT_CHUNKSIZE

    assert _get_column_chunksize('1m', 'blah') == '1m'
    assert _get_column_chunksize(3, 'blah') == 3
    assert _get_column_chunksize({'blah': '1g'}, 'blah') == '1g'
    assert _get_column_chunksize({'id': '1m'}, 'blah') == DEFAULT_CHUNKSIZE
    with pytest.raises(ValueError):
        _get_column_chunksize({'id', '1m'}, 'blah')
        _get_column_chunksize(['blah', 'id'], 'blah')
        _get_column_chunksize(('blah', 'id'), 'blah')
