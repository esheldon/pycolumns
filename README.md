A simple, efficient column-oriented, pythonic data store.

The focus is currently on efficiency of reading and writing.  Each table column
can optionally be compressed using blosc, and can be indexed for fast searches.
Basic consistency is ensured for the columns in the table, but the database is
not fully ACID.

The storage is a simple directory with files on disk.  The data for each column
in a table are stored in separate files for simplicity and efficiency.

Examples
--------

```python
>>> import pycolumns as pyc

# instantiate a column database from the specified coldir
>>> c = pyc.Columns('/some/path/mycols.cols')

# display some info about the columns
>>> c
Columns:
  dir: test.cols
  nrows: 64348146

  Table Columns:
    name             dtype    comp index
    -----------------------------------
    id                 <i8    zstd True
    name              <U10    zstd True
    x                  <f4    None False
    y                  <f4    None False

  Dictionaries:
    name
    ----------------------------
    meta

  Sub-Columns Directories:
    name
    ----------------------------
    telemetry

# Above we see the main types supported:  A table of columns, dictionaries, and
# sub-Columns directories, which are themselves full Columns.  the id and name
# columns have zstd compression and indexes for fast searching

# display info about column 'id'
>>> c['id']
Column:
  name: id
  filename: test.cols/id.array
  type: array
  index: True
  compression:
      cname: zstd
      clevel: 5
      shuffle: bitshuffle
  chunksize: 1m
  dtype: <i8
  nrows: 64348146

# number of rows in table
>>> c.nrows
64348146

# read all columns into a single rec array.  By default the dict
# columns are not loaded

>>> data = c.read()

# using asdict=True puts the data into a dict.  The dict data
# are also loaded in this case
>>> data = c.read(asdict=True)

# specify columns
>>> data = c.read(columns=['id', 'x'])

# dict columns can be specified if asdict is True.  Dicts can also
# be read as single columns, see below
>>> data = c.read(columns=['id', 'x', 'meta'], asdict=True)

# specifying a set of rows as sequence/array or slice
>>> data = c.read(columns=['id', 'x'], rows=[3, 225, 1235])
>>> data = c.read(columns=['id', 'x'], rows=slice(10, 20))

# read all data from column 'id' as an array rather than recarray
# alternative syntaxes
>>> ind = c['id'][:]
>>> ind = c['id'].read()

# read a subset of rows
# slicing
>>> ind = c['id'][25:125]

# specifying a set of rows
>>> rows = [3, 225, 1235]
>>> ind = c['id'][rows]
>>> ind = c['id'].read(rows=rows)

# query on indexed columns
>>> ind = c['id'] > 25
>>> ind = c['id'].between(25, 35)
>>> ind = c['id'] == 25

# read the corresponding data
>>> ccd = c['ccd'][ind]
>>> data = c.read(columns=['ra', 'dec'], rows=ind)

# composite searches over multiple columns
>>> ind = c['id'].between(15, 25) | (c['name'] == 'cxj2')
>>> ind = c['id'].between(15, 250) & (c['id'] != 66) & (c['name'] != 'af23')

# reading a dictionary column
>>> meta = c['meta'].read()

# enries can actually be another pycolumns directory
>>> cols['telemetry']
Columns:
  dir: test.cols/telemetry.cols
  nrows: 10

  Table Columns:
    name             dtype    comp index
    -----------------------------------
    obsid              <i8    zstd True
    voltage            <f4    None False

>>> v = cols['telemetry']['voltage'][:]

#
# Creating a columns data store and adding or updating data
#

# the easiest way is to create from an existing array or dict of arrays

dtype = [
    ('id', 'i8'),
    ('x', 'f4'),
    ('y', 'f4'),
    ('name', 'U10'),
]
num = 10
data = np.zeros(num, dtype=dtype)
data['id'] = np.arange(num)
data['x'] = rng.uniform(size=num)
data['y'] = rng.uniform(size=num)
data['name'] = data['id'].astype('U10')

cols = pyc.Columns.from_array(coldir, data)

# This version uses default compression for id and name
cols = pyc.Columns.from_array(coldir, data, compression=['id', 'name'])

# Append more data to the columns. The input data is a structured
# array or a dict of arrays.

>>> c.append(data1)
>>> c.append(data2)

# add indexes for id and name
cols['id'].create_index()
cols['name'].create_index()

# you can also create directly from a schema.  The schema itself
# can be created from an array, individual Column schemas, or from a dict
# here we set the chunksize for compressed columns to 10 megabytes

schema = pyc.TableSchema.from_array(array, compression=['id'], chunksize='10m')
cols = pyc.Columns.create(coldir, schema=schema)

# or you can build the schema from column schema
cx = pyc.ColumnSchema('x', dtype='f4')

# compression can be set to True to get the defaults, or a dict
# specifying blosc options
cid = pyc.ColumnSchema('id', dtype='i8', compression=True)
cname = pyc.ColumnSchema(
    'name',
    dtype='U5',
    compression={'cname': 'zstd', 'zlevel': 5, 'shuffle': 'bitshuffle'}
)

schema = pyc.TableSchema([cid, cx, cname])

# You can also use a dict
sch = {
    'id': {
        'dtype': 'i8',
        'compress': True,
    },
    'x': {'dtype': 'f4'},
}

schema = pyc.TableSchema.from_schema(sch)

# add an uncompressed column, filling with zeros
# currently only uncompressed columns can be added after
# the columns have data

>>> cschema = pyc.ColumnSchema('newcol', dtype='f4')
>>> c.create_column(cschema)
>>> assert np.all(c['newcol'][:] == 0)

# add a dictionary column.
>>> c.create_dict('weather')
>>> c['weather'].write({'temp': 30.1, 'humid': 0.5})


# overwrite dict column
>>> c['weather'].write({'temp': 33.2, 'humid': 0.3, 'windspeed': 60.5})

# Update column data in place.  Currently only uncompressed columns can be
# updated
c['x'][10:20] = 3
c['name'][[5, 6]] = ['alpha', 'beta']
c['y'][50] = 1.25

# update dictionary.  Only the fields in the input dictionary are updated or
# added
c['meta'].update({'extra': 5})

# get all names, including dictionary and sub Columns
# same as list(c.keys())
>>> c.names
['telemetry', 'id', 'y', 'x', 'name', 'meta']

# only array column names
>>> c.column_names
['id', 'y', 'x', 'name']

# only dict columns
>>> c.dict_names
['meta', 'weather']

# only sub Columns directories
>>> c.subcols_names
['telemetry']

# reload all columns or specified column/column list
>>> c.reload()

# delete a column or other entry.  This will prompt the user
# to confirm the deletion
>>> c.delete_entry('id')
# no prompt
>>> c.delete_entry('id', yes=True)

# delete all data in the columns store
# with prompt
>>> c.delete()
# no prompt
>>> c.delete(yes=True)

# to configure the amount of memory used during index creation, specify
# cache_mem is 0.5 gigabytes
>>> cols = pyc.Columns(fname, cache_mem='0.5g')
```

Installation
------------
pip install pycolumns

Dependencies
------------

pycolumns depends on numpy and [blosc](https://github.com/Blosc/python-blosc)
for data compression. Both dependencies will be automatically installed when
installing pycolumns with pip
