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
>>> cols = pyc.Columns('/some/path/mycols.cols')

# display some info about the columns
>>> cols
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

  Metadata:
    name
    ----------------------------
    meta
    versions

  Sub Tables:
    name
    ----------------------------
    telemetry/

# Above we see the main types supported:  A table of columns, metadata entries, and
# sub tables, which are themselves full Columns.  The id and name
# columns have zstd compression and indexes for fast searching
#
# Note there is a main table in the root directory as well as a named
# table telemetry/ in a subdirectory.  We can print info for it as well
>>> cols['telemetry/']
Columns:
  dir: test.cols/telemetry.cols
  nrows: 25

  Table Columns:
    name             dtype    comp index
    -----------------------------------
    temp               <f8    None False
    voltate            <f4    None False

# Only objects in the root are shown above, but we can
# print the full directory structure which shows that
# telemetry also has a sub table
>>> cols.list()
root has 4 columns 2 metadata
telemetry/
  periods/

>>> cols.list(full=True)
- id
- name
- x
- y
- {meta}
- {versions}
telemetry/
  - humid
  - temp
  periods/
    - start
    - end

# display info about column 'id'
>>> cols['id']
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
>>> cols.nrows
64348146

#
# reading data
#

# read all columns into a single structured array.
>>> data = cols[:]

# read a subset of columns into a structured array
>>> columns = ['id', 'x']
>>> data = cols[columns][:]

# specifying a set of rows as sequence or slice
>>> rows = [3, 225, 1235]
>>> data = cols[rows]
>>> data = cols[20:30]

# column and row subsets
>>> data = cols[columns][rows]
>>> data = cols[columns][10:20]

# read data from the telemetry/ table
>>> data = cols['telemetry/'][:]
>>> v = cols['telemetry/']['voltage'][:]

# You can get the table object directly.  It is also a Columns object
>>> telcols = cols['telemetry/']
>>> v = telcols['voltage'][:]

# You can also access with a full path
>>> v = cols['telemetry/voltage'][:]

#
# working with single columns
#

# read data from a column as an array rather than as part of a
# structured array
>>> x = cols['x'][:]
>>> x = cols['x'][25:125]
>>> x = cols['x'][rows]

#
# queries on indexed columns
#

>>> ind = cols['id'] > 25
>>> ind = cols['id'].between(25, 35)
>>> ind = cols['id'] == 25

# read the corresponding data
>>> ccd = cols['ccd'][ind]
>>> data = cols[columns][ind]
>>> data = cols.read(columns=['ra', 'dec'], rows=ind)

# composite searches over multiple columns
>>> ind = cols['id'].between(15, 25) | (cols['name'] == 'cxj2')
>>> ind = (cols['id'] != 66) & (cols['name'] != 'af23')

#
# reading data from a metadata entry.
#

>>> cols.meta['versions'].read()
>>> {'data_vers': 'v5b', 's_vers': '1.1.2'}

#
# As an alternative to slicing you can use the the .read method
# both for Columns and individual columns
#

>>> data = cols.read()
>>> data = cols.read(columns=columns)
>>> data = cols.read(columns=columns, rows=rows)
>>> data = cols.read(columns=columns, rows=slice(10, 20))
>>> x = cols['x'].read()
>>> x = cols['x'].read(rows=rows)

# using .read you can get the data as a dict of arrays
>>> data = cols.read(asdict=True)

#
# Creating a columns data store and adding or updating data
#

# create an empty Columns
cols = pyc.Columns.create(coldir)

# Create a new Columns store with a table in the root
# The schema is determined from input data

dtype = [('id', 'i8'), ('name', 'U10')]
num = 10
data = np.zeros(num, dtype=dtype)
data['id'] = np.arange(num)
data['name'] = data['id'].astype('U10')

cols = pyc.Columns.create_from_array(coldir, data)

# add more tables not in the root.  from_array() creates the schema from the
# input array and by default appends the data.  Note table names must end in
# the / character

# Use default compression
cols.from_array(data2, name='sub1/', compression=['id', 'name'])
cols.from_array(data3, name='sub1/sub2/')

# two ways to access sub tables, via intermediate Columns objects or by full
# path
cols['sub1/sub2/']
cols['sub1/sub2/id']
cols['sub1']['sub2/']['id']

# Append more data to the columns. The input data is a structured
# array or a dict of arrays.

>>> cols.append(moredata1)
>>> cols['sub1/sub2/'].append(moredata2)

# add indexes for id and name
cols['id'].create_index()
cols['name'].create_index()

# you can also create tables from a schema.  The schema itself
# can be created from an array, individual Column schemas, or from a dict

# here we use compression for id, and set the compressed chunk
# size to 10 megabytes

schema = pyc.TableSchema.from_array(array, compression=['id'], chunksize='10m')
cols.create_table(schema=schema, name='fromschema/')

# or you can build the schema from individual column schema
cid = pyc.ColumnSchema('id', dtype='i8', compression=True)
cname = pyc.ColumnSchema('name', dtype='i8', compression=True)
schema = pyc.TableSchema([cid, cname])

# Note you specify blosc compression info in detail
cname = pyc.ColumnSchema(
    'name',
    dtype='U5',
    compression={'cname': 'zstd', 'clevel': 5, 'shuffle': 'bitshuffle'}
    chunksize='10m',  # 10 megabytes
)

# You can also use a dict
sch = {
    'id': {
        'dtype': 'i8',
        'compress': True,
    },
    'x': {'dtype': 'f4'},
}

schema = pyc.TableSchema.from_schema(sch)

#
# add a new column.
#

# Default fill value is zeros, but you can set it with fill_value
>>> cschema = pyc.ColumnSchema('newfcol', dtype='f4')
>>> cols.create_column(cschema)
>>> assert np.all(cols['newfcol'][:] == 0)

>>> cschema = pyc.ColumnSchema('newscol', dtype='U4', fill_value='none')
>>> cols.create_column(cschema)
>>> assert np.all(cols['newscol'][:] == 'none')

# add a metadata entry
>>> weather = {'temperature': 30, 'humidity': 50}
>>> cols.create_meta('weather', weather)

# Update column data
cols['x'][10:20] = 3
cols['name'][[5, 6]] = ['alpha', 'beta']
cols['y'][50] = 1.25

# if you have indices it can be better not to rebuild them
# until all your updates are finished. Use an updating context
# for this: the index is not updated until exiting the context

with cols.updating():
    cols.append(newdata1)
    cols.append(newdata2)

with cols['id'].updating():
    cols['id'][5:10] = 33
    cols['id'][99:200] = 66

# if you have updated compressed columns, you will at some point want to run
# vaccum which defragments the files

# vacuum all compressed columns
cols.vacuum()

# a specific column
cols['id'].vacuum()

# you can vacuum on exit from updating contexts
with cols['id'].updating(vacuum=True):
    cols['id'][5:10] = 33
    cols['id'][99:200] = 66

# You can use update to update dictionary metadata.
# Only the fields in the input dictionary are updated or added
>>> cols.meta['weather'].update({'temperature': 30.1, 'extra': 5})

# completely overwrite
>>> cols['weather'].write([3, 4, 5])

#
# getting names of entries
#

# get all names, including direct sub tables of the root
>>> cols.names
['telemetry/', 'id', 'y', 'x', 'name', 'meta']

# only array column names
>>> cols.column_names
['id', 'y', 'x', 'name']

# only metadata names
>>> cols.meta_names
['versions', 'weather']

# only sub table directories (only those directly above root are shown)
>>> cols.sub_table_names
['telemetry/']

# reload all columns or specified column/column list
>>> cols.reload()

# delete a column or other entry.  This will prompt the user
# to confirm the deletion
>>> cols.delete_entry('id')
# no prompt
>>> cols.delete_entry('id', yes=True)

# delete all data in the columns store
# with prompt
>>> cols.delete()
# no prompt
>>> cols.delete(yes=True)

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
