A simple, efficient column-oriented, pythonic data store.

The focus is currently on efficiency of reading and writing.  The code is pure
python but searching and reading data is fast due to the use of numpy memory
maps and column indexing.  Basic consistency is ensured but the database is not
fully ACID.

The storage is a simple directory with files on disk.

Examples
--------

```python
>>> import pycolumns as pyc

# instantiate a column database from the specified coldir
>>> c = pyc.Columns('/some/path/mycols.cols')

# display some info about the columns
>>> c
Columns Directory:

  dir: /some/path/mydata.cols
  Columns:
    name             type  dtype index  shape
    --------------------------------------------------
    ccd             array    <i2 True   (64348146,)
    dec             array    <f8 False  (64348146,)
    exposurename    array   |S20 True   (64348146,)
    id              array    <i8 True   (64348146,)
    imag            array    <f4 False  (64348146,)
    ra              array    <f8 False  (64348146,)
    x               array    <f4 False  (64348146,)
    y               array    <f4 False  (64348146,)
    g               array    <f8 False  (64348146, 2)
    meta             dict


  Sub-Columns Directories:
    name
    --------------------------------------------------
    psfstars

# display info about column 'id'
>>> c['id']
Column:
  "id"
  filename: ./id.array
  type: col
  shape: (64348146,)
  has index: False
  dtype: <i8

# number of rows in table
>>> c.nrows

# read all columns into a single rec array.  By default the dict
# columns are not loaded

>>> data = c.read()

# using asdict=True puts the data into a dict.  The dict data
# are loaded in this case
>>> data = c.read(asdict=True)

# specify columns
>>> data = c.read(columns=['id', 'flux'])

# dict columns can be specified if asdict is True.  Dicts can also
# be read as single columns, see below
>>> data = c.read(columns=['id', 'flux', 'meta'], asdict=True)

# specifying a set of rows as sequence/array or slice
>>> data = c.read(columns=['id', 'flux'], rows=[3, 225, 1235])
>>> data = c.read(columns=['id', 'flux'], rows=slice(10, 20))

# read all data from column 'id' as an array rather than recarray
# alternative syntaxes
>>> ind = c['id'][:]
>>> ind = c['id'].read()
>>> ind = c.read_column('id')

# read a subset of rows
# slicing
>>> ind = c['id'][25:125]

# specifying a set of rows
>>> rows = [3, 225, 1235]
>>> ind = c['id'][rows]
>>> ind = c.read_column('id', rows=rows)

# reading a dict column
>>> meta = c['meta'].read()

# Create indexes for fast searching
>>> c['id'].create_index()

# get indices for some conditions
>>> ind = c['id'] > 25
>>> ind = c['id'].between(25, 35)
>>> ind = c['id'] == 25

# read the corresponding data
>>> ccd = c['ccd'][ind]
>>> data = c.read(columns=['ra', 'dec'], rows=ind)

# composite searches over multiple columns
>>> ind = (c['id'] == 25) & (col['ra'] < 15.23)
>>> ind = c['id'].between(15, 25) | (c['id'] == 55)
>>> ind = c['id'].between(15, 250) & (c['id'] != 66) & (c['ra'] < 100)

# write columns from the fields in a rec array names in the data correspond
# to column names.  If this is the first time writing data, the columns are
# created, and on subsequent writes, the columns must match

>>> c.append(recdata)
>>> c.append(new_data)

# append data from the fields in a FITS file
>>> c.from_fits(fitsfile_name)

# add a dict column.
>>> c.create_column('weather', 'dict')
>>> c['weather'].write({'temp': 30.1, 'humid': 0.5})

# overwrite dict column
>>> c['weather'].write({'temp': 33.2, 'humid': 0.3, 'windspeed': 60.5})

# you should not generally create array columns, since they
# can get out of sync with existing columns.  This will by default
# raise an exception, but you can send verify=False if you know
# what you are doing.  In the future support will be added for this.
>>> c.create_column('test', 'array')

# update values for an array column
>>> c['id'][35] = 10
>>> c['id'][35:35+3] = [8, 9, 10]
>>> c['id'][rows] = idvalues

# get the column names
>>> c.colnames
['ccd', 'dec', 'exposurename', 'id', 'imag', 'ra', 'x', 'y', 'g',
 'meta', 'psfstars']
# only array columns
>>> c.array_colnames
['ccd', 'dec', 'exposurename', 'id', 'imag', 'ra', 'x', 'y', 'g']
# only dict columns
>>> c.dict_colnames
['meta', 'weather']
# only sub pycolumns directories
>>> c.cols_colnames
['psfstars']

# reload all columns or specified column/column list
>>> c.reload()

# delete all data.  This will ask for confirmation
>>> c.delete()

# delete column and its data
>>> c.delete_column('ra')

# to modify the amount of memory used during index creation, specify
# cache_mem in gigabytes.  Default 1 gig
>>> cols = pyc.Columns(fname, cache_mem=0.5)

# columns can actually be another pycolumns directory
>>> psfcols = cols['psfstars']
>>> psfcols
  dir: /some/path/mydata.cols/psfstars.cols
  Columns:
    name             type  dtype index  shape
    --------------------------------------------------
    ccd             array    <i2 True   (64348146,)
    id              array    <i8 True   (64348146,)
    imag            array    <f4 False  (64348146,)
    x               array    <f4 False  (64348146,)
    y               array    <f4 False  (64348146,)
    ...etc
```

Dependencies
------------
numpy


