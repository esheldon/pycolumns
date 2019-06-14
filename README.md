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
>>> c=pyc.Columns('/some/path/mycols.cols')

# display some info about the columns
>>> c
Column Directory:

  dir: /some/path/mydata.cols
  Columns:
    name             type  dtype index  shape
    --------------------------------------------------
    ccd             array    <i2 True   (64348146,)
    dec             array    <f8 False  (64348146,)
    exposurename    array   |S20 True   (64348146,)
    id              array    <i8 False  (64348146,)
    imag            array    <f4 False  (64348146,)
    ra              array    <f8 False  (64348146,)
    x               array    <f4 False  (64348146,)
    y               array    <f4 False  (64348146,)
    g               array    <f8 False  (64348146, 2)
    meta             dict


  Sub-Column Directories:
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

# get the column names
>>> c.colnames
['ccd', 'dec', 'exposurename', 'id', 'imag', 'ra', 'x', 'y', 'g', 'meta']

# reload all columns or specified column/column list
>>> c.reload(name=None)

# read all data from array column 'id'
# alternative syntaxes, including numpy style slicing
>>> ind = c['id'][:]
>>> ind = c['id'].read()
>>> ind = c.read_column('id')

# dict columns are read as a dict. No slicing for dicts
>>> meta = c['meta'].read()

# read a subset of rows
# slicing
>>> ind = c['id'][25:125]

# specifying a set of rows
>>> rows=[3, 225, 1235]
>>> ind = c['id'][rows]
>>> ind = c.read_column('id', rows=rows)

# read all columns into a single rec array.  By default the dict
# columns are not loaded

>>> data = c.read()

# using asdict=True puts the data into a dict.  The dict data
# are loaded in this case
>>> data = c.read(asdict=True)

# specify columns
>>> data = c.read(columns=['id', 'flux'], rows=rows)

# dict columns can be specified if asdict is True
>>> data = c.read(columns=['id', 'flux', 'meta'], asdict=True)

# Create indexes for fast searching
>>> c['id'].create_index()

# get indices for some condition
>>> ind = c['id'] > 25
>>> ind = c['id'].between(25, 35)
>>> ind = c['id'] == 25

# find all matches
>>> ind = c['id'].match([35, 77])

# read the corresponding data
>>> ccd = c['ccd'][ind]
>>> data = c.read(columns=['ra', 'dec'], rows=ind)

# composite searches over multiple columns
>>> ind = (c['id'] == 25) & (col['ra'] < 15.23)
>>> ind = c['id'].between(15, 25) | (c['id'] == 55)
>>> ind = c['id'].between(15, 250) & (c['id'] != 66) & (c['ra'] < 100)
>>> ind = c['id'].between(15, 250) & c['id'].match([35, 99])

# speed up reads by sorting indices
>>> ind.sort()
>>> data = c.read(columns=['ra', 'dec'], rows=ind)

# you can check if the index is already sorted
>>> if not ind.is_sorted:
>>>    ind.sort()

# update values for a column
>>> c['id'][35] = 10
>>> c['id'][35:35+3] = [8, 9, 10]
>>> c['id'][rows] = idvalues

# write multiple columns from the fields in a rec array
# names in the data correspond to column names.
# If columns are not present, they are created
# but row count consistency must be maintained for all array
# columns and this is checked.

>>> c.append(recdata)

# append data from the fields in a FITS file
>>> c.from_fits(fitsfile_name)

# add a dict column
>>> c.create_column('meta')
>>> c['meta'].write({'test': 'hello'})
>>> d = c['meta'].read()

# heirarchical sets of columns are also supported
>>> c['psfstars']
Column Directory:

  dir: /some/path/mydata.cols/psfstars.cols
  Columns:
    name             type  dtype index  shape
    --------------------------------------------------
    ccd             array    <i2 True   (348146,)
    exposurename    array   |S20 True   (348146,)
    x               array    <f4 False  (348146,)
    y               array    <f4 False  (348146,)
```

Dependencies
------------
numpy


