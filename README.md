A simple, efficient column-oriented, pythonic data store.

The focus is currently on efficiency of reading and writing.  Basic consistency
is ensured but the database is not fully ACID.

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
    id              array    <i4 False  (64348146,)
    imag            array    <f4 False  (64348146,)
    ra              array    <f8 False  (64348146,)
    x               array    <f4 False  (64348146,)
    y               array    <f4 False  (64348146,)
    g               array    <f8 False  (64348146, 2)

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
  dtype: <i4

# get the column names
>>> c.colnames
['ccd', 'dec', 'exposurename', 'id', 'imag', 'ra', 'x', 'y', 'g']

# reload all columns or specified column/column list
>>> c.reload(name=None)

# read all data from column 'id'
# alternative syntaxes
>>> id = c['id'][:]
>>> id = c['id'].read()
>>> id = c.read_column('id')

# read a subset of rows
# slicing 
>>> id = c['id'][25:125]

# specifying a set of rows
>>> rows=[3, 225, 1235]
>>> id = c['id'][rows]
>>> id = c.read_column('id', rows=rows)

# read multiple columns into a single rec array
>>> data = c.read(columns=['id', 'flux'], rows=rows)

# or put different columns into fields of a dictionary instead of
# packing them into a single array
>>> data = c.read(columns=['id', 'flux'], asdict=True)

# Create indexes for fast searching
>>> c['col'].create_index()

# get indices for some condition
>>> ind = c['col'] > 25
>>> ind = c['col'].between(25,3 5)
>>> ind = c['col'] == 25
>>> ind = c['col'].match([25, 77])

# read the corresponding data
>>> ccd=c['ccd'][ind]
>>> data=c.read_columns(['ra', 'dec'], rows=ind)

# composite searches over multiple columns
>>> ind = (c['col1'] == 25) & (col['col2'] < 15.23)
>>> ind = c['col1'].between(15,25) | (c['col2'] != 66)
>>> ind = c['col1'].between(15,25) & (c['col2'] != 66) & (c['col3'] < 5)

# create a new column or append data to a column
>>> c.write_column(name, data)

# append to existing column, alternative syntax
>>> c['id'].write(data)

# write multiple columns from the fields in a rec array
# names in the data correspond to column names
>>> c.write(recdata)

# append more data
>>> c.write(recdata)

# write/append data from the fields in a FITS file
>>> c.from_fits(fitsfile_name)
```
