A simple, efficient column-oriented, pythonic database.  Data are input and
output as numerical python arrays, and indexing is provided by berkeley db
through the numpydb package.

The focus is currently on efficiency of reading and writing.  This is not yet
a proper database in terms of transactional integrity.  I think of it as
essentially a write once, ready many store.

Currently this package depends on fitsio
    git@github.com:esheldon/fitsio.git
and optionally numpydb
    git@github.com:esheldon/numpydb.git
for indexing.  A TODO is to examine copying the needed parts into pycolumns.

== Examples ==
```python
>>> import pycolumns as pyc


# instantiate a column database from the specified coldir
>>> c=pyc.Columns('/some/path/mycols.cols')


# display some info about the columns
>>> c
Column Directory:

  dir: /some/path/mydata.cols
  Columns:
    name             type  dtype index  size
    --------------------------------------------------
    ccd               col    <i2 True   64348146
    dec               col    <f8 False  64348146
    exposurename      col   |S20 True   64348146
    id                col    <i4 False  64348146
    imag              col    <f4 False  64348146
    ra                col    <f8 False  64348146
    x                 col    <f4 False  64348146
    y                 col    <f4 False  64348146

  Sub-Column Directories:
    name
    --------------------------------------------------
    psfstars


  ...

# display info about column 'id'
>>> c['id']
Column:
  "id"
  filename: ./id.col
  type: col
  size: 64348146
  has index: False
  dtype:
    [('id', '<i4')]


# get the column names
>>> c.colnames()
['ccd','dec','exposurename','id','imag','ra','x','y']

# reload all columns or specified column/column list
>>> c.reload(name=None)

# read all data from column 'id'
# alternative syntaxes
>>> id = c['id'][:]
>>> id = c['id'].read()
>>> id = c.read_column('id')

# Also get metadata
>>> id, id_meta = c.read_column('id', rows=rows, meta=True)
>>> meta=c['id'].read_meta()

# read a subset of rows
# slicing 
>>> id = c['id'][25:125]

# specifying a set of rows
>>> rows=[3,225,1235]
>>> id = c['id'][rows]
>>> id = c.read_column('id', rows=rows)


# read multiple columns into a single rec array
>>> data = c.read_columns(['id','flux'], rows=rows)

# or put different columns into fields of a dictionary instead of
# packing them into a single array
>>> data = c.read_columns(['id','flux'], asdict=True)

# If numpydb is available, you can create indexes and
# perform fast searching
>>> c['col'].create_index()

# get indices for some condition
>>> ind=(c['col'] > 25)
>>> ind=c['col'].between(25,35)
>>> ind=(c['col'] == 25)
>>> ind=c['col'].match([25,77])

# read the corresponding data
>>> ccd=c['ccd'][ind]
>>> data=c.read_columns(['ra','dec'], rows=ind)

# composite searches over multiple columns
>>> ind = (c['col1'] == 25) & (col['col2'] < 15.23)
>>> ind = c['col1'].between(15,25) | (c['col2'] != 66)
>>> ind = c['col1'].between(15,25) & (c['col2'] != 66) & (c['col3'] < 5)

# create column or append data to a column
>>> c.write_column(name, data)

# append to existing column, alternative syntax
>>> c['id'].write(data)

# write multiple columns from the fields in a rec array
# names in the data correspond to column names
>>> c.write_columns(recdata)

# write/append data from the fields in a .rec file or .fits file
>>> c.from_rec(recfile_name)
>>> c.from_fits(fitsfile_name)
```
