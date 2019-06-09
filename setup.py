from distutils.core import setup,Extension


# data_files copies the ups/esutil.table into prefix/ups
setup(
    name='pycolumns',
    version='0.1.0',
    description='A simple, efficient, pythonic column database',
    packages=['pycolumns'],
)
