import os
from setuptools import setup, find_packages

__version__ = None
pth = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'pycolumns',
    'version.py',
)

with open(pth, 'r') as fp:
    exec(fp.read())


# data_files copies the ups/esutil.table into prefix/ups
setup(
    name='pycolumns',
    version=__version__,
    description='A simple, efficient, pythonic column data store',
    packages=find_packages(),
)
