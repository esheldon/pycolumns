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

with open(os.path.join(os.path.dirname(__file__), "README.md")) as fp:
    long_description = fp.read()

setup(
    name='pycolumns',
    packages=find_packages(),
    license='MIT',
    url='https://github.com/esheldon/pycolumns',
    version=__version__,
    description='A simple, efficient, pythonic column data store',
    long_description=long_description,
    long_description_content_type='text/markdown; charset=UTF-8; variant=GFM',
    setup_requires=['numpy', 'fitsio'],
    install_requires=['numpy', 'fitsio'],
)
