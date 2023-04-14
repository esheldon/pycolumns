import os
import glob
from setuptools import setup, find_packages
from setuptools.extension import Extension
from setuptools.command import build_ext

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

ext = Extension(
    "pycolumns._column",
    ["pycolumns/_column.c"],
)


class BuildExt(build_ext.build_ext):
    '''Custom build_ext command to hide the numpy import
    Inspired by http://stackoverflow.com/a/21621689/1860757'''
    def finalize_options(self):
        '''add numpy includes to the include dirs'''
        build_ext.build_ext.finalize_options(self)
        import numpy as np
        self.include_dirs.append(np.get_include())
        self.include_dirs.extend(glob.glob("pymangle/*h"))


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
    ext_modules=[ext],
    cmdclass={'build_ext': BuildExt},
)
