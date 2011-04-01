from distutils.core import setup,Extension
import distutils.sysconfig
from os.path import join as path_join
import os,sys

main_libdir=distutils.sysconfig.get_python_lib()
pylib_install_subdir = main_libdir.replace(distutils.sysconfig.PREFIX+os.sep,'')

if not os.path.exists('ups'):
    os.mkdir('ups')
tablefile=open('ups/columns.table','w')
tab="""
setupOptional("python")
setupOptional("numpydb")
setupOptional("esutil")
envPrepend(PYTHONPATH,${PRODUCT_DIR}/%s)
""" % pylib_install_subdir 

tablefile.write(tab)
tablefile.close()

data_files=[]
data_files.append( ('ups',[path_join('ups','columns.table')] ) )

# data_files copies the ups/esutil.table into prefix/ups
setup(name='columns',
      description='A simple, efficient, pythonic column database',
      packages=['columns'],
      data_files=data_files)



