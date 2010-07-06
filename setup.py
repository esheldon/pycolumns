from distutils.core import setup,Extension
from os.path import join as path_join
import os,sys


# create the ups table
pyvers='%s.%s' % sys.version_info[0:2]
d1='lib/python%s/site-packages' % pyvers
d2='lib64/python%s/site-packages' % pyvers

if not os.path.exists('ups'):
    os.mkdir('ups')
tablefile=open('ups/columns.table','w')
tab="""
setupOptional("python")
setupOptional("numpydb")
setupOptional("esutil")
envPrepend(PYTHONPATH,${PRODUCT_DIR}/%s)
envPrepend(PYTHONPATH,${PRODUCT_DIR}/%s)
""" % (d1,d2)
tablefile.write(tab)
tablefile.close()

data_files=[]
data_files.append( ('ups',[path_join('ups','columns.table')] ) )

# data_files copies the ups/esutil.table into prefix/ups
setup(name='columns',
      description='Simple, efficient, pythonic column database',
      packages=['columns'],
      data_files=data_files)



