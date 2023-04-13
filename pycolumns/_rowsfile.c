#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

// #include <string.h>
// #include <math.h>
#include <Python.h>
#include "numpy/arrayobject.h" 

#define PYROWSFILE_SSIZE 21
// END\n\n
#define PYROWSFILE_ESIZE 5

// where we should be after reading header
#define PYROWSFILE_ELOC 47

struct PyRowsFile {
    PyObject_HEAD

    const char* fname;
    const char* mode;
    int verbose;
    FILE* fptr;

    char nrows_string[PYROWSFILE_SSIZE];
    char dtype_string[PYROWSFILE_SSIZE];

    int has_data;
    PY_LONG_LONG nrows;
};

static int
PyRowsFile_read_header(struct PyRowsFile* self)
{
    size_t nread = 0;
    char end[PYROWSFILE_ESIZE];

    memset(end, 0, PYROWSFILE_ESIZE);

    fseek(self->fptr, 0, SEEK_SET);

    nread = fread(
        self->nrows_string,
        PYROWSFILE_SSIZE,
        1,
        self->fptr
    );
    fprintf(stderr, "nread: %ld\n", nread);
    fprintf(stderr, "%s", self->nrows_string);
    if (nread != 1) {
    // if (nread != PYROWSFILE_SSIZE) {
        return 0;
    }
    fprintf(stderr, "loc: %ld\n", ftell(self->fptr));

    nread = fread(
        self->dtype_string,
        PYROWSFILE_SSIZE,
        1,
        self->fptr
    );
    fprintf(stderr, "nread: %ld\n", nread);
    fprintf(stderr, "%s", self->dtype_string);
    if (nread != 1) {
    // if (ferror(self->fptr)) {
    // if (nread != PYROWSFILE_SSIZE) {
        return 0;
    }
    fprintf(stderr, "loc: %ld\n", ftell(self->fptr));

    // Now the END and extra newline
    nread = fread(
        end,
        PYROWSFILE_ESIZE,
        1,
        self->fptr
    );
    fprintf(stderr, "nread: %ld\n", nread);
    fprintf(stderr, "%s", end);
    if (nread != 1) {
    // if (ferror(self->fptr)) {
    // if (nread != PYROWSFILE_SSIZE) {
        return 0;
    }

    fprintf(stderr, "loc: %ld\n", ftell(self->fptr));
    if (ftell(self->fptr) != PYROWSFILE_ELOC) {
        return 0;
    }
    return 1;
}

static int
PyRowsFile_init(struct PyRowsFile* self, PyObject *args, PyObject *kwds)
{
    char* fname=NULL;
    char* mode=NULL;
    int verbose=0;
    if (!PyArg_ParseTuple(args, (char*)"ssi", &fname, &mode, &verbose)) {
        return -1;
    }

    self->fname = (char *) fname;
    self->mode = (char *) mode;
    self->verbose = verbose;
    self->has_data = 0;
    self->nrows = 0;

    memset(self->nrows_string, 0, PYROWSFILE_SSIZE);
    memset(self->dtype_string, 0, PYROWSFILE_SSIZE);

    self->fptr = fopen(self->fname, self->mode);
    if (!self->fptr) {
        PyErr_Format(PyExc_IOError, "Error opening %s with mode %s",
                     self->fname, self->mode);
        return -1;
    }

    if (self->mode[0] == 'r') {
        if (!PyRowsFile_read_header(self)) {
            // fclose(self->fptr);
            PyErr_Format(PyExc_IOError,
                         "Error parsing header of %s",
                         self->fname);
            return -1;
        }
    }

    return 0;
}


static void
PyRowsFile_dealloc(struct PyRowsFile* self)
{

    if (self->fptr != NULL) {
        fclose(self->fptr);
        self->fptr = NULL;
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject *
PyRowsFile_repr(struct PyRowsFile* self) {
    char buff[4096];
  
    snprintf(buff, 4096,
             "RowsFile\n"
             "    file: %s\n"
             "    mode: %s\n"
             "    verbose: %d\n"
             "    nrows: %lld\n"
             "    has_data: %d\n",
             self->fname, self->mode, self->verbose,
             self->nrows, self->has_data);

    return PyUnicode_FromString((const char*)buff);
}

static PyObject *
PyRowsFile_get_filename(struct PyRowsFile* self) {
    return PyUnicode_FromString(self->fname);
}

static PyMethodDef PyRowsFile_methods[] = {
    {"get_filename", (PyCFunction)PyRowsFile_get_filename, METH_VARARGS, 
     "get_filename()\n"
     "\n"
     "Get the filename.\n"},
    {NULL}  /* Sentinel */
};

static PyTypeObject PyRowsFileType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_rowsfile.RowsFile",             /*tp_name*/
    sizeof(struct PyRowsFile), /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)PyRowsFile_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    //0,                         /*tp_repr*/
    (reprfunc)PyRowsFile_repr,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    "A class to work with rows of data.\n",
    0,                     /* tp_traverse */
    0,                     /* tp_clear */
    0,                     /* tp_richcompare */
    0,                     /* tp_weaklistoffset */
    0,                     /* tp_iter */
    0,                     /* tp_iternext */
    PyRowsFile_methods,             /* tp_methods */
    0,             /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    //0,     /* tp_init */
    (initproc)PyRowsFile_init,      /* tp_init */
    0,                         /* tp_alloc */
    PyType_GenericNew,                 /* tp_new */
};

static PyMethodDef rowsfile_methods[] = {
    {NULL}  /* Sentinel */
};


static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_rowsfile",      /* m_name */
    "Defines the RowsFile class",  /* m_doc */
    -1,                  /* m_size */
    rowsfile_methods,    /* m_methods */
    NULL,                /* m_reload */
    NULL,                /* m_traverse */
    NULL,                /* m_clear */
    NULL,                /* m_free */
};

#ifndef PyMODINIT_FUNC  /* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC
PyInit__rowsfile(void) 
{
    PyObject* m;


    PyRowsFileType.tp_new = PyType_GenericNew;

    if (PyType_Ready(&PyRowsFileType) < 0) {
        return NULL;
    }
    m = PyModule_Create(&moduledef);
    if (m==NULL) {
        return NULL;
    }

    Py_INCREF(&PyRowsFileType);
    PyModule_AddObject(m, "RowsFile", (PyObject *)&PyRowsFileType);

    import_array();

    return m;
}
