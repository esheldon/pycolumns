#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

// #include <string.h>
// #include <math.h>
#include <Python.h>
#include "numpy/arrayobject.h" 

// size of header lines before END, including newlines
#define PYCOLUMNFILE_SSIZE 29

// How much to read after last header line to eat up END and two
// newlines
// END\n\n
#define PYCOLUMNFILE_ESIZE 5

// where we should be after reading header
#define PYCOLUMNFILE_ELOC 92

// Max size of these strings
#define PYCOLUMNFILE_DTYPE_LEN 20
#define PYCOLUMNFILE_FVERS_LEN 20

struct PyColumnFile {
    PyObject_HEAD

    const char* fname;
    const char* mode;
    int verbose;
    FILE* fptr;

    int has_header;
    PY_LONG_LONG nrows;
    char dtype[PYCOLUMNFILE_DTYPE_LEN];
    char fvers[PYCOLUMNFILE_FVERS_LEN];
};

static PyObject *
PyColumnFile_get_filename(struct PyColumnFile* self) {
    return PyUnicode_FromString(self->fname);
}
static PyObject *
PyColumnFile_get_nrows(struct PyColumnFile* self) {
    return PyLong_FromLongLong(self->nrows);
}


static int
PyColumnFile_read_nrows(struct PyColumnFile* self)
{
    size_t nread = 0;
    char instring[PYCOLUMNFILE_SSIZE] = {0};

    // read nrows string
    nread = fread(
        instring,
        PYCOLUMNFILE_SSIZE,
        1,
        self->fptr
    );
    fprintf(stderr, "nread: %ld\n", nread);
    fprintf(stderr, "%s", instring);
    if (nread != 1) {
        return 0;
    }
    fprintf(stderr, "loc: %ld\n", ftell(self->fptr));
    nread = sscanf(instring, "NROWS = %lld", &self->nrows);
    if (nread != 1) {
        return 0;
    }

    return 1;
}

static int
PyColumnFile_read_dtype(struct PyColumnFile* self)
{
    size_t nread = 0;
    char instring[PYCOLUMNFILE_SSIZE] = {0};

    // read nrows string
    nread = fread(
        instring,
        PYCOLUMNFILE_SSIZE,
        1,
        self->fptr
    );
    fprintf(stderr, "nread: %ld\n", nread);
    fprintf(stderr, "%s", instring);
    if (nread != 1) {
        return 0;
    }
    fprintf(stderr, "loc: %ld\n", ftell(self->fptr));
    nread = sscanf(instring, "DTYPE = %s", self->dtype);
    if (nread != 1) {
        return 0;
    }

    return 1;
}

static int
PyColumnFile_read_fvers(struct PyColumnFile* self)
{
    size_t nread = 0;
    char instring[PYCOLUMNFILE_SSIZE] = {0};

    // read nrows string
    nread = fread(
        instring,
        PYCOLUMNFILE_SSIZE,
        1,
        self->fptr
    );
    fprintf(stderr, "nread: %ld\n", nread);
    fprintf(stderr, "%s", instring);
    if (nread != 1) {
        return 0;
    }
    fprintf(stderr, "loc: %ld\n", ftell(self->fptr));
    nread = sscanf(instring, "FVERS = %s", self->fvers);
    if (nread != 1) {
        return 0;
    }

    return 1;
}

static int
PyColumnFile_read_end(struct PyColumnFile* self)
{
    size_t nread = 0;
    char instring[PYCOLUMNFILE_SSIZE] = {0};

    // read nrows string
    nread = fread(
        instring,
        PYCOLUMNFILE_ESIZE,
        1,
        self->fptr
    );
    fprintf(stderr, "nread: %ld\n", nread);
    fprintf(stderr, "%s", instring);
    if (nread != 1) {
        return 0;
    }

    return 1;
}


static int
read_header(struct PyColumnFile* self)
{

    // SEEK_SET is from beginning
    fseek(self->fptr, 0, SEEK_SET);

    if (!PyColumnFile_read_nrows(self)) {
        return 0;
    }
    if (!PyColumnFile_read_dtype(self)) {
        return 0;
    }
    if (!PyColumnFile_read_fvers(self)) {
        return 0;
    }
    if (!PyColumnFile_read_end(self)) {
        return 0;
    }

    fprintf(stderr, "loc: %ld\n", ftell(self->fptr));
    if (ftell(self->fptr) != PYCOLUMNFILE_ELOC) {
        PyErr_Format(PyExc_IOError,
                     "After head read got loc %lld instead of %lld",
                     ftell(self->fptr), PYCOLUMNFILE_ELOC);
        return 0;
    }

    self->has_header = 1;
    return 1;
}


static int
write_nrows(struct PyColumnFile* self, PY_LONG_LONG nrows) {
    int nwrote = 0;
    nwrote = fprintf(self->fptr, "NROWS = %20lld\n", nrows);
    if (nwrote == 0) {
        PyErr_Format(PyExc_IOError,
                     "Error writing initial nrows to %s with mode %s",
                     self->fname, self->mode);
        return 0;
    } else {
        return 1;
    }
}

// write the header and load it
static PyObject*
PyColumnFile_write_initial_header(
    struct PyColumnFile* self,
    PyObject *args,
    PyObject *kwds
)
{
    char *dtype = NULL;
    char *fvers = NULL;
    int nwrote = 0;

    if (!PyArg_ParseTuple(args, (char*)"ss", &dtype, &fvers)) {
        return NULL;
    }

    fseek(self->fptr, 0, SEEK_SET);

    if (!write_nrows(self, 0)) {
        return NULL;
    }
    fprintf(self->fptr, "DTYPE = %20s\n", dtype);
    fprintf(self->fptr, "FVERS = %20s\n", fvers);
    nwrote = fprintf(self->fptr, "END\n\n");

    if (nwrote == 0) {
        PyErr_Format(PyExc_IOError,
                     "Error writing initial header to %s with mode %s",
                     self->fname, self->mode);
        return NULL;
    }

    if (!read_header(self)) {
        return NULL;
    }

    Py_RETURN_NONE;
}


static int
update_nrows(struct PyColumnFile* self, npy_intp rows_added) {
    fseek(self->fptr, 0, SEEK_SET);

    self->nrows += rows_added;
    return write_nrows(self, self->nrows);
}

// Append data to file
static PyObject*
PyColumnFile_append(
    struct PyColumnFile* self,
    PyObject *args,
    PyObject *kwds
)
{
    PyObject* array = NULL;

    if (!PyArg_ParseTuple(args, (char*)"O", &array)) {
        return NULL;
    }

    fseek(self->fptr, 0, SEEK_END);
    if (PyArray_ToFile(array, self->fptr, "", "") != 0) {
        PyErr_Format(PyExc_IOError,
                     "Error writing data to %s with mode %s",
                     self->fname, self->mode);
        return NULL;
    }

    if (!update_nrows(self, PyArray_SIZE(array))) {
        PyErr_Format(PyExc_IOError,
                     "Could not update nrows in %s",
                     self->fname);
        return NULL;
    }

    Py_RETURN_NONE;
}


static PY_LONG_LONG
get_row_offset(row, elsize) {
    return PYCOLUMNFILE_ELOC + row * elsize;
}

// read data into input array as a slice
// array must be contiguous
static PyObject*
PyColumnFile_read_slice(
    struct PyColumnFile* self,
    PyObject *args,
    PyObject *kwds
)
{
    PyObject* arrayo = NULL;
    PyArrayObject* array = NULL;
    long long start =0;
    npy_intp num = 0;
    npy_intp end = 0;
    npy_intp elsize = 0;
    npy_intp offset = 0;
    npy_intp nread = 0;
    npy_intp max_possible = 0;

    if (!PyArg_ParseTuple(args, (char*)"OL", &arrayo, &start)) {
        return NULL;
    }

    array = (PyArrayObject *) arrayo;

    num = PyArray_SIZE(array);
    elsize = PyArray_ITEMSIZE(array);
    offset = get_row_offset(start, elsize);
    end = get_row_offset(start + num, elsize);
    max_possible = get_row_offset(self->nrows, elsize);

    if (offset > max_possible || end > max_possible) {
        PyErr_Format(PyExc_IOError,
                     "Attempt to read rows [%ld, %ld) goes beyond EOF",
                     start, start + num);
        return NULL;
    }
    fprintf(stderr, "seeking to: %ld\n", offset);

    // SEEK_SET is from beginning
    // note fseek does not set EOF or an error, would need to
    // try the read first, hence check above
    fseek(self->fptr, offset, SEEK_SET);

    fprintf(stderr, "at: %ld\n", ftell(self->fptr));

    fprintf(stderr, "reading: elsize: %ld num: %ld\n", elsize, num);

    NPY_BEGIN_ALLOW_THREADS;
    nread = fread(
        PyArray_DATA(array),
        elsize,
        num,
        self->fptr
    );
    NPY_END_ALLOW_THREADS;

    fprintf(stderr, "read: %ld\n", nread);
    if (nread != num) {
        PyErr_Format(PyExc_IOError,
                     "Error reading %lld from %s",
                     num, self->fname);
        return NULL;
    }

    Py_RETURN_NONE;
}




static int
PyColumnFile_init(struct PyColumnFile* self, PyObject *args, PyObject *kwds)
{
    char* fname = NULL;
    char* mode = NULL;
    int verbose = 0;

    if (!PyArg_ParseTuple(args, (char*)"ssi", &fname, &mode, &verbose)) {
        return -1;
    }

    self->fname = (char *) fname;
    self->mode = (char *) mode;
    self->verbose = verbose;
    self->has_header = 0;
    self->nrows = 0;

    self->fptr = fopen(self->fname, self->mode);
    if (!self->fptr) {
        PyErr_Format(PyExc_IOError, "Error opening %s with mode %s",
                     self->fname, self->mode);
        return -1;
    }

    if (self->mode[0] == 'r') {
        if (!read_header(self)) {
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
PyColumnFile_dealloc(struct PyColumnFile* self)
{

    if (self->fptr != NULL) {
        fclose(self->fptr);
        self->fptr = NULL;
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject *
PyColumnFile_repr(struct PyColumnFile* self) {
    char buff[4096];
  
    snprintf(buff, 4096,
             "ColumnFile\n"
             "    file: %s\n"
             "    mode: %s\n"
             "    verbose: %d\n"
             "    nrows: %lld\n"
             "    dtype: %s\n"
             "    fvers: %s\n"
             "    has_header: %d\n",
             self->fname, self->mode, self->verbose,
             self->nrows, self->dtype, self->fvers,
             self->has_header);

    return PyUnicode_FromString((const char*)buff);
}

static PyMethodDef PyColumnFile_methods[] = {
    {"get_filename",
     (PyCFunction)PyColumnFile_get_filename,
     METH_VARARGS, 
     "get_filename()\n"
     "\n"
     "Get the filename.\n"},

    {"get_nrows",
     (PyCFunction)PyColumnFile_get_nrows,
     METH_VARARGS, 
     "get_nrows()\n"
     "\n"
     "Get the number of rows.\n"},

    {"write_initial_header",
     (PyCFunction)PyColumnFile_write_initial_header,
     METH_VARARGS, 
     "write_initial_header()\n"
     "\n"
     "Write an initial header.\n"},

    {"append",
     (PyCFunction)PyColumnFile_append,
     METH_VARARGS, 
     "append()\n"
     "\n"
     "Append data.\n"},

    {"read_slice",
     (PyCFunction)PyColumnFile_read_slice,
     METH_VARARGS, 
     "read_slice()\n"
     "\n"
     "Read slice of data into input array.\n"},

    {NULL}  /* Sentinel */
};

static PyTypeObject PyColumnFileType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_columnfile.ColumnFile",             /*tp_name*/
    sizeof(struct PyColumnFile), /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)PyColumnFile_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    //0,                         /*tp_repr*/
    (reprfunc)PyColumnFile_repr,                         /*tp_repr*/
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
    PyColumnFile_methods,             /* tp_methods */
    0,             /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    //0,     /* tp_init */
    (initproc)PyColumnFile_init,      /* tp_init */
    0,                         /* tp_alloc */
    PyType_GenericNew,                 /* tp_new */
};

static PyMethodDef columnfile_methods[] = {
    {NULL}  /* Sentinel */
};


static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_columnfile",      /* m_name */
    "Defines the ColumnFile class",  /* m_doc */
    -1,                  /* m_size */
    columnfile_methods,    /* m_methods */
    NULL,                /* m_reload */
    NULL,                /* m_traverse */
    NULL,                /* m_clear */
    NULL,                /* m_free */
};

#ifndef PyMODINIT_FUNC  /* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC
PyInit__columnfile(void) 
{
    PyObject* m;


    PyColumnFileType.tp_new = PyType_GenericNew;

    if (PyType_Ready(&PyColumnFileType) < 0) {
        return NULL;
    }
    m = PyModule_Create(&moduledef);
    if (m==NULL) {
        return NULL;
    }

    Py_INCREF(&PyColumnFileType);
    PyModule_AddObject(m, "ColumnFile", (PyObject *)&PyColumnFileType);

    import_array();

    return m;
}
