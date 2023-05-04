/*
   TODO

   - python wrapper class or inherited
   - read one without checking erros for speed?  Or is read slice OK?
   - maybe more error checking internally
   - printouts when verbose
*/
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

// #include <string.h>
// #include <math.h>
#include <Python.h>
#include "numpy/arrayobject.h" 

struct PyColumn {
    PyObject_HEAD

    const char* fname;
    const char* mode;
    int verbose;
    FILE* fptr;

};

static int
ensure_arrays_same_size(
  PyArrayObject* arr1, const char* name1,
  PyArrayObject* arr2, const char* name2
  ) {
    npy_intp num1 = 0, num2 = 0;
    num1 = PyArray_SIZE(arr1);
    num2 = PyArray_SIZE(arr2);

    if (num1 != num2) {
        PyErr_Format(
            PyExc_ValueError,
            "%s has size %" NPY_INTP_FMT
            " but %s has size %" NPY_INTP_FMT, name1, num1, name2, num2
        );
        return 0;
    } else {
        return 1;
    }
}

static int
check_array_size(PyArrayObject* array, npy_intp expected) {
    npy_intp num = 0;
    num = PyArray_SIZE(array);

    if (num != expected) {
        PyErr_Format(
            PyExc_ValueError,
            "array has size %" NPY_INTP_FMT
            " but expected %" NPY_INTP_FMT, num, expected
        );
        return 0;
    } else {
        return 1;
    }
}

static npy_intp
pyc_get_row_offset(npy_intp row, npy_intp elsize) {
    return row * elsize;
}

static void
pyc_seek_row(struct PyColumn* self, npy_intp row, npy_intp elsize) {
    npy_intp offset = 0;
    offset = pyc_get_row_offset(row, elsize);
    fseek(self->fptr, offset, SEEK_SET);
}

// Resize the file to nbytes, expanding with zeros if necessary
static PyObject*
PyColumn_resize_bytes(
    struct PyColumn* self,
    PyObject *args,
    PyObject *kwds
)
{
    long long nbytes = 0;

    if (!PyArg_ParseTuple(args, (char*)"L", &nbytes)) {
        return NULL;
    }

    rewind(self->fptr);
    if (0 != ftruncate(fileno(self->fptr), nbytes)) {
        PyErr_Format(PyExc_IOError,
                     "Could not truncate file to %lld bytes: %s",
                     nbytes, strerror(errno));
        return NULL;
    }
    fflush(self->fptr);

    Py_RETURN_NONE;
}

// Append data to file
static PyObject*
PyColumn_fill_slice(
    struct PyColumn* self,
    PyObject *args,
    PyObject *kwds
)
{
    long long start = 0, stop = 0;
    npy_intp elsize = 0, index = 0;
    PyArrayObject* array = NULL;

    if (!PyArg_ParseTuple(args, (char*)"OLL", &array, &start, &stop)) {
        return NULL;
    }

    if (!check_array_size(array, 1)) {
        return NULL;
    }

    elsize = PyArray_ITEMSIZE(array);

    for (index=start; index < stop; index ++) {
        pyc_seek_row(self, index, elsize);

        if (PyArray_ToFile(array, self->fptr, "", "") != 0) {
            PyErr_Format(PyExc_IOError,
                         "Error writing data to %s with mode %s",
                         self->fname, self->mode);
            return NULL;
        }
    }

    fflush(self->fptr);

    Py_RETURN_NONE;
}


// Append data to file
static PyObject*
PyColumn_append(
    struct PyColumn* self,
    PyObject *args,
    PyObject *kwds
)
{
    PyArrayObject* array = NULL;

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
    fflush(self->fptr);

    Py_RETURN_NONE;
}

// write data from the specified row location
static PyObject*
PyColumn_write_at(
    struct PyColumn* self,
    PyObject *args,
    PyObject *kwds
)
{
    PyArrayObject* array = NULL;
    long long start = 0;
    npy_intp elsize = 0;

    if (!PyArg_ParseTuple(args, (char*)"OL", &array, &start)) {
        return NULL;
    }

    elsize = PyArray_ITEMSIZE(array);
    pyc_seek_row(self, start, elsize);

    if (PyArray_ToFile(array, self->fptr, "", "") != 0) {
        PyErr_Format(PyExc_IOError,
                     "Error writing data to %s with mode %s",
                     self->fname, self->mode);
        return NULL;
    }
    fflush(self->fptr);

    Py_RETURN_NONE;
}

/*
   Write rows into file
   No error checking is done on the data type of the input array
   The rows should be sorted for efficiency, but this is not checked
   rows should be native npy_int64 but this is not checked
*/

static PyObject*
PyColumn_write_rows(
    struct PyColumn* self,
    PyObject *args,
    PyObject *kwds
)
{
    PyArrayObject* array = NULL, *rows = NULL;
    npy_int64 row = 0, elsize = 0;
    npy_intp nwrote = 0;
    PyArrayIterObject *it = NULL;
    // NPY_BEGIN_THREADS_DEF;

    if (!PyArg_ParseTuple(args, (char*)"OO", &array, &rows)) {
        return NULL;
    }

    if (!ensure_arrays_same_size(rows, "rows", array, "array")) {
        return NULL;
    }

    elsize = PyArray_ITEMSIZE(array);
    it = (PyArrayIterObject *) PyArray_IterNew((PyObject *)array);

    // NPY_BEGIN_THREADS;

    while (it->index < it->size) {
        row = *(npy_int64 *) PyArray_GETPTR1(rows, it->index);

        pyc_seek_row(self, row, elsize);

        nwrote = fwrite(
            (void *) it->dataptr,
            elsize,
            1,
            self->fptr
        );
        if (nwrote < 1) {
            // NPY_END_THREADS;
            PyErr_Format(PyExc_IOError,
                         "Error writing row %" NPY_INTP_FMT
                         " to file", row);
            Py_DECREF(it);
            return NULL;
        }
        PyArray_ITER_NEXT(it);
    }
    // NPY_END_THREADS;
    Py_DECREF(it);

    Py_RETURN_NONE;
}

/*
   Write rows into file with a sort index
   No error checking is done on the data type of the input array
   The rows should be sorted for efficiency, but this is not checked
   rows should be native npy_int64 but this is not checked
*/

static PyObject*
PyColumn_write_rows_sortind(
    struct PyColumn* self,
    PyObject *args,
    PyObject *kwds
)
{
    PyArrayObject* array = NULL, *rows = NULL, *sortind = NULL;
    npy_int64 row = 0, s = 0, elsize = 0, index = 0, size = 0;
    npy_intp nwrote = 0;
    void *ptr = NULL;
    // NPY_BEGIN_THREADS_DEF;

    if (!PyArg_ParseTuple(args, (char*)"OOO", &array, &rows, &sortind)) {
        return NULL;
    }

    if (!ensure_arrays_same_size(rows, "rows", array, "array")) {
        return NULL;
    }
    if (!ensure_arrays_same_size(rows, "rows", sortind, "sortind")) {
        return NULL;
    }

    size = PyArray_SIZE(array);
    elsize = PyArray_ITEMSIZE(array);

    // NPY_BEGIN_THREADS;

    for (index = 0; index < size; index ++) {
        s = *(npy_int64 *) PyArray_GETPTR1(sortind, index);
        ptr = PyArray_GETPTR1(array, s);
        row = *(npy_int64 *) PyArray_GETPTR1(rows, s);

        pyc_seek_row(self, row, elsize);

        nwrote = fwrite(
            ptr,
            elsize,
            1,
            self->fptr
        );
        if (nwrote < 1) {
            // NPY_END_THREADS;
            PyErr_Format(PyExc_IOError,
                         "Error writing row %" NPY_INTP_FMT
                         " to file", row);
            return NULL;
        }
    }
    // NPY_END_THREADS;

    Py_RETURN_NONE;
}


/*
   Read slice into array
   No error checking is done on the data type of the input array
*/
static PyObject*
PyColumn_read_slice(
    struct PyColumn* self,
    PyObject *args,
    PyObject *kwds
)
{
    PyArrayObject* array = NULL;
    long long start = 0;
    npy_intp num = 0, row = 0, nread = 0, elsize = 0;
    PyArrayIterObject *it = NULL;

    if (!PyArg_ParseTuple(args, (char*)"OL", &array, &start)) {
        return NULL;
    }

    elsize = PyArray_ITEMSIZE(array);
    num = PyArray_SIZE(array);

    pyc_seek_row(self, start, elsize);

    if (PyArray_ISCONTIGUOUS(array)) {
        if (self->verbose) {
             fprintf(stderr, "reading as contiguous\n");
        }
        // NPY_BEGIN_ALLOW_THREADS;
        nread = fread(PyArray_DATA(array),
                      elsize,
                      num,
                      self->fptr);
        // NPY_END_ALLOW_THREADS;
        if (nread != num) {
            PyErr_Format(PyExc_IOError,
                "Error reading slice [%" NPY_INTP_FMT ", %" NPY_INTP_FMT ") "
                "from %s", start, start+num, self->fname);
            return NULL;
        }
    } else {
        if (self->verbose) {
            fprintf(stderr, "reading as non contiguous\n");
        }
        // NPY_BEGIN_THREADS_DEF;

        it = (PyArrayIterObject *) PyArray_IterNew((PyObject *)array);

        // NPY_BEGIN_THREADS;

        row = start;
        while (it->index < it->size) {
            nread = fread(
                (void *) it->dataptr,
                elsize,
                1,
                self->fptr 
            );
            if (nread < 1) {
                // NPY_END_THREADS;
                PyErr_Format(PyExc_IOError,
                             "Error reading row %" NPY_INTP_FMT
                             " from file", row);
                Py_DECREF(it);
                return NULL;
            }
            row += 1;
            PyArray_ITER_NEXT(it);
        }
        // NPY_END_THREADS;
        Py_DECREF(it);

    }

    Py_RETURN_NONE;
}

/*
   Read rows into array
   No error checking is done on the data type of the input array
   The rows should be sorted for efficiency, but this is not checked
   rows should be native npy_int64 but this is not checked
*/

static PyObject*
PyColumn_read_rows(
    struct PyColumn* self,
    PyObject *args,
    PyObject *kwds
)
{
    PyArrayObject* array = NULL, *rows = NULL;
    npy_int64 row = 0, elsize = 0;
    npy_intp nread = 0;
    PyArrayIterObject *it = NULL;
    // NPY_BEGIN_THREADS_DEF;

    if (!PyArg_ParseTuple(args, (char*)"OO", &array, &rows)) {
        return NULL;
    }

    if (!ensure_arrays_same_size(rows, "rows", array, "array")) {
        return NULL;
    }

    elsize = PyArray_ITEMSIZE(array);
    it = (PyArrayIterObject *) PyArray_IterNew((PyObject *)array);

    // NPY_BEGIN_THREADS;

    while (it->index < it->size) {
        row = *(npy_int64 *) PyArray_GETPTR1(rows, it->index);

        pyc_seek_row(self, row, elsize);

        nread = fread(
            (void *) it->dataptr,
            elsize,
            1,
            self->fptr 
        );
        if (nread < 1) {
            // NPY_END_THREADS;
            PyErr_Format(PyExc_IOError,
                         "Error reading row %" NPY_INTP_FMT
                         " from file", row);
            Py_DECREF(it);
            return NULL;
        }
        PyArray_ITER_NEXT(it);
    }
    // NPY_END_THREADS;
    Py_DECREF(it);

    Py_RETURN_NONE;
}

/*
   Read rows into array with sort indices
   No error checking is done on the data type of the input array
   The rows should be sorted for efficiency, but this is not checked
   rows should be native npy_int64 but this is not checked
*/

static PyObject*
PyColumn_read_rows_sortind(
    struct PyColumn* self,
    PyObject *args,
    PyObject *kwds
)
{
    PyArrayObject* array = NULL, *rows = NULL, *sortind = NULL;
    npy_int64 row = 0, s = 0, elsize = 0, index = 0, size = 0;
    npy_intp nread = 0;
    // NPY_BEGIN_THREADS_DEF;
    void *ptr = NULL;

    if (!PyArg_ParseTuple(args, (char*)"OOO", &array, &rows, &sortind)) {
        return NULL;
    }

    if (!ensure_arrays_same_size(rows, "rows", array, "array")) {
        return NULL;
    }
    if (!ensure_arrays_same_size(rows, "rows", sortind, "sortind")) {
        return NULL;
    }

    size = PyArray_SIZE(array);
    elsize = PyArray_ITEMSIZE(array);

    // NPY_BEGIN_THREADS;

    for (index = 0; index < size; index ++) {
        s = *(npy_int64 *) PyArray_GETPTR1(sortind, index);
        ptr = PyArray_GETPTR1(array, s);
        row = *(npy_int64 *) PyArray_GETPTR1(rows, s);

        pyc_seek_row(self, row, elsize);

        nread = fread(
            ptr,
            elsize,
            1,
            self->fptr 
        );
        if (nread < 1) {
            // NPY_END_THREADS;
            PyErr_Format(PyExc_IOError,
                         "Error reading row %" NPY_INTP_FMT
                         " from file", row);
            return NULL;
        }
    }
    // NPY_END_THREADS;

    Py_RETURN_NONE;
}


static PyObject*
PyColumn_read_row(
    struct PyColumn* self,
    PyObject *args,
    PyObject *kwds
)
{
    PyArrayObject* array = NULL;
    long long row = 0;
    npy_intp nread = 0, elsize = 0;

    if (!PyArg_ParseTuple(args, (char*)"OL", &array, &row)) {
        return NULL;
    }

    if (!check_array_size(array, 1)) {
        return NULL;
    }

    elsize = PyArray_ITEMSIZE(array);

    pyc_seek_row(self, row, elsize);

    nread = fread(
        (void *) PyArray_GETPTR1(array, 0),
        elsize,
        1,
        self->fptr 
    );

    if (nread < 1) {
        PyErr_Format(PyExc_IOError,
                     "Error reading row %" NPY_INTP_FMT
                     " from file", row);
        return NULL;
    }

    Py_RETURN_NONE;
}

static int
PyColumn_init(struct PyColumn* self, PyObject *args, PyObject *kwds)
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

    self->fptr = fopen(self->fname, self->mode);
    if (!self->fptr) {
        PyErr_Format(PyExc_IOError, "Error opening %s with mode %s",
                     self->fname, self->mode);
        return -1;
    }

    return 0;
}

static PyObject *
PyColumn_close(struct PyColumn* self)
{
    if (self->fptr != NULL) {
        fclose(self->fptr);
        self->fptr = NULL;
    }
    Py_RETURN_NONE;
}



static void
PyColumn_dealloc(struct PyColumn* self)
{

    if (self->fptr != NULL) {
        fclose(self->fptr);
        self->fptr = NULL;
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject *
PyColumn_repr(struct PyColumn* self) {
    char buff[4096];
  
    snprintf(buff, 4096,
             "column\n"
             "    file: %s\n"
             "    mode: %s\n"
             "    verbose: %d\n",
             self->fname, self->mode, self->verbose);

    return PyUnicode_FromString((const char*)buff);
}

static PyMethodDef PyColumn_methods[] = {
    {"_resize_bytes",
     (PyCFunction)PyColumn_resize_bytes,
     METH_VARARGS, 
     "_resize_bytes()\n"
     "\n"
     "Resize the file to nbytes, expanding and filling with zeros if needed"},

    {"_fill_slice",
     (PyCFunction)PyColumn_fill_slice,
     METH_VARARGS, 
     "_fill_slice()\n"
     "\n"
     "Fill the slice with the give value."},

    {"_append",
     (PyCFunction)PyColumn_append,
     METH_VARARGS, 
     "_append()\n"
     "\n"
     "Append data.\n"},

    {"_write_at",
     (PyCFunction)PyColumn_write_at,
     METH_VARARGS, 
     "_write_at()\n"
     "\n"
     "Write data from the specified row.\n"},

    {"_write_rows",
     (PyCFunction)PyColumn_write_rows,
     METH_VARARGS, 
     "_write_rows()\n"
     "\n"
     "Write data at the specified row.\n"},

    {"_write_rows_sortind",
     (PyCFunction)PyColumn_write_rows_sortind,
     METH_VARARGS, 
     "_write_rows_sortind()\n"
     "\n"
     "Write data at the specified row.\n"},

    {"_read_slice",
     (PyCFunction)PyColumn_read_slice,
     METH_VARARGS, 
     "_read_slice()\n"
     "\n"
     "Read slice of data into input array.\n"},

    {"_read_rows",
     (PyCFunction)PyColumn_read_rows,
     METH_VARARGS, 
     "_read_rows()\n"
     "\n"
     "Read rows into input array.\n"},

    {"_read_rows_sortind",
     (PyCFunction)PyColumn_read_rows_sortind,
     METH_VARARGS, 
     "_read_rows_sortind()\n"
     "\n"
     "Read rows into input array using sort indices.\n"},

    /*
    {"_read_rows_pages",
     (PyCFunction)PyColumn_read_rows_pages,
     METH_VARARGS, 
     "_read_rows_pages()\n"
     "\n"
     "Read rows into input array.\n"},
     */

    {"_read_row",
     (PyCFunction)PyColumn_read_row,
     METH_VARARGS, 
     "_read_row()\n"
     "\n"
     "Read single row into input array.\n"},

    {"close",
     (PyCFunction)PyColumn_close,
     METH_VARARGS, 
     "close()\n"
     "\n"
     "close the file stream.\n"},

    {NULL}  /* Sentinel */
};

static PyTypeObject PyColumnType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_column.Column",             /*tp_name*/
    sizeof(struct PyColumn), /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)PyColumn_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    //0,                         /*tp_repr*/
    (reprfunc)PyColumn_repr,                         /*tp_repr*/
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
    PyColumn_methods,             /* tp_methods */
    0,             /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    //0,     /* tp_init */
    (initproc)PyColumn_init,      /* tp_init */
    0,                         /* tp_alloc */
    PyType_GenericNew,                 /* tp_new */
};

static PyMethodDef column_methods[] = {
    {NULL}  /* Sentinel */
};


static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_column_pywrap",      /* m_name */
    "Defines the Column class",  /* m_doc */
    -1,                  /* m_size */
    column_methods,    /* m_methods */
    NULL,                /* m_reload */
    NULL,                /* m_traverse */
    NULL,                /* m_clear */
    NULL,                /* m_free */
};

#ifndef PyMODINIT_FUNC  /* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC
PyInit__column_pywrap(void) 
{
    PyObject* m;


    PyColumnType.tp_new = PyType_GenericNew;

    if (PyType_Ready(&PyColumnType) < 0) {
        return NULL;
    }
    m = PyModule_Create(&moduledef);
    if (m==NULL) {
        return NULL;
    }

    Py_INCREF(&PyColumnType);
    PyModule_AddObject(m, "Column", (PyObject *)&PyColumnType);

    import_array();

    return m;
}
