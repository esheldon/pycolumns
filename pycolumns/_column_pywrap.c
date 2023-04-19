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

#define PYCOLUMN_FVERS 0

// size of header lines before END, including newlines
#define PYCOLUMN_HLINE_SIZE 29

// How much to read after last header line to eat up END and two
// newlines
// END\n\n
#define PYCOLUMN_ESIZE 5

// where we should be after reading header
#define PYCOLUMN_DATA_START 92

// Max size of these strings
#define PYCOLUMN_DTYPE_LEN 20
#define PYCOLUMN_FVERS_LEN 20

#define PYCOLUMN_PAGE_SIZE 2880

struct PyColumnPage {
    npy_intp pagenum;  // current loaded pagenum
    npy_intp begin;  // offset into file in bytes
    npy_intp end;  // offset into file in bytes, inclusive
    char buffer[PYCOLUMN_PAGE_SIZE];
};

struct PyColumn {
    PyObject_HEAD

    const char* fname;
    const char* mode;
    int verbose;
    FILE* fptr;

    int has_header;
    PY_LONG_LONG nrows;
    npy_intp file_end;

    char dtype[PYCOLUMN_DTYPE_LEN];
    char fvers[PYCOLUMN_FVERS_LEN];

    PyArray_Descr *descr;
    npy_intp elsize;

    struct PyColumnPage page;
};

static PyObject *
PyColumn_get_filename(struct PyColumn* self) {
    return PyUnicode_FromString(self->fname);
}
static PyObject *
PyColumn_get_mode(struct PyColumn* self) {
    return PyUnicode_FromString(self->mode);
}
static PyObject *
PyColumn_get_dtype(struct PyColumn* self) {
    return PyUnicode_FromString(self->dtype);
}
static PyObject *
PyColumn_get_nrows(struct PyColumn* self) {
    return PyLong_FromLongLong(self->nrows);
}
static PyObject *
PyColumn_has_header(struct PyColumn* self) {
    if (self->has_header) {
        Py_RETURN_TRUE;
    } else {
        Py_RETURN_FALSE;
    }
}

static int
pyc_check_elsize(struct PyColumn* self, PyObject* array) {
    npy_intp elsize = 0;

    elsize = PyArray_ITEMSIZE(array);

    if (elsize != self->elsize) {
        PyErr_Format(
            PyExc_ValueError,
            "input array elsize %" NPY_INTP_FMT
            " does not match file %" NPY_INTP_FMT,
            elsize, self->elsize
        );
        return 0;
    } else {
        return 1;
    }
}

static int
ensure_arrays_same_size(PyObject* arr1, const char* name1,
                        PyObject* arr2, const char* name2) {
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

static int
pyc_check_row(struct PyColumn* self, npy_intp row) {
    if (row > (self->nrows - 1)) {
        PyErr_Format(
            PyExc_ValueError,
            "row %" NPY_INTP_FMT
            " is out of bounds [0, %" NPY_INTP_FMT ")",
            row, self->nrows
        );
        return 0;
    } else {
        return 1;
    }
}

static int
check_slice(npy_intp start, npy_intp num, npy_intp nrows) {
    npy_intp end = 0;
    end = start + num;

    if (start < 0 || end > nrows) {
        PyErr_Format(
            PyExc_ValueError,
            "slice [%" NPY_INTP_FMT ", %"NPY_INTP_FMT ") "
            "is out of bounds [0, %" NPY_INTP_FMT ")",
            start, end, nrows
        );
        return 0;
    } else {
        return 1;
    }
}


static npy_intp
pyc_get_row_offset(struct PyColumn* self, npy_intp row) {
    return PYCOLUMN_DATA_START + row * self->elsize;
}

static void
pyc_seek_row(struct PyColumn* self, npy_intp row) {
    npy_intp offset = 0;
    offset = pyc_get_row_offset(self, row);
    fseek(self->fptr, offset, SEEK_SET);
}

static npy_intp
pyc_get_page_offset(struct PyColumn* self, npy_intp pagenum) {
    return PYCOLUMN_DATA_START +  pagenum * PYCOLUMN_PAGE_SIZE;
}

static npy_intp
PyColumn_get_page_containing(struct PyColumn* self, npy_intp row) {
    npy_intp row_offset, pagenum;
    row_offset = pyc_get_row_offset(self, row);

    row_offset -= PYCOLUMN_DATA_START;
    pagenum = row_offset / PYCOLUMN_PAGE_SIZE;
    return pagenum;
}

static int pyc_load_page(struct PyColumn* self, npy_intp pagenum) {
    npy_intp num2read = 0, nread = 0;

    self->page.pagenum = pagenum;

    self->page.begin = pyc_get_page_offset(self, pagenum);

    /* fprintf(stderr, "reading page: %ld\n", pagenum); */
    /* fprintf(stderr, "seeking to: %ld\n", self->page.begin); */
    fseek(self->fptr, self->page.begin, SEEK_SET);

    // end is like a slice
    self->page.end = self->page.begin + PYCOLUMN_PAGE_SIZE;
    if (self->page.end > self->file_end) {
        self->page.end = self->file_end;
    }

    num2read = self->page.end - self->page.begin;
    // fprintf(stderr, "num2read: %ld\n", num2read);

    if (num2read > PYCOLUMN_PAGE_SIZE) {
        // NPY_END_THREADS;
        PyErr_Format(PyExc_IOError,
                     "Got num2read %" NPY_INTP_FMT
                     " greater than buffsize %ld",
                     num2read, PYCOLUMN_PAGE_SIZE);
        return 0;
    }

    nread = fread(
        (const void *) self->page.buffer,
        num2read,
        1,
        self->fptr 
    );
    // fprintf(stderr, "nread: %ld\n", nread);
    if (nread < 1) {
        // NPY_END_THREADS;
        PyErr_Format(PyExc_IOError,
                     "Error reading page %" NPY_INTP_FMT
                     " from file", pagenum);
        return 0;
    }

}

static void copy_bytes_from_page(struct PyColumn* self, char *ptr,
                                 npy_intp start, npy_intp nbytes) {
    memcpy((void *)ptr, self->page.buffer + start, nbytes);
}

static int pyc_read_row_from_pages(
       struct PyColumn* self, npy_intp row, char* ptr
)
{
    npy_intp pagenum = 0, row_offset = 0, row_end = 0,
             offset_in_page = 0, bytes_to_copy = 0, bytes_to_copy2 = 0;

    pagenum = PyColumn_get_page_containing(self, row);

    if (pagenum != self->page.pagenum) {
        if (!pyc_load_page(self, pagenum)) {
            return 0;
        }
    }

    row_offset = pyc_get_row_offset(self, row);
    row_end = row_offset + self->elsize;

    offset_in_page = row_offset - self->page.begin;
    /* fprintf(stderr, */
    /*         "row_offset: %ld page begin: %ld row_end: %ld page end: %ld\n", */
    /*         row_offset, self->page.begin, row_end, self->page.end); */
    /* fprintf(stderr, "offset_in_page: %ld end_in_page: %ld\n", */
    /*         offset_in_page, offset_in_page + self->elsize); */

    // if (end_in_page > self->page.end) {
    if (row_end > self->page.end) {
        // fprintf(stderr, "crosses page boundary\n");
        // we will only read some of the bytes from this page
        // bytes_to_copy = self->page.end - offset_in_page;
        bytes_to_copy = self->page.end - row_offset;
        copy_bytes_from_page(self, ptr, offset_in_page, bytes_to_copy);

        bytes_to_copy2 = row_end - self->page.end;

        // load next page
        if (!pyc_load_page(self, pagenum + 1)) {
            return 0;
        }
        copy_bytes_from_page(self, ptr + bytes_to_copy, 0, bytes_to_copy2);

    } else {
        // fprintf(stderr, "does not cross page boundary\n");
        copy_bytes_from_page(self, ptr, offset_in_page, self->elsize);
    }

    return 1;

}

static int
pyc_read_nrows(struct PyColumn* self)
{
    size_t nread = 0, retval = 0;
    char instring[PYCOLUMN_HLINE_SIZE] = {0};

    // read nrows line
    nread = fread(
        instring,
        PYCOLUMN_HLINE_SIZE,
        1,
        self->fptr
    );
    if (nread != 1) {
        PyErr_Format(PyExc_IOError,
                     "Could not read nrows header line\n");
        goto ERROR;
    }
    nread = sscanf(instring, "NROWS = %lld", &self->nrows);
    if (nread != 1) {
        PyErr_Format(
            PyExc_IOError,
            "Bad nrows header line %s",
            instring
        );
        goto ERROR;
    }

    retval = 1;

ERROR:
    return retval;
}

static int
pyc_read_dtype(struct PyColumn* self)
{
    size_t nread = 0, retval = 0;
    char instring[PYCOLUMN_HLINE_SIZE] = {0};

    // read dtype line
    nread = fread(
        instring,
        PYCOLUMN_HLINE_SIZE,
        1,
        self->fptr
    );
    if (nread != 1) {
        PyErr_Format(PyExc_IOError,
                     "Could not read dtype header line\n");
        goto ERROR;
    }
    nread = sscanf(instring, "DTYPE = %s", self->dtype);
    if (nread != 1) {
        PyErr_Format(
            PyExc_IOError,
            "Bad dtype header line %s",
            instring
        );
        goto ERROR;
    }

    retval = 1;

ERROR:
    return retval;
}

static int
pyc_read_fvers(struct PyColumn* self)
{
    size_t nread = 0, retval = 0;
    char instring[PYCOLUMN_HLINE_SIZE] = {0};

    // read fvers line
    nread = fread(
        instring,
        PYCOLUMN_HLINE_SIZE,
        1,
        self->fptr
    );
    if (nread != 1) {
        PyErr_Format(PyExc_IOError,
                     "Could not read fvers header line\n");
        goto ERROR;
    }
    nread = sscanf(instring, "FVERS = %s", self->fvers);
    if (nread != 1) {
        PyErr_Format(
            PyExc_IOError,
            "Bad fvers header line %s",
            instring
        );
        goto ERROR;
    }

    retval = 1;

ERROR:
    return retval;
}

static int
pyc_read_end(struct PyColumn* self)
{
    size_t nread = 0, retval = 0;
    char instring[PYCOLUMN_HLINE_SIZE] = {0};

    // read END and two newlines
    nread = fread(
        instring,
        PYCOLUMN_ESIZE,
        1,
        self->fptr
    );
    if (nread != 1) {
        PyErr_Format(PyExc_IOError,
                     "Could not read header END\n");
        goto ERROR;
    }

    retval = 1;

ERROR:
    return retval;
}

static void
pyc_set_file_end(struct PyColumn* self) {
    self->file_end = pyc_get_row_offset(self, self->nrows);
}

static int
pyc_read_header(struct PyColumn* self)
{

    int retval = 0;
    PyObject *str = NULL;

    // SEEK_SET is from beginning
    fseek(self->fptr, 0, SEEK_SET);

    if (!pyc_read_nrows(self)) {
        goto ERROR;
    }
    if (!pyc_read_dtype(self)) {
        goto ERROR;
    }
    if (!pyc_read_fvers(self)) {
        goto ERROR;
    }
    if (!pyc_read_end(self)) {
        goto ERROR;
    }

    if (ftell(self->fptr) != PYCOLUMN_DATA_START) {
        PyErr_Format(PyExc_IOError,
                     "After head read got loc %lld instead of %lld",
                     ftell(self->fptr), PYCOLUMN_DATA_START);
        goto ERROR;
    }
    str = PyUnicode_FromString(self->dtype);
    if (!PyArray_DescrConverter(str, &self->descr)) {
        PyErr_Format(PyExc_IOError,
                     "Could not convert '%s' to a dtype",
                     self->dtype);
        goto ERROR;
    }
    self->elsize = self->descr->elsize;
    pyc_set_file_end(self);

    retval = 1;
    self->has_header = 1;

ERROR:
    Py_XDECREF(str);
    return retval;

}


static int
pyc_write_nrows(struct PyColumn* self, PY_LONG_LONG nrows) {
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

static int
pyc_update_nrows(struct PyColumn* self, npy_intp rows_added) {
    fseek(self->fptr, 0, SEEK_SET);

    self->nrows += rows_added;
    pyc_set_file_end(self);
    return pyc_write_nrows(self, self->nrows);
}

// write the header and load it
static PyObject*
PyColumn_write_initial_header(
    struct PyColumn* self,
    PyObject *args,
    PyObject *kwds
)
{
    char *dtype = NULL;
    int nwrote = 0;

    if (!PyArg_ParseTuple(args, (char*)"s", &dtype)) {
        return NULL;
    }
    if (self->has_header) {
        PyErr_Format(PyExc_RuntimeError,
                     "%s is already initialized",
                     self->fname);
        return NULL;
    }

    fseek(self->fptr, 0, SEEK_SET);

    if (!pyc_write_nrows(self, 0)) {
        return NULL;
    }
    fprintf(self->fptr, "DTYPE = %20s\n", dtype);
    fprintf(self->fptr, "FVERS = %20d\n", PYCOLUMN_FVERS);
    nwrote = fprintf(self->fptr, "END\n\n");

    if (nwrote == 0) {
        PyErr_Format(PyExc_IOError,
                     "Error writing initial header to %s with mode %s",
                     self->fname, self->mode);
        return NULL;
    }

    if (!pyc_read_header(self)) {
        return NULL;
    }

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

    if (!pyc_update_nrows(self, PyArray_SIZE(array))) {
        PyErr_Format(PyExc_IOError,
                     "Could not update nrows in %s",
                     self->fname);
        return NULL;
    }

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
    npy_intp num = 0, row = 0, nread = 0;
    PyArrayIterObject *it = NULL;

    if (!PyArg_ParseTuple(args, (char*)"OL", &array, &start)) {
        return NULL;
    }

    if (!pyc_check_elsize(self, array)) {
        return NULL;
    }
    num = PyArray_SIZE(array);
    if (!check_slice(start, num, self->nrows)) {
        return NULL;
    }

    pyc_seek_row(self, start);

    if (PyArray_ISCONTIGUOUS(array)) {
        if (self->verbose) {
             fprintf(stderr, "reading as contiguous\n");
        }
        // NPY_BEGIN_ALLOW_THREADS;
        nread = fread(PyArray_DATA(array),
                      self->elsize,
                      num,
                      self->fptr);
        // NPY_END_ALLOW_THREADS;
        if (nread != num) {
            PyErr_Format(PyExc_IOError,
                "Error reading slice [%" NPY_INTP_FMT ", %" NPY_INTP_FMT ")"
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
                (const void *) it->dataptr,
                self->elsize,
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
    npy_int64 row = 0;
    npy_intp nread = 0;
    PyArrayIterObject *it = NULL;
    // NPY_BEGIN_THREADS_DEF;

    if (!PyArg_ParseTuple(args, (char*)"OO", &array, &rows)) {
        return NULL;
    }

    if (!pyc_check_elsize(self, array)) {
        return NULL;
    }
    if (!ensure_arrays_same_size(rows, "rows", array, "array")) {
        return NULL;
    }

    it = (PyArrayIterObject *) PyArray_IterNew((PyObject *)array);

    // NPY_BEGIN_THREADS;

    while (it->index < it->size) {
        row = *(npy_int64 *) PyArray_GETPTR1(rows, it->index);

        if (row > (self->nrows - 1)) {
            // NPY_END_THREADS;
            PyErr_Format(PyExc_IOError,
                         "Attempt to read row %" NPY_INTP_FMT
                         " in file with %" NPY_INTP_FMT "rows",
                         row, self->nrows);
            Py_DECREF(it);
            return NULL;
        }

        pyc_seek_row(self, row);

        nread = fread(
            (const void *) it->dataptr,
            self->elsize,
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
   Read rows into array
   No error checking is done on the data type of the input array
   The rows should be sorted for efficiency, but this is not checked
   rows should be native npy_int64 but this is not checked
*/
static PyObject*
PyColumn_read_rows_pages(
    struct PyColumn* self,
    PyObject *args,
    PyObject *kwds
)
{
    PyArrayObject* array = NULL, *rows = NULL;
    npy_int64 row = 0;
    npy_intp nread = 0;
    PyArrayIterObject *it = NULL;

    // NPY_BEGIN_THREADS_DEF;

    if (!PyArg_ParseTuple(args, (char*)"OO", &array, &rows)) {
        return NULL;
    }

    if (!pyc_check_elsize(self, array)) {
        return NULL;
    }
    if (!ensure_arrays_same_size(rows, "rows", array, "array")) {
        return NULL;
    }

    it = (PyArrayIterObject *) PyArray_IterNew((PyObject *)array);

    // NPY_BEGIN_THREADS;

    while (it->index < it->size) {
        row = *(npy_int64 *) PyArray_GETPTR1(rows, it->index);

        if (row > (self->nrows - 1)) {
            // NPY_END_THREADS;
            PyErr_Format(PyExc_IOError,
                         "Attempt to read row %" NPY_INTP_FMT
                         " in file with %" NPY_INTP_FMT "rows",
                         row, self->nrows);
            Py_DECREF(it);
            return NULL;
        }

        if (!pyc_read_row_from_pages(self, row, it->dataptr)) {
            Py_DECREF(it);
            return NULL;
        }

        PyArray_ITER_NEXT(it);
    }
    // NPY_END_THREADS;
    Py_DECREF(it);

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
    npy_intp nread = 0;

    if (!PyArg_ParseTuple(args, (char*)"OL", &array, &row)) {
        return NULL;
    }

    if (!pyc_check_elsize(self, array)) {
        return NULL;
    }
    if (!check_array_size(array, 1)) {
        return NULL;
    }
    if (!pyc_check_row(self, row)) {
        return NULL;
    }

    pyc_seek_row(self, row);

    nread = fread(
        (const void *) PyArray_GETPTR1(array, 0),
        self->elsize,
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
    self->has_header = 0;
    self->nrows = 0;
    self->file_end = 0; // we find out when we read the header
    self->descr = NULL;
    self->elsize = -1;
    self->page.pagenum = -1;

    self->fptr = fopen(self->fname, self->mode);
    if (!self->fptr) {
        PyErr_Format(PyExc_IOError, "Error opening %s with mode %s",
                     self->fname, self->mode);
        return -1;
    }

    if (self->mode[0] == 'r') {
        if (!pyc_read_header(self)) {
            return -1;
        }
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
    Py_XDECREF(self->descr);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject *
PyColumn_repr(struct PyColumn* self) {
    char buff[4096];
  
    snprintf(buff, 4096,
             "column\n"
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

static PyMethodDef PyColumn_methods[] = {
    {"get_filename",
     (PyCFunction)PyColumn_get_filename,
     METH_VARARGS, 
     "get_filename()\n"
     "\n"
     "Get the filename.\n"},

    {"get_mode",
     (PyCFunction)PyColumn_get_mode,
     METH_VARARGS, 
     "get_mode()\n"
     "\n"
     "Get the file open mode.\n"},

    {"get_nrows",
     (PyCFunction)PyColumn_get_nrows,
     METH_VARARGS, 
     "get_nrows()\n"
     "\n"
     "Get the number of rows.\n"},

    {"get_dtype",
     (PyCFunction)PyColumn_get_dtype,
     METH_VARARGS, 
     "get_dtype()\n"
     "\n"
     "Get the dtype string.\n"},

    {"has_header",
     (PyCFunction)PyColumn_has_header,
     METH_VARARGS, 
     "has_header()\n"
     "\n"
     "Returns True of header has been written.\n"},


    {"init",
     (PyCFunction)PyColumn_write_initial_header,
     METH_VARARGS, 
     "init()\n"
     "\n"
     "Write an initial header.\n"},

    {"_append",
     (PyCFunction)PyColumn_append,
     METH_VARARGS, 
     "append()\n"
     "\n"
     "Append data.\n"},

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

    {"_read_rows_pages",
     (PyCFunction)PyColumn_read_rows_pages,
     METH_VARARGS, 
     "_read_rows_pages()\n"
     "\n"
     "Read rows into input array.\n"},

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
