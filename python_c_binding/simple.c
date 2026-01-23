
#include <Python.h>

static PyObject *simple_add(PyObject *self, PyObject *args)
{

    int a, b;

    if (!PyArg_ParseTuple(args, "ii", &a, &b))
    {
        return NULL;
    }

    return PyLong_FromLong(a + b);
}

static PyMethodDef SimpleMethods[] = {

    {"add", simple_add, METH_VARARGS, "Return the sum of two integers."},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef simplemodule = {
    PyModuleDef_HEAD_INIT,
    "simple",
    NULL,
    -1,
    SimpleMethods};

PyMODINIT_FUNC PyInit_simple(void)
{
    return PyModule_Create(&simplemodule);
}