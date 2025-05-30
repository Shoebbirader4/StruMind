#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

extern "C" void solve_fem(double* K, double* F, double* x, int n);

namespace py = pybind11;

void py_solve_fem(py::array_t<double> K, py::array_t<double> F, py::array_t<double> x) {
    int n = K.shape(0);
    solve_fem((double*)K.data(), (double*)F.data(), (double*)x.mutable_data(), n);
}

PYBIND11_MODULE(cppsolver, m) {
    m.def("solve_fem", &py_solve_fem, "Call C++ FEM solver");
} 