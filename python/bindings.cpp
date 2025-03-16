#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>
#include <convexa.h>

namespace py = pybind11;

PYBIND11_MODULE(convexa_core, m) {
    m.def("host_dft", &host_dft, "Compute DFT of a real-valued signal", py::arg("signal"));
    m.def("host_convolve", &host_convolve, "Convolve signal with kernel", 
          py::arg("signal"), py::arg("kernel"));
}
