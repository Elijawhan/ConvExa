#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <convexa.h>
#include <cxkernels.h>
PYBIND11_MAKE_OPAQUE(std::vector<double>);

namespace py = pybind11;

PYBIND11_MODULE(convexa_core, m) {
    m.def("host_dft", &ConvExa::host_dft, "Compute DFT of a real-valued signal", py::arg("signal"));

    m.def("host_convolve", &ConvExa::host_convolve, "Convolve signal with kernel", 
          py::arg("signal"), py::arg("kernel"));

    m.def("host_convolve_timing", &CXTiming::host_convolve, "Convolve signal with kernel", 
        py::arg("signal"), py::arg("kernel"), py::arg("result"));

    m.def("device_convolve_timing", &CXTiming::device_convolve, "Convolve signal with kernel", 
        py::arg("signal"), py::arg("kernel"), py::arg("result"));

    py::bind_vector<std::vector<double> >(m, "dArray");
}
