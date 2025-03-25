#pragma once
#include <convexa.h>
#include <cxkernels.h>
#include "helper.h"
void d_main() {
    printf("Hello from DFT Main!\n");

    std::vector<double> myVec = {2, 1, 3, 5, 7, 6, 4};
    std::vector<std::complex<double>> result;
    double h_runtime = CXTiming::host_dft(myVec, result);
    printf("Host ran for: %f ms.\n", h_runtime);
    HELP::print_vec_complex(result);
    
    std::vector<std::complex<double>> result_kernel;
    double k_runtime = CXTiming::device_dft(myVec, result_kernel);
    printf("Kernel ran for: %f ms.\n", k_runtime);
    HELP::print_vec_complex(result_kernel);
    
}