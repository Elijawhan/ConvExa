#pragma once
#include <convexa.h>
#include <cxkernels.h>
#include "helper.h"
void d_main() {
    printf("Hello from DFT Main!\n");

    std::vector<double> myVec = {2, 1, 3, 5, 7, 6, 4};
    std::vector<std::complex<double>> result;
    float h_runtime = CXTiming::host_dft<double>(myVec, result);
    printf("Host ran for: %f ms.\n", h_runtime);
    HELP::print_vec_complex(result);
    
    std::vector<float> myVecf = {2, 1, 3, 5, 7, 6, 4};
    std::vector<std::complex<float>> result_kernel;
    float k_runtime = CXTiming::device_dft<float>(myVecf, result_kernel);
    printf("Kernel ran for: %f ms.\n", k_runtime);
    HELP::print_vec_complex(result_kernel);
    
}