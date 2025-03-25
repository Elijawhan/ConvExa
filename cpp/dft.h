#pragma once
#include <convexa.h>
#include <cxkernels.h>
#include "helper.h"
void d_main() {
    printf("Hello from DFT Main!\n");

    std::vector<double> myVec = {2, 1, 3, 5, 7, 6, 4};
    std::vector<std::complex<double>> result = ConvExa::host_dft(myVec);
    HELP::print_vec_complex(result);
    
    std::vector<std::complex<double>> result_kernel;
    double runtime = CXTiming::device_dft(myVec, result_kernel);
    printf("Comparing to kernel t = %f...\n", runtime);
    HELP::print_vec(result_kernel);
    
}