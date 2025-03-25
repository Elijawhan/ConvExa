#pragma once
#include <convexa.h>
#include <cxkernels.h>
#include "helper.h"
void d_main() {
    printf("\n******************************************\n");
    printf("Beginning DFT Test:\n");

    std::vector<double> myVec = {2, 1, 3, 5, 7, 6, 4};
    std::vector<float> myVecf = {2, 1, 3, 5, 7, 6, 4};

    std::vector<std::complex<double>> result;
    std::vector<std::complex<float>> resultf;
    std::vector<std::complex<double>> result_kernel;
    std::vector<std::complex<float>> result_kernelf;

    printf("Performing warm-up kernel runs...\n");
    (void)CXTiming::device_dft<double>(myVec, result);
    (void)CXTiming::device_dft<float>(myVecf, resultf);
    result.clear();
    resultf.clear();

    printf("\n=========== Testing doubles... ===========\n");
    float h_runtime = CXTiming::host_dft<double>(myVec, result);
    float k_runtime = CXTiming::device_dft<double>(myVec, result_kernel);
    printf("Host (double) ran for: %f ms.\n", h_runtime);
    HELP::print_vec_complex(result);
    printf("Kernel (double) ran for: %f ms.\n", k_runtime);
    HELP::print_vec_complex(result_kernel);

    printf("\n======== Testing 32-bit floats... ========\n");
    float h_runtimef = CXTiming::host_dft<float>(myVecf, resultf);
    float k_runtimef = CXTiming::device_dft<float>(myVecf, result_kernelf);
    printf("Host (float) ran for: %f ms.\n", h_runtimef);
    HELP::print_vec_complex(resultf);
    printf("Kernel (float) ran for: %f ms.\n", k_runtimef);
    HELP::print_vec_complex(result_kernelf);
    
}