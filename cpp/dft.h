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
    auto error_tupled = HELP::relative_error<std::complex<double>>(result_kernel, result);
    printf("Host (double) ran for %f ms.\n", h_runtime);
    HELP::print_vec_complex(result);
    printf("Kernel (double) ran for %f ms with an error of %.8f (Absolute) and %.8f (Relative)\n",
        k_runtime, std::get<0>(error_tupled), std::get<1>(error_tupled));
    HELP::print_vec_complex(result_kernel);

    printf("\n======== Testing 32-bit floats... ========\n");
    float h_runtimef = CXTiming::host_dft<float>(myVecf, resultf);
    float k_runtimef = CXTiming::device_dft<float>(myVecf, result_kernelf);
    auto error_tuplef = HELP::relative_error<std::complex<float>>(result_kernelf, resultf);
    printf("Host (float) ran for: %f ms.\n", h_runtimef);
    HELP::print_vec_complex(resultf);
    printf("Kernel (float) ran for: %f ms with an error of %.8f (Absolute) and %.8f (Relative)\n",
        k_runtimef, std::get<0>(error_tuplef), std::get<1>(error_tuplef));
    HELP::print_vec_complex(result_kernelf);
    
}