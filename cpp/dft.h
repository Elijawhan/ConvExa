#pragma once
#include <convexa.h>
#include <cxkernels.h>
#include <algorithm>
#include "helper.h"
void d_main() 
{
    
    printf("\n******************************************\n");
    printf("Beginning DFT Test:\n");

    //std::vector<double> myVec = {2, 1, 3, 5, 7, 6, 4};
    //std::vector<float> myVecf = {2, 1, 3, 5, 7, 6, 4};

    std::vector<int16_t> myVec_int;
    std::vector<double> myVec;
    std::vector<float> myVecf;
    // HELP::wav_hdr hdr = HELP::read_wav("./cpp/audio/badadeedur.wav", &myVec_int);

    for (int i = 0; i < 100; i++)
    {
        myVec.push_back(static_cast<double>(i));
        myVecf.push_back(static_cast<float>(i));
    }

    //myVec.resize(myVec_int.size());
    //myVecf.resize(myVec_int.size());
    
    
    //std::transform(myVec_int.begin(), myVec_int.end(), 
    //               myVec.begin(), [](int x){ return static_cast<double>(x); });
    //std::transform(myVec_int.begin(), myVec_int.end(), 
    //               myVecf.begin(), [](int x){ return static_cast<float>(x); });
    
    /*
    for (size_t i = 0; i < myVec_int.size(); i++) {
        myVec.push_back(static_cast<double>(myVec_int[i]));
        myVecf.push_back(static_cast<float>(myVec_int[i]));
    }
    */
    printf("Done converting Audio to floating point...\n");

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
    (void)CXTiming::host_dft<double>(myVec, result);
    (void)CXTiming::device_dft<double>(myVec, result_kernel);
    auto error_tupled = HELP::relative_error<std::complex<double>>(result_kernel, result);

    float h_runtime = 0.0, k_runtime = 0.0;
    for (int i = 0; i < numRuns; i++)
    {
        h_runtime += CXTiming::host_dft<double>(myVec, result);
        k_runtime += CXTiming::device_dft<double>(myVec, result_kernel);
    }
    h_runtime /= numRuns;
    k_runtime /= numRuns;

    printf("Host (double) ran for %f ms averaged over %d runs.\n", h_runtime, numRuns);
    //HELP::print_vec_complex(result);
    printf("Kernel (float) ran for %f ms averaged over %d runs.\n Error: %.8f (Absolute) and %.8f (Relative)\n",
            k_runtime, numRuns, std::get<0>(error_tupled), std::get<1>(error_tupled));
    //HELP::print_vec_complex(result_kernel);

    printf("\n======== Testing 32-bit floats... ========\n");
    (void)CXTiming::host_dft<float>(myVecf, resultf);
    (void)CXTiming::device_dft<float>(myVecf, result_kernelf);
    auto error_tuplef = HELP::relative_error<std::complex<float>>(result_kernelf, resultf);

    float h_runtimef = 0.0, k_runtimef = 0.0;
    for (int i = 0; i < numRuns; i++)
    {
        h_runtimef += CXTiming::host_dft<double>(myVec, result);
        k_runtimef += CXTiming::device_dft<double>(myVec, result_kernel);
    }
    h_runtimef /= numRuns;
    k_runtimef /= numRuns;

    printf("Host (float) ran for %f ms averaged over %d runs.\n", h_runtimef, numRuns);
    //HELP::print_vec_complex(resultf);
    printf("Kernel (float) ran for %f ms averaged over %d runs.\nError: %.8f (Absolute) and %.8f (Relative)\n",
        k_runtimef, numRuns, std::get<0>(error_tuplef), std::get<1>(error_tuplef));
    //HELP::print_vec_complex(result_kernelf);
    
}