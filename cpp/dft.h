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
    //HELP::wav_hdr hdr = HELP::read_wav("./cpp/audio/badadeedur.wav", &myVec_int);
    
    for (int i = 0; i < 1000; i++)
    {
        myVec.push_back(static_cast<double>(i));
        myVecf.push_back(static_cast<float>(i));
    }
    
    myVec.resize(myVec_int.size());
    myVecf.resize(myVec_int.size());
    
    /*
    std::transform(myVec_int.begin(), myVec_int.end(), 
                   myVec.begin(), [](int x){ return static_cast<double>(x); });
    std::transform(myVec_int.begin(), myVec_int.end(), 
                   myVecf.begin(), [](int x){ return static_cast<float>(x); });
    */
    
    //printf("Done converting Audio to floating point...\n");
    printf("Size of input vectors: %d\n", myVec.size());

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
    //(void)CXTiming::host_dft<double>(myVec, result);
    //(void)CXTiming::device_dft<double>(myVec, result_kernel);
    //long double rel_error_d = HELP::relative_error(result_kernel, result);
    long double rel_error_d = 0;
    float h_runtime = 0.0, k_runtime = 0.0;
    for (int i = 0; i < numRuns; i++)
    {
        //h_runtime += CXTiming::host_dft<double>(myVec, result);
        k_runtime += CXTiming::device_dft<double>(myVec, result_kernel);
    }
    h_runtime /= numRuns;
    k_runtime /= numRuns;

    printf("Host (double) ran for %f ms averaged over %d runs.\n", h_runtime, numRuns);
    //HELP::print_vec_complex(result);
    printf("Kernel (double) ran for %f ms averaged over %d runs.\nError: %.20Lf\n",
            k_runtime, numRuns, rel_error_d);
    //HELP::print_vec_complex(result_kernel);

    printf("\n======== Testing 32-bit floats... ========\n");
    //(void)CXTiming::host_dft<float>(myVecf, resultf);
    //(void)CXTiming::device_dft<float>(myVecf, result_kernelf);
    //long double rel_error_f = HELP::relative_error(result_kernelf, resultf);
    long double rel_error_f = 0;
    float h_runtimef = 0.0, k_runtimef = 0.0;
    for (int i = 0; i < numRuns; i++)
    {
        //h_runtimef += CXTiming::host_dft<double>(myVec, result);
        k_runtimef += CXTiming::device_dft<double>(myVec, result_kernel);
    }
    h_runtimef /= numRuns;
    k_runtimef /= numRuns;

    printf("Host (float) ran for %f ms averaged over %d runs.\n", h_runtimef, numRuns);
    //HELP::print_vec_complex(resultf);
    printf("Kernel (float) ran for %f ms averaged over %d runs.\nError: %.20Lf\n",
        k_runtimef, numRuns, rel_error_f);
    //HELP::print_vec_complex(result_kernelf);
    
}