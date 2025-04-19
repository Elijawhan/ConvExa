#pragma once
#include <convexa.h>
#include <cxkernels.h>
#include <algorithm>
#include <random>
#include "helper.h"
void d_main() 
{
    printf("\n******************************************\n");
    printf("Beginning DFT Test:\n");
    std::vector<int16_t> myVec_int;
    std::vector<double> myVec;
    std::vector<float> myVecf;

    // SMALL SIGNAL TEST
    //myVec = {2, 1, 3, 5, 7, 6, 4};
    //myVecf = {2, 1, 3, 5, 7, 6, 4};
    std::vector<double> throwaway_vec = {2, 1, 3, 5, 7, 6, 4};
    std::vector<float> throwaway_vecF = {2, 1, 3, 5, 7, 6, 4};

    // AUDIO FILE TEST (TAKES FOREVER ON HOST)
    /*
    HELP::wav_hdr hdr = HELP::read_wav("./cpp/audio/badadeedur.wav", &myVec_int);
    myVec.resize(myVec_int.size());
    myVecf.resize(myVec_int.size());
    std::transform(myVec_int.begin(), myVec_int.end(), 
                   myVec.begin(), [](int x){ return static_cast<double>(x); });
    std::transform(myVec_int.begin(), myVec_int.end(), 
                   myVecf.begin(), [](int x){ return static_cast<float>(x); });
    printf("Done converting Audio to floating point...\n");
    */

    // RANDOM NUMBER TEST (PICK ANY SIZE; <= 1000 for host)
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(1.0, 10000.0);
    for (int i = 0; i < 1000; i++)
    {
        myVec.push_back(dist(gen));
        myVecf.push_back(static_cast<float>(myVec.back()));
        //myVec.push_back(static_cast<double>(i));
        //myVecf.push_back(static_cast<float>(i));
    }
    
    printf("Size of input vectors: %d\n", myVec.size());

    std::vector<std::complex<double>> result;
    std::vector<std::complex<float>> resultf;
    std::vector<std::complex<double>> result_kernel;
    std::vector<std::complex<float>> result_kernelf;
    std::vector<std::complex<double>> result_kernel2;
    std::vector<std::complex<float>> result_kernelf2;

    printf("Performing warm-up kernel runs...\n");
    (void)CXTiming::device_dft<double>(throwaway_vec, result_kernel);
    (void)CXTiming::device_dft<float>(throwaway_vecF, result_kernelf);
    (void)CXTiming::device_fft_radix2<double>(throwaway_vec, result_kernel2);
    (void)CXTiming::device_fft_radix2<float>(throwaway_vecF, result_kernelf2);
    result_kernel.clear();
    result_kernelf.clear();
    result_kernel2.clear();
    result_kernelf2.clear();

    printf("\n======== Testing 64-bit doubles... ========\n");
    (void)CXTiming::host_dft<double>(myVec, result);
    (void)CXTiming::device_dft<double>(myVec, result_kernel);
    (void)CXTiming::device_fft_radix2<double>(myVec, result_kernel2);

    long double rel_error_d = HELP::relative_error(result_kernel, result);
    long double rel_error_d2 = HELP::relative_error(result_kernel2, result);

    float h_runtime = 0.0, k_runtime = 0.0, k_runtime2 = 0.0;
    for (int i = 0; i < numRuns; i++)
    {
        h_runtime += CXTiming::host_dft<double>(myVec, result);
        k_runtime += CXTiming::device_dft<double>(myVec, result_kernel);
        k_runtime2 += CXTiming::device_fft_radix2<double>(myVec, result_kernel2);
    }
    h_runtime /= numRuns;
    k_runtime /= numRuns;
    k_runtime2 /= numRuns;

    printf("Host (double) ran for %f ms averaged over %d runs.\n", h_runtime, numRuns);
    //HELP::print_vec_complex(result);

    printf("DFT Kernel (double) ran for %f ms averaged over %d runs.\nError: %.20Lf\n",
            k_runtime, numRuns, rel_error_d);
    //HELP::print_vec_complex(result_kernel);

    printf("FFT Radix2 Kernel (double) ran for %f ms averaged over %d runs.\nError: %.20Lf\n",
            k_runtime2, numRuns, rel_error_d2);
    //HELP::print_vec_complex(result_kernel);

    if (rel_error_d2 < HELP::MAX_RELATIVE_ERROR_DOUBLE)
        printf("FFT Error PASS!\n");
    else
        printf("FFT Error FAIL!\n");

    printf("\n======== Testing 32-bit floats... ========\n");
    (void)CXTiming::host_dft<float>(myVecf, resultf);
    (void)CXTiming::device_dft<float>(myVecf, result_kernelf);
    (void)CXTiming::device_fft_radix2<float>(myVecf, result_kernelf2);
    long double rel_error_f = HELP::relative_error(result_kernelf, resultf);
    long double rel_error_f2 = HELP::relative_error(result_kernelf2, resultf);

    float h_runtimef = 0.0, k_runtimef = 0.0, k_runtimef2 = 0.0;
    for (int i = 0; i < numRuns; i++)
    {
        h_runtimef += CXTiming::host_dft<float>(myVecf, resultf);
        k_runtimef += CXTiming::device_dft<float>(myVecf, result_kernelf);
        k_runtimef2 += CXTiming::device_fft_radix2<float>(myVecf, result_kernelf2);
    }
    h_runtimef /= numRuns;
    k_runtimef /= numRuns;
    k_runtimef2 /= numRuns;

    printf("Host (float) ran for %f ms averaged over %d runs.\n", h_runtimef, numRuns);
    //HELP::print_vec_complex(resultf);

    printf("DFT Kernel (float) ran for %f ms averaged over %d runs.\nError: %.20Lf\n",
        k_runtimef, numRuns, rel_error_f);
    //HELP::print_vec_complex(result_kernelf);

    printf("FFT Radix2 Kernel (float) ran for %f ms averaged over %d runs.\nError: %.20Lf\n",
        k_runtimef2, numRuns, rel_error_f2);
    //HELP::print_vec_complex(result_kernelf);

    if (rel_error_f2 < HELP::MAX_RELATIVE_ERROR_FLOAT)
        printf("FFT Error PASS!\n");
    else
        printf("FFT Error FAIL!\n");
    
}