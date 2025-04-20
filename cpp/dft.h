#pragma once
#include <convexa.h>
#include <cxkernels.h>
#include <algorithm>
#include <random>
#include <functional>
#include <helper.h>

template< typename T >
void test_dft_kernel(std::function< float (const std::vector<T>&, std::vector<std::complex<T>>&) > test_func,
                     std::function< float (const std::vector<T>&, std::vector<std::complex<T>>&) > reference_func,
                     const std::vector<T> &stimulus, long double max_error);


void d_main() 
{
    printf("\n******************************************\n");
    printf("Beginning Fourier Transform Testing..\n");
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
    std::uniform_real_distribution<double> dist(1.0, 100.0);
    for (int i = 0; i < pow(2,10); i++)
    {
        myVec.push_back(dist(gen));
        myVecf.push_back(static_cast<float>(myVec.back()));
        //myVec.push_back(static_cast<double>(i)*100000);
        //myVecf.push_back(static_cast<float>(i)*100000);
    }
    
    printf("Size of input vectors: %d\n", myVec.size());
    std::cout << "==================== " << "FP64" << " ====================" << std::endl;
    /*
    std::cout << "TESTING DFT:" << std::endl;
    test_dft_kernel<double>(
        static_cast<std::function< float (const std::vector<double>&, std::vector<std::complex<double>>&) >>(CXTiming::device_dft<double>),
        static_cast<std::function< float (const std::vector<double>&, std::vector<std::complex<double>>&) >>(CXTiming::host_dft<double>),
        myVec, HELP::MAX_RELATIVE_ERROR_DOUBLE
    );
    std::cout << std::endl;
    */
    std::cout << "TESTING FFT RADIX2:" << std::endl;
    test_dft_kernel<double>(
        static_cast<std::function< float (const std::vector<double>&, std::vector<std::complex<double>>&) >>(CXTiming::device_fft_radix2<double>),
        static_cast<std::function< float (const std::vector<double>&, std::vector<std::complex<double>>&) >>(CXTiming::cufft<double>),
        myVec, HELP::MAX_RELATIVE_ERROR_DOUBLE
    );
    std::cout << std::endl;
    /*
    std::cout << "TESTING FFT RADIX2 W/ HOST:" << std::endl;
    test_dft_kernel<double>(
        static_cast<std::function< float (const std::vector<double>&, std::vector<std::complex<double>>&) >>(CXTiming::host_fft_radix2<double>),
        static_cast<std::function< float (const std::vector<double>&, std::vector<std::complex<double>>&) >>(CXTiming::cufft<double>),
        myVec, HELP::MAX_RELATIVE_ERROR_DOUBLE
    );
    std::cout << std::endl;
    */
    std::cout << std::endl;
    std::cout << "==================== " << "FP32" << " ====================" << std::endl;
    /*
    std::cout << "TESTING DFT:" << std::endl;
    test_dft_kernel<float>(
        static_cast<std::function< float (const std::vector<float>&, std::vector<std::complex<float>>&) >>(CXTiming::device_dft<float>),
        static_cast<std::function< float (const std::vector<float>&, std::vector<std::complex<float>>&) >>(CXTiming::host_dft<float>),
        myVecf, HELP::MAX_RELATIVE_ERROR_FLOAT
    );
    std::cout << std::endl;
    */
    std::cout << "TESTING FFT RADIX2:" << std::endl;
    test_dft_kernel<float>(
        static_cast<std::function< float (const std::vector<float>&, std::vector<std::complex<float>>&) >>(CXTiming::device_fft_radix2<float>),
        static_cast<std::function< float (const std::vector<float>&, std::vector<std::complex<float>>&) >>(CXTiming::cufft<float>),
        myVecf, HELP::MAX_RELATIVE_ERROR_FLOAT
    );
    std::cout << std::endl;
    /*
    std::cout << "TESTING FFT RADIX2 W/ HOST:" << std::endl;
    test_dft_kernel<float>(
        static_cast<std::function< float (const std::vector<float>&, std::vector<std::complex<float>>&) >>(CXTiming::host_fft_radix2<float>),
        static_cast<std::function< float (const std::vector<float>&, std::vector<std::complex<float>>&) >>(CXTiming::cufft<float>),
        myVecf, HELP::MAX_RELATIVE_ERROR_FLOAT
    );
    std::cout << std::endl;
    */
}

template< typename T >
void test_dft_kernel(std::function< float (const std::vector<T>&, std::vector<std::complex<T>>&) > test_func,
                     std::function< float (const std::vector<T>&, std::vector<std::complex<T>>&) > reference_func,
                     const std::vector<T> &stimulus, long double max_error)
{
    std::vector<std::complex<T>> reference_result;
    std::vector<std::complex<T>> test_result;

    // Warm-up kernel
    (void)test_func(stimulus, test_result);
    test_result.clear();

    // Test relative error
    (void)reference_func(stimulus, reference_result);
    (void)test_func(stimulus, test_result);

    long double rel_error = HELP::relative_error(test_result, reference_result);

    float h_runtime = 0.0, k_runtime = 0.0;
    for (int i = 0; i < numRuns; i++)
    {
        h_runtime += reference_func(stimulus, reference_result);
        k_runtime += test_func(stimulus, test_result);
    }
    h_runtime /= numRuns;
    k_runtime /= numRuns;

    printf("Host ran for %f ms averaged over %d runs.\n", h_runtime, numRuns);
    printf("Kernel ran for %f ms averaged over %d runs.\nError: %.20Lf\n",
            k_runtime, numRuns, rel_error);
    if (rel_error < max_error)
        printf("Error PASS!\n");
    else
        printf("Error FAIL!\n");

    //HELP::print_vec_complex(test_result);
    //HELP::print_vec_complex(reference_result);
}