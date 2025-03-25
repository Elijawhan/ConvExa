#include <convexa.h>
#include <cxkernels.h>
#include "helper.h"
#define numRuns 10
void c_main() {
    printf("\n******************************************\n");
    printf("Beginning Convolution Test:\n");
    std::vector<int16_t> data;
    HELP::wav_hdr hdr = HELP::read_wav("./cpp/audio/badadeedur.wav", &data);
    std::vector<int16_t> myKernel = {-1, 1, -2, 2, 1, -1}; 

    std::vector<int16_t> myResult_h;
    float base_convolve_h_timing = CXTiming::host_convolve(data, myKernel, myResult_h);
    std::vector<int16_t> myResult_d;
    float base_convolve_d_timing = CXTiming::device_convolve(data, myKernel, myResult_d);

    double re = std::get<0>(HELP::relative_error(myResult_d, myResult_h));
    printf("Relative Error device basic: %f ", re );
    if (re < HELP::MAX_RELATIVE_ERROR) printf("PASS\n"); else printf("FAIL\n");

    
    base_convolve_h_timing = 0;
    for (int i = 0; i < numRuns; i ++) {
        myResult_h.clear();
        base_convolve_h_timing += CXTiming::host_convolve(data, myKernel, myResult_h);
    }
    base_convolve_h_timing = base_convolve_h_timing / numRuns;
    base_convolve_d_timing = 0;
    for (int i = 0; i < numRuns; i ++) {
        myResult_d.clear();
        base_convolve_d_timing += CXTiming::device_convolve(data, myKernel, myResult_d);
    }
    base_convolve_d_timing = base_convolve_d_timing / numRuns;

    
    printf("Timing for %d points over %d runs\n", data.size() +myKernel.size() - 1, numRuns);
    printf("Host Impl Timing: %f ms\n", base_convolve_h_timing);
    printf("Basic Convolution Kernel Timing: %f ms\n", base_convolve_d_timing);
        HELP::wav_hdr hdr = HELP::read_wav("./cpp/audio/badadeedur.wav", &data);
    std::vector<int16_t> myKernel = {-1, 1, -2, 2, 1, -1}; 

    std::vector<int16_t> myResult_h;
    float base_convolve_h_timing = CXTiming::host_convolve(data, myKernel, myResult_h);
    std::vector<int16_t> myResult_d;
    float base_convolve_d_timing = CXTiming::device_convolve(data, myKernel, myResult_d);

    double re = std::get<0>(HELP::relative_error(myResult_d, myResult_h));
    printf("Relative Error device basic: %f ", re );
    if (re < HELP::MAX_RELATIVE_ERROR) printf("PASS\n"); else printf("FAIL\n");

    
    base_convolve_h_timing = 0;
    for (int i = 0; i < numRuns; i ++) {
        myResult_h.clear();
        base_convolve_h_timing += CXTiming::host_convolve(data, myKernel, myResult_h);
    }
    base_convolve_h_timing = base_convolve_h_timing / numRuns;
    base_convolve_d_timing = 0;
    for (int i = 0; i < numRuns; i ++) {
        myResult_d.clear();
        base_convolve_d_timing += CXTiming::device_convolve(data, myKernel, myResult_d);
    }
    base_convolve_d_timing = base_convolve_d_timing / numRuns;

    
    printf("Timing for %d points over %d runs\n", data.size() +myKernel.size() - 1, numRuns);
    printf("Host Impl Timing: %f ms\n", base_convolve_h_timing);
    printf("Basic Convolution Kernel Timing: %f ms\n", base_convolve_d_timing);



    HELP::write_wav("./cpp/audio/badadeedu.wav", myResult_d, hdr);

    
}