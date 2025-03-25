#include <convexa.h>
#include <cxkernels.h>
#include "helper.h"

void c_main() {
    printf("Hello from Convolve main!\n");
    std::vector<int16_t> data;
    HELP::wav_hdr hdr = HELP::read_wav("./cpp/audio/badadeedur.wav", &data);
    std::vector<int16_t> myKernel = {-1, 1, -2, 2, 1, -1}; 

    std::vector<int16_t> myResult_h;
    float base_convolve_h_timing = CXTiming::host_convolve(data, myKernel, myResult_h);
    std::vector<int16_t> myResult_d;
    float base_convolve_d_timing = CXTiming::device_convolve(data, myKernel, myResult_d);
    



    HELP::write_wav("./cpp/audio/badadeedu.wav", myResult_d, hdr);

    
}