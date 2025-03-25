#include <convexa.h>
#include <cxkernels.h>
#include "helper.h"

void c_main() {
    printf("Hello from Convolve main!\n");
    std::vector<int16_t> data;
    HELP::wav_hdr hdr = HELP::read_wav("/home/uahclsd0085/ConvExa/cpp/audio/badadeedur.wav", &data);
    std::vector<int16_t> myVec = {2, 1, 3, 5, 7, 6, 4, 5, 7, 4, 2, 5, 5, 2, 9, 10};
    std::vector<int16_t> myKernel = {-1, 1, -2, 2, 1, -1}; 

    std::vector<int16_t> myResult_h;
    float base_convolve_h_timing = CXTiming::host_convolve(data, myKernel, myResult_h);
    std::vector<int16_t> myResult_d;
    float base_convolve_d_timing = CXTiming::device_convolve(data, myKernel, myResult_d);
    



    HELP::write_wav("/home/uahclsd0085/ConvExa/cpp/audio/badadeedu.wav", myResult_d, hdr);

    
}