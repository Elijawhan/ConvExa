#include <convexa.h>
#include <cxkernels.h>
#include "helper.h"

void c_main() {
    printf("Hello from Convolve main!\n");
    std::vector<uint16_t> myVec = {2, 1, 3, 5, 7, 6, 4, 5, 7, 4, 2, 5, 5, 2, 9, 10};
    std::vector<uint16_t> myKernel = {1, 2, 2, 1}; 

    std::vector<uint16_t> myResult_h;
    float base_convolve_h_timing = CXTiming::host_convolve(myVec, myKernel, myResult_h);
    std::vector<uint16_t> myResult_d;
    float base_convolve_d_timing = CXTiming::device_convolve(myVec, myKernel, myResult_d);
    

    printf("%f thing\n", base_convolve_h_timing);
    printf("%f thing\n", base_convolve_d_timing);

    HELP::print_vec(myResult_h);
    HELP::print_vec(myResult_d);

    
}