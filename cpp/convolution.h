#include <convexa.h>
#include <cxkernels.h>
#include "helper.h"

void c_main() {
    printf("Hello from Convolve main!\n");
    std::vector<double> myVec = {2, 1, 3, 5, 7, 6, 4};
    std::vector<double> myKernel = {0.125, -0.125}; 

    std::vector<double> myResult_h;
    float base_convolve_h_timing = CXTiming::host_convolve(myVec, myKernel, myResult_h);
    std::vector<double> myResult_d;
    float base_convolve_d_timing = CXTiming::device_convolve(myVec, myKernel, myResult_d);
    

    printf("%f thing\n", base_convolve_h_timing);
    printf("%f thing\n", base_convolve_d_timing);

    HELP::print_vec(myResult_h);
    HELP::print_vec(myResult_d);

    
}