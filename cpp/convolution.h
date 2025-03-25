#include <convexa.h>
#include "helper.h"

void c_main() {
    printf("Hello from Convolve main!\n");
    std::vector<double> myVec = {2, 1, 3, 5, 7, 6, 4};
    std::vector<double> myKernel = {0.125, -0.125}; 

    std::vector<double> myResult = ConvExa::host_convolve(myVec, myKernel);

    HELP::print_vec(myResult);
    
}