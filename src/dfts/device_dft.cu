#include <convexa.h>

namespace CXKernels
{
__global__ void device_dft(cuda::std::complex<float>* signal)
{
    int index = blockDim.x*blockIdx.x + threadIdx.x;
}
}