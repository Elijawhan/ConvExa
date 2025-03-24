#include <convexa.h>
#include <cuda_runtime.h>
#include <cuda/std/complex>

namespace CXKernels
{
__global__ void device_dft(float* signal, uint32_t length, cuda::std::complex<float>* result)
{
    for (uint32_t index = blockDim.x*blockIdx.x + threadIdx.x;
         index < length;
         index += blockDim.x * gridDim.x)
    {
        cuda::std::complex<float> sum(0.0, 0.0);
        for (uint32_t i = 0; i < length; i++)
        {
            cuda::std::complex<float> exponential = cuda::std::expf(
                -(ConvExa::j_cuda * 2.0 * ConvExa::pi * 
                  double(index) * double(i) / double(length))
            );
            sum += exponential;
        }
        result[index] = sum;
    }
}
}