#include <convexa.h>
#include <cxkernels.h>
#include <cuda_runtime.h>
#include <cuda/std/complex>
namespace CXKernels
{
template< typename T = double >
__global__ void device_dft(const T* signal, const uint32_t length, cuda::std::complex<T>* result)
{
    for (uint32_t index = blockDim.x*blockIdx.x + threadIdx.x;
         index < length;
         index += blockDim.x * gridDim.x)
    {
        cuda::std::complex<T> sum(0.0, 0.0);
        for (uint32_t i = 0; i < length; i++)
        {
            cuda::std::complex<T> exponential = cuda::std::expf(
                -(ConvExa::j_f * 2.0 * ConvExa::pi * 
                 static_cast<T>(index) * static_cast<T>(i) / static_cast<T>(length))
            );
            sum += exponential;
        }
        result[index] = sum;
    }
}
}

namespace ConvExa
{
template< typename T = double >
void device_dft(const T* signal, const uint32_t length, cuda::std::complex<T>* result)
{
    dim3 num_threads = 32;
    dim3 num_blocks = (N + num_threads - 1) / num_threads;

    CXKernels::device_dft<<<num_blocks, num_threads>>>(
        signal, length, result
    );
}
}