#include <convexa.h>
#include <cuda_runtime.h>
#include <cuda/std/complex>
namespace CXKernels
{
    template <typename T = double>
    __global__ void basic_full_convolve(T *A, T *B, T *C, unsigned int aN, unsigned int bN, unsigned int cN)
    {
        unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
        for (int n = index; n < cN; n += blockDim.x * gridDim.x) // scuttles down the signal
        {
            T sum = 0.0;
            // Convert the pixel
            for (int i = 0; i < bN; i += 1) // scuttles down the kernel
            {
                if (n >= i && (n - i) < aN)
                    sum += B[i] * A[n - i];
            }
            C[n] = sum;
        }
    }
}
double CXTiming::device_convolve(const std::vector<double> &signal, const std::vector<double> &kernel, std::vector<double> &output)
{
}