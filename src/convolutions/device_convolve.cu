#include <convexa.h>

namespace CXKernels {
    __global__ void basic_full_convolve(float*A, float*B, float*C, unsigned int aN, unsigned int bN, unsigned int cN) {
        unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
        for (int n = index; n < cN; n += blockDim.x * gridDim.x) // scuttles down the signal
        {
            float sum = 0.0f;
            // Convert the pixel
            for (int i = 0; i < bN; i += 1) // scuttles down the kernel
            {
                if (n >= i && (n-i) < aN)
                sum += B[i] * A[n - i];
            }
            C[n] = sum;
        }
    }
}