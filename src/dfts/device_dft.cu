#include <convexa.h>
#include <cxkernels.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cuda/std/complex>

namespace CXKernels
{
    __device__ constexpr cuda::std::complex<double> j_d(0.0, 1.0);
    __device__ constexpr cuda::std::complex<float> j_f(0.0, 1.0);
    __device__ constexpr double pi = 3.141592653589793238462643383279;
    __device__ constexpr float pi_f = 3.141592653589793238462643383279f;
}
namespace CXKernels
{
__global__ void device_dft(const double* signal, const uint32_t length, cuda::std::complex<double>* result)
{
    for (uint32_t index = blockDim.x*blockIdx.x + threadIdx.x;
         index < length;
         index += blockDim.x * gridDim.x)
    {
        cuda::std::complex<double> sum(0.0, 0.0);
        for (uint32_t i = 0; i < length; i++)
        {
            cuda::std::complex<double> exponential = cuda::std::exp(
                -(j_d * 2.0 * pi * 
                 static_cast<double>(index) * static_cast<double>(i) / static_cast<double>(length))
            );
            sum += exponential;
        }
        result[index] = sum;
    }
}
}

/*
void ConvExa::device_dft(const double* signal, const uint32_t length, cuda::std::complex<double>* result)
{
    dim3 num_threads = 32;
    dim3 num_blocks = (N + num_threads - 1) / num_threads;

    CXKernels::device_dft<<<num_blocks, num_threads>>>(
        signal, length, result
    );
}
*/

float CXTiming::device_dft(const std::vector<double> &signal, std::vector<std::complex<double>> &result)
{
    double* device_a = nullptr;
    cuda::std::complex<double>* device_c = nullptr;
    uint32_t length = signal.size();

    size_t byte_size_sig = length * sizeof(double);
    size_t byte_size_output = length * sizeof(std::complex<double>);
    result.reserve(length);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    checkCudaErrors(cudaMalloc(&device_a, byte_size_sig));
    checkCudaErrors(cudaMemcpy(device_a, signal.data(), byte_size_sig, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc(&device_c, byte_size_output));

    dim3 num_threads = 1024;
    dim3 num_blocks = (length + num_threads.x - 1) / num_threads.x;
    cudaEventRecord(start);
    // Memory Loaded, Perform Computations...
    CXKernels::device_dft<<<num_blocks, num_threads>>>(
        device_a, length, device_c
    );
    cudaEventRecord(stop);

    // Finish Computations before this block
    checkCudaErrors(cudaMemcpy(result.data(), device_c, byte_size_output, cudaMemcpyDeviceToHost));

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    checkCudaErrors(cudaFree(device_a));
    checkCudaErrors(cudaFree(device_c));

    return milliseconds;
}