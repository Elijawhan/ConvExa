#include <convexa.h>
#include <cxkernels.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cuda/std/complex>

namespace CXKernels
{
    __device__ constexpr cuda::std::complex<double> j_d(0.0, 1.0);
    __device__ constexpr cuda::std::complex<float> j_f(0.0, 1.0);
    __device__ constexpr double pi_d = 3.141592653589793238462643383279;
    __device__ constexpr float pi_f = 3.141592653589793238462643383279f;

template <typename T>
__global__ void device_dft(const T* signal, const size_t length, cuda::std::complex<T>* result)
{
    for (uint32_t index = blockDim.x*blockIdx.x + threadIdx.x;
         index < length;
         index += blockDim.x * gridDim.x)
    {
        cuda::std::complex<T> sum(0.0, 0.0);
        for (uint32_t i = 0; i < length; i++)
        {
            cuda::std::complex<T> exponential = cuda::std::exp(
                -(j_d * 2.0 * pi_d * 
                 static_cast<T>(index) * static_cast<T>(i) / static_cast<T>(length))
            );
            sum += static_cast<cuda::std::complex<T>>(signal[i]) * exponential;
        }
        result[index] = sum;
    }
}
template <>
__global__ void device_dft(const float* signal, const size_t length, cuda::std::complex<float>* result)
{
    for (uint32_t index = blockDim.x*blockIdx.x + threadIdx.x;
         index < length;
         index += blockDim.x * gridDim.x)
    {
        cuda::std::complex<float> sum(0.0, 0.0);
        for (uint32_t i = 0; i < length; i++)
        {
            cuda::std::complex<float> exponential = cuda::std::exp(
                -(j_f * 2.0f * pi_f * 
                 static_cast<float>(index) * static_cast<float>(i) / static_cast<float>(length))
            );
            sum += static_cast<cuda::std::complex<float>>(signal[i]) * exponential;
        }
        result[index] = sum;
    }
}
}

template <typename T>
float CXTiming::device_dft(const std::vector<T> &signal, std::vector<std::complex<T>> &result)
{
    T* device_a = nullptr;
    cuda::std::complex<T>* device_c = nullptr;
    uint32_t length = signal.size();

    size_t byte_size_sig = length * sizeof(T);
    size_t byte_size_output = length * sizeof(std::complex<T>);
    result.resize(length);

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
template float CXTiming::device_dft<double>(const std::vector<double> &signal, std::vector<std::complex<double>> &result);
template float CXTiming::device_dft<float>(const std::vector<float> &signal, std::vector<std::complex<float>> &result);