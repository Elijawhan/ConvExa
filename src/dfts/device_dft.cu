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
template __global__ void device_dft<float>(const float* signal, const uint32_t length, cuda::std::complex<float>* result);
template __global__ void device_dft<double>(const double* signal, const uint32_t length, cuda::std::complex<double>* result);
}

template< typename T = double >
void ConvExa::device_dft(const T* signal, const uint32_t length, cuda::std::complex<T>* result)
{
    dim3 num_threads = 32;
    dim3 num_blocks = (N + num_threads - 1) / num_threads;

    CXKernels::device_dft<<<num_blocks, num_threads>>>(
        signal, length, result
    );
}

template< typename T = double >
void CXTiming::device_dft(const std::vector<T> &signal, std::vector<std::complex<T>> result)
{
    float *device_a = nullptr;
    float *device_c = nullptr;

    size_t byte_size_sig = signal.size() * sizeof(T);
    size_t byte_size_output = byte_size_sig;
    result.reserve(signal.size());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    checkCudaErrors(cudaMalloc(&device_a, byte_size_sig));
    checkCudaErrors(cudaMemcpy(device_a, signal.data(), byte_size_sig, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc(&device_c, byte_size_output));

    dim3 num_threads = 1024;
    dim3 num_blocks = (N + num_threads.x - 1) / num_threads.x;
    cudaEventRecord(start);
    // Memory Loaded, Perform Computations...
    CXKernels::device_dft<<<num_blocks, num_threads>>>(
        signal, length, result
    );
    cudaEventRecord(stop);

    // Finish Computations before this block
    checkCudaErrors(cudaMemcpy(output.data(), device_c, byte_size_output, cudaMemcpyDeviceToHost));

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    checkCudaErrors(cudaFree(device_a));
    checkCudaErrors(cudaFree(device_c));

    return milliseconds;
}

template void CXTiming::device_dft(const std::vector<float> &signal, std::vector<std::complex<float>> result);
template void CXTiming::device_dft(const std::vector<double> &signal, std::vector<std::complex<double>> result);