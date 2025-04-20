#include <convexa.h>
#include <cxkernels.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <iostream>
#include <helper.h>
namespace CXKernels
{
template <typename T>
__device__ inline cuda::std::complex<T> twiddle_factor(int32_t k, int32_t N) 
{
    T angle = -2.0 * CUDART_PI * k / static_cast<T>(N);
    T s, c;
    sincos(angle, &s, &c);
    return cuda::std::complex<T>(c, s);
}

__device__ inline uint32_t reverse_bits(uint32_t x)
{
    x = ((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1);
    x = ((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2);
    x = ((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4);
    x = ((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8);
    return (x >> 16) | (x << 16);/**/
}

template <typename T>
__global__ void radix2_preprocess(const T* signal, const size_t length, cuda::std::complex<T>* result)
{
    uint32_t index = blockDim.x*blockIdx.x + threadIdx.x;
    uint32_t reverse;
    uint32_t logN = log2f(static_cast<float>(length));
    for (int i = index; i < length; i += blockDim.x*gridDim.x)
    {
        reverse = reverse_bits(i);
        reverse >>= (32 - logN);
        result[index] = cuda::std::complex<T>(signal[reverse], 0);
    }
}


template <typename T>
__global__ void device_fft_radix2_read(cuda::std::complex<T>* result, const size_t length, cuda::std::complex<T>* temp_result, uint32_t step)
{
    uint32_t index = blockDim.x*blockIdx.x + threadIdx.x;
    // Size of the butterfly (2^idx)
    uint32_t block = 1 << step;
    // Number of elements to butterfly
    uint32_t half = block >> 1;
    uint32_t group = index / half;
    uint32_t start = group * block;
    uint32_t rel_pos = index % half;

    // Typically, kdx == index, but this may not be true later on, so we calculate it
    uint32_t kdx = start + rel_pos;
    if (kdx < length)
    {
        // N = block (size of butterfly group)
        // k = idx % block (index within that group)
        cuda::std::complex<T> twiddle = twiddle_factor<T>(rel_pos, block);
        temp_result[kdx] = result[kdx];
        temp_result[kdx + half] = twiddle * result[kdx + half];
    }
}
template <typename T>
__global__ void device_fft_radix2_write(cuda::std::complex<T>* result, const size_t length, cuda::std::complex<T>* temp_result, uint32_t step)
{
    uint32_t index = blockDim.x*blockIdx.x + threadIdx.x;
    // Size of the butterfly (2^idx)
    uint32_t block = 1 << step;
    // Number of elements to butterfly
    uint32_t half = block >> 1;
    uint32_t group = index / half;
    uint32_t start = group * block;
    uint32_t rel_pos = index % half;

    // Typically, kdx == index, but this may not be true later on, so we calculate it
    uint32_t kdx = start + rel_pos;
    if (kdx < length)
    {
        result[kdx] = temp_result[kdx] + temp_result[kdx + half];
        result[kdx + half] = temp_result[kdx] - temp_result[kdx + half];
    }
}
}

template <typename T>
float CXTiming::device_fft_radix2(const std::vector<T> &signal, std::vector<std::complex<T>> &result)
{
    T* device_a = nullptr;
    cuda::std::complex<T>* device_c = nullptr;
    cuda::std::complex<T>* device_temp1 = nullptr;
    cuda::std::complex<T>* device_temp2 = nullptr;
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
    checkCudaErrors(cudaMalloc(&device_temp1, byte_size_output));
    checkCudaErrors(cudaMalloc(&device_temp2, byte_size_output));

    dim3 num_threads = 1024;
    dim3 num_blocks = (length + num_threads.x - 1) / num_threads.x;

    int num_stages = log2f(length);
    
    if (length > 1)
    {
        cudaEventRecord(start);
        CXKernels::radix2_preprocess<<<num_blocks, num_threads>>>(
            device_a, length, device_c
        );
        for (int step = 1; step <= num_stages; step++)
        {
            CXKernels::device_fft_radix2_read<<<num_blocks, num_threads>>>(
                device_c, length, device_temp1, step
            );
            //cudaDeviceSynchronize();
            CXKernels::device_fft_radix2_write<<<num_blocks, num_threads>>>(
                device_c, length, device_temp1, step
            );
            cudaDeviceSynchronize();
        }
        cudaEventRecord(stop);
    }
    cudaEventSynchronize(stop);
    // Finish Computations before this block
    checkCudaErrors(cudaMemcpy(result.data(), device_c, byte_size_output, cudaMemcpyDeviceToHost));

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    checkCudaErrors(cudaFree(device_a));
    checkCudaErrors(cudaFree(device_c));
    checkCudaErrors(cudaFree(device_temp1));
    checkCudaErrors(cudaFree(device_temp2));
    return milliseconds;
}
template float CXTiming::device_fft_radix2<double>(const std::vector<double> &signal, std::vector<std::complex<double>> &result);
template float CXTiming::device_fft_radix2<float>(const std::vector<float> &signal, std::vector<std::complex<float>> &result);