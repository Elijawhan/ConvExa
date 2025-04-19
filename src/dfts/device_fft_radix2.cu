#include <convexa.h>
#include <cxkernels.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

namespace CXKernels
{
template <typename T>
__device__ inline cuda::std::complex<T> twiddle_factor(int32_t k, size_t N) 
{
    T angle = -2.0 * CUDART_PI * k / static_cast<T>(N);
    return cuda::std::complex<T>(cuda::std::cos(angle), cuda::std::sin(angle));
}

template <typename T>
__global__ void device_fft_radix2(const T* signal, const size_t length, cuda::std::complex<T>* result)
{
    uint32_t index = blockDim.x*blockIdx.x + threadIdx.x;

    // Load the real part into the result 
    for (uint32_t i = 0; i < length; i += blockDim.x*gridDim.x)
        result[i] = cuda::std::complex<T>(signal[i], 0);
    __syncthreads();

    // In-place FFT
    for (uint32_t idx = 1; idx <= log2f(static_cast<float>(length)); idx++)
    {
        // Size of the butterfly (2^idx)
        int32_t block = 1 << idx;
        // Number of elements to butterfly
        int32_t half = block >> 1;

        // Grid-stride over the FFT in case it is too large
        for (uint32_t jdx = index; jdx < length; jdx += blockDim.x*gridDim.x)
        {
            uint32_t group = jdx / block;
            uint32_t start = group * block;
            uint32_t rel_pos = jdx & block;

            // Typically, kdx == index, but this may not be true later on, so we calculate it
            uint32_t kdx = start + rel_pos;

            // Only work on half of the elements
            if (rel_pos < half)
            {
                // N = block (size of butterfly group)
                // k = idx % block (index within that group)
                cuda::std::complex<T> twiddle = twiddle_factor<T>(rel_pos, block);

                cuda::std::complex<T> real = result[kdx];
                cuda::std::complex<T> imag = twiddle * result[kdx + half];
                result[kdx] = real + imag;
                result[kdx + half] = real = imag;
            }
        }
        __syncthreads();
    }
}

}

template <typename T>
float CXTiming::device_fft_radix2(const std::vector<T> &signal, std::vector<std::complex<T>> &result)
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
    CXKernels::device_fft_radix2<<<num_blocks, num_threads>>>(
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
template float CXTiming::device_fft_radix2<double>(const std::vector<double> &signal, std::vector<std::complex<double>> &result);
template float CXTiming::device_fft_radix2<float>(const std::vector<float> &signal, std::vector<std::complex<float>> &result);