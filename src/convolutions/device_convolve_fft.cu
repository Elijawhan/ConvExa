#include <convexa.h>
#include <cxkernels.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <cufft.h>
#include <helper.h>

#define BLOCK_LEN 1024
namespace CXKernels
{
    template <typename T = double>
    __global__ void vec_multiply(T *A, T *B, T *C, unsigned int N)
    {
        unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
        for (int n = index; n < N; n += blockDim.x * gridDim.x) // scuttles down the signal
        {
            C[n] = A[n] * B[n];
        }
    }
}
template <typename T = double>
float CXTiming::device_convolve_fft(const std::vector<T> &signal, const std::vector<T> &kernel, std::vector<T> &output)
{
    T *device_a = nullptr;
    T *device_b = nullptr;
    T *device_c = nullptr;
    size_t byte_size_sig = signal.size() * sizeof(T);
    size_t byte_size_kernel = kernel.size() * sizeof(T);
    size_t ol = (signal.size() + kernel.size() - 1);
    size_t byte_size_output = ol * sizeof(T);
    output.resize(ol);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    checkCudaErrors(cudaMalloc(&device_a, byte_size_sig));
    checkCudaErrors(cudaMemcpy(device_a, signal.data(), byte_size_sig, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc(&device_b, byte_size_kernel));
    checkCudaErrors(cudaMemcpy(device_b, kernel.data(), byte_size_kernel, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc(&device_c, byte_size_output));
    // We don't actually need to do any copying, as a matter of fact, that could result in stinky.
    // checkCudaErrors(cudaMemcpy(device_c, C, byte_size, cudaMemcpyHostToDevice));

    dim3 blockSize(BLOCK_LEN);
    if (ol < BLOCK_LEN)
        blockSize.x = ol;
    int blocks = signal.size() / blockSize.x + 1;
    dim3 gridSize(blocks);
    cudaEventRecord(start);
    // Memory Loaded, Perform Computations...
    CXKernels::vec_multiply<T><<<gridSize, blockSize>>>(device_a, device_b, device_c, ol);

    cudaEventRecord(stop);

    // Finish Computations before this block
    checkCudaErrors(cudaMemcpy(output.data(), device_c, byte_size_output, cudaMemcpyDeviceToHost));
    cudaGetLastError();

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    checkCudaErrors(cudaFree(device_a));
    checkCudaErrors(cudaFree(device_b));
    checkCudaErrors(cudaFree(device_c));

    return milliseconds;
}

template float CXTiming::device_convolve_fft<double>(const std::vector<double> &, const std::vector<double> &, std::vector<double> &);
template float CXTiming::device_convolve_fft<float>(const std::vector<float> &, const std::vector<float> &, std::vector<float> &);
// template float CXTiming::device_convolve_overlap_add<uint16_t>(const std::vector<uint16_t>&, const std::vector<uint16_t>&, std::vector<uint16_t>&);
// template float CXTiming::device_convolve_overlap_add<int16_t>(const std::vector<int16_t>&, const std::vector<int16_t>&, std::vector<int16_t>&);