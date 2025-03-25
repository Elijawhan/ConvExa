#include <convexa.h>
#include <cxkernels.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
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
template <typename T= double>
float CXTiming::device_convolve(const std::vector<T> &signal, const std::vector<T> &kernel, std::vector<T> &output)
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


    dim3 blockSize(1024);
    if (ol < 1024) blockSize.x = ol;
    int blocks = signal.size() / blockSize.x + 1;
    dim3 gridSize(blocks);
    cudaEventRecord(start);
    // Memory Loaded, Perform Computations...
    CXKernels::basic_full_convolve<T><<<gridSize, blockSize>>>(device_a, device_b, device_c, signal.size(), kernel.size(), ol );

    cudaEventRecord(stop);

    // Finish Computations before this block
    checkCudaErrors(cudaMemcpy(output.data(), device_c, byte_size_output, cudaMemcpyDeviceToHost));

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    checkCudaErrors(cudaFree(device_a));
    checkCudaErrors(cudaFree(device_b));
    checkCudaErrors(cudaFree(device_c));

    return milliseconds;
}

template float CXTiming::device_convolve<double>(const std::vector<double>&, const std::vector<double>&, std::vector<double>&);
template float CXTiming::device_convolve<uint16_t>(const std::vector<uint16_t>&, const std::vector<uint16_t>&, std::vector<uint16_t>&);
template float CXTiming::device_convolve<int16_t>(const std::vector<int16_t>&, const std::vector<int16_t>&, std::vector<int16_t>&);