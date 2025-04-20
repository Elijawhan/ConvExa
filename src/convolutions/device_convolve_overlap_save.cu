#include <convexa.h>
#include <cxkernels.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#define BLOCK_LEN 1024
namespace CXKernels
{
    template <typename T>
    __global__ void overlap_save_full_convolve(T *A, T *B, T *C, unsigned int aN, unsigned int bN, unsigned int cN)
    {
        extern __shared__ __align__(sizeof(T)) unsigned char my_smem[];
        T *s_sig = reinterpret_cast<T *>(my_smem);
        unsigned int sig_size = bN + BLOCK_LEN ;
        // printf("%d Size!", sig_size);

        unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
        for (int n = index; n < ((cN / BLOCK_LEN) + 1) * BLOCK_LEN; n += blockDim.x * gridDim.x) // scuttles down the signal
        {
            for (int sig_i = threadIdx.x; sig_i < sig_size ; sig_i+= blockDim.x) {
                int sig_index =  n - bN + sig_i - threadIdx.x;
                if (sig_index >= 0 && sig_index < aN) { s_sig[sig_i] = A[sig_index];}
                else s_sig[sig_i] = 0;
            }
            __syncthreads();

            if (n < cN)
            {
                T sum = 0.0;
                // Convert the pixel
                for (int i = 0; i < bN; i += 1) // scuttles down the kernel
                {
                    sum += B[ i] * s_sig[bN   +threadIdx.x - i];//* A[n - i];
                }
                C[n] = sum;
            }
        }
    }
}
template <typename T>
float CXTiming::device_convolve_overlap_save(const std::vector<T> &signal, const std::vector<T> &kernel, std::vector<T> &output)
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
    size_t shmem = (BLOCK_LEN + kernel.size() )* sizeof(T);
    cudaEventRecord(start);
    // Memory Loaded, Perform Computations...
    CXKernels::overlap_save_full_convolve<T><<<gridSize, blockSize, shmem>>>(device_a, device_b, device_c, signal.size(), kernel.size(), ol);

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

template float CXTiming::device_convolve_overlap_save<double>(const std::vector<double> &, const std::vector<double> &, std::vector<double> &);
template float CXTiming::device_convolve_overlap_save<float>(const std::vector<float> &, const std::vector<float> &, std::vector<float> &);
// template float CXTiming::device_convolve_overlap_add<uint16_t>(const std::vector<uint16_t>&, const std::vector<uint16_t>&, std::vector<uint16_t>&);
// template float CXTiming::device_convolve_overlap_add<int16_t>(const std::vector<int16_t>&, const std::vector<int16_t>&, std::vector<int16_t>&);

template <typename T>
std::vector<std::vector<T>> ConvExa::batch_convolution(const std::vector<std::vector<T>> &signals, const std::vector<std::vector<T>> &kernels)
{
    std::vector<std::vector<T>> results;
    std::vector<T*> device_signals;
    std::vector<T*> device_kernels;
    std::vector<T*> device_results;
    std::vector<cudaStream_t> streams;

    T* d_signal = nullptr,* d_kernel = nullptr,* d_result = nullptr;
    // Load the memory
    for (uint32_t idx = 0; idx < signals.size(); idx++)
    {
        d_signal = nullptr; d_kernel = nullptr; d_result = nullptr;
        
        uint32_t sig_length = signals[idx].size();
        uint32_t sig_size = sig_length * sizeof(T);
        uint32_t ker_length = kernels[idx].size();
        uint32_t ker_size = ker_length * sizeof(T);
        uint32_t result_size = (sig_length + ker_length - 1) * sizeof(T);

        checkCudaErrors(cudaMalloc(&d_signal, sig_size));
        checkCudaErrors(cudaMalloc(&d_kernel, ker_size));
        checkCudaErrors(cudaMalloc(&d_result, result_size));

        cudaStream_t stream{0};
        checkCudaErrors(cudaStreamCreate(&stream));
        checkCudaErrors(cudaMemcpyAsync(d_signal, signals[idx].data(), sig_size,
                        cudaMemcpyHostToDevice, stream));
        checkCudaErrors(cudaMemcpyAsync(d_kernel, kernels[idx].data(), ker_size,
                        cudaMemcpyHostToDevice, stream));

        streams.push_back(stream);
        device_signals.push_back(d_signal);
        device_kernels.push_back(d_kernel);
        device_results.push_back(d_result);
    }

    for (uint32_t idx = 0; idx < signals.size(); idx++)
    {
        uint32_t sig_length = signals[idx].size();
        uint32_t ker_length = kernels[idx].size();
        uint32_t result_length = (sig_length + ker_length - 1);

        dim3 blockSize(1024);
        int blocks = sig_length / blockSize.x + 1;
        dim3 gridSize(blocks);
        size_t shmem = (1024 + ker_length) * sizeof(T);
        // Wait for memcpy to complete
        checkCudaErrors(cudaStreamSynchronize(streams[idx]));

        /* Fire off the Convolution */
        CXKernels::overlap_save_full_convolve<T> <<<gridSize, blockSize, shmem, streams[idx]>>> (
            device_signals[idx], device_kernels[idx], device_results[idx], 
            sig_length, ker_length, result_length
        );
    }

    for (uint32_t idx = 0; idx < signals.size(); idx++)
    {
        uint32_t sig_length = signals[idx].size();
        uint32_t ker_length = kernels[idx].size();
        uint32_t result_length = (sig_length + ker_length - 1);
        uint32_t result_size = result_length * sizeof(T);
        
        std::vector<T> result; 
        result.resize(result_length);
        results.push_back(result);
        // Wait for convolution to complete
        checkCudaErrors(cudaStreamSynchronize(streams[idx]));
        checkCudaErrors(cudaMemcpyAsync(results[idx].data(), device_results[idx], result_size,
                        cudaMemcpyDeviceToHost, streams[idx]));

        // Memory copy is complete
        checkCudaErrors(cudaStreamSynchronize(streams[idx]));
    }
    checkCudaErrors(cudaDeviceSynchronize());
    return results;
}
template std::vector<std::vector<float>> ConvExa::batch_convolution(const std::vector<std::vector<float>> &signals, const std::vector<std::vector<float>> &kernels);