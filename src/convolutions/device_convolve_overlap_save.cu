#include <convexa.h>
#include <cxkernels.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#define BLOCK_LEN 1024
namespace CXKernels
{
    template <typename T = double>
    __global__ void overlap_save_full_convolve(T *A, T *B, T *C, unsigned int aN, unsigned int bN, unsigned int cN)
    {
        extern __shared__ __align__(sizeof(T)) unsigned char my_smem[];
        T *s_sig = reinterpret_cast<T *>(my_smem);
        unsigned int w_offset = bN / 2;
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
template <typename T = double>
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