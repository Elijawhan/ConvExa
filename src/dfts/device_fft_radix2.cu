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
    //if (index == 0) printf("Hello from radix2_preprocess!\n");
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

__global__ void debug_kernel() 
{
    printf("Hello from debug_kernel!\n");
}
}

template <typename T>
float CXTiming::device_fft_radix2(const std::vector<T> &signal, std::vector<std::complex<T>> &result)
{
    T* device_signal = nullptr;
    cuda::std::complex<T>* device_result = nullptr;
    cuda::std::complex<T>* device_temp1 = nullptr;
    uint32_t length = signal.size();
    int fft_size = 1;
    // Choose size of FFT (Nearest Pow 2)
    while (fft_size < length) fft_size <<= 1;

    size_t byte_size_sig_small = length * sizeof(T);
    size_t byte_size_sig = fft_size * sizeof(T);
    size_t byte_size_output = fft_size * sizeof(std::complex<T>);
    result.resize(fft_size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    checkCudaErrors(cudaMalloc(&device_signal, byte_size_sig));
    checkCudaErrors(cudaMemset(device_signal, 0, fft_size * sizeof(T))); // Zero Pad
    checkCudaErrors(cudaMemcpy(device_signal, signal.data(), byte_size_sig_small, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc(&device_result, byte_size_output));
    checkCudaErrors(cudaMalloc(&device_temp1, byte_size_output));

    dim3 num_threads = 1024;
    dim3 num_blocks = (fft_size + num_threads.x - 1) / num_threads.x;

    int num_stages = log2f(fft_size);
    if (fft_size > 1)
    {
        cudaEventRecord(start);
        CXKernels::radix2_preprocess<<<num_blocks, num_threads>>>(
            device_signal, fft_size, device_result
        );
        for (int step = 1; step <= num_stages; step++)
        {
            CXKernels::device_fft_radix2_read<<<num_blocks, num_threads>>>(
                device_result, fft_size, device_temp1, step
            );
            //cudaDeviceSynchronize();
            CXKernels::device_fft_radix2_write<<<num_blocks, num_threads>>>(
                device_result, fft_size, device_temp1, step
            );
            cudaDeviceSynchronize();
        }
        cudaEventRecord(stop);
    }
    cudaEventSynchronize(stop);
    // Finish Computations before this block
    checkCudaErrors(cudaMemcpy(result.data(), device_result, byte_size_output, cudaMemcpyDeviceToHost));

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    checkCudaErrors(cudaFree(device_signal));
    checkCudaErrors(cudaFree(device_result));
    checkCudaErrors(cudaFree(device_temp1));

    T scale = 1.0 / static_cast<T>(fft_size);
    for (int i = 0; i < result.size(); ++i) {
        result[i] *= fft_size;
    }
    return milliseconds;
}
template float CXTiming::device_fft_radix2<double>(const std::vector<double> &signal, std::vector<std::complex<double>> &result);
template float CXTiming::device_fft_radix2<float>(const std::vector<float> &signal, std::vector<std::complex<float>> &result);

template <typename T>
float CXTiming::device_fft_radix2_graphed(const std::vector<T> &signal, std::vector<std::complex<T>> &result)
{
    T* device_signal = nullptr;
    cuda::std::complex<T>* device_result = nullptr;
    cuda::std::complex<T>* device_temp1 = nullptr;
    uint32_t length = signal.size();
    int fft_size = 1;
    // Choose size of FFT (Nearest Pow 2)
    while (fft_size < length) fft_size <<= 1;

    size_t byte_size_sig_small = length * sizeof(T);
    size_t byte_size_sig = fft_size * sizeof(T);
    size_t byte_size_output = fft_size * sizeof(std::complex<T>);
    result.resize(fft_size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    checkCudaErrors(cudaMalloc(&device_signal, byte_size_sig));
    checkCudaErrors(cudaMemset(device_signal, 0, fft_size * sizeof(T))); // Zero Pad
    checkCudaErrors(cudaMemcpy(device_signal, signal.data(), byte_size_sig_small, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc(&device_result, byte_size_output));
    checkCudaErrors(cudaMalloc(&device_temp1, byte_size_output));

    dim3 num_threads = 1024;
    dim3 num_blocks = (fft_size + num_threads.x - 1) / num_threads.x;

    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreate(&stream));
    cudaGraph_t graph;
    checkCudaErrors(cudaGraphCreate(&graph, 0));
    cudaGraphNode_t prev_node = nullptr;

    int num_stages = log2f(fft_size);
    if (fft_size > 1)
    {
        checkCudaErrors(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        CXKernels::radix2_preprocess<<<num_blocks, num_threads, 0, stream>>>(
            device_signal, fft_size, device_result
        );
        for (int step = 1; step <= num_stages; step++)
        {
            CXKernels::device_fft_radix2_read<<<num_blocks, num_threads, 0, stream>>>(
                device_result, fft_size, device_temp1, step
            );
            //cudaDeviceSynchronize();
            CXKernels::device_fft_radix2_write<<<num_blocks, num_threads, 0, stream>>>(
                device_result, fft_size, device_temp1, step
            );
            //cudaStreamSynchronize(stream);
        }
        checkCudaErrors(cudaStreamEndCapture(stream, &graph));
    }

    // Create graph and stream
    cudaGraphExec_t graph_exec;
    cudaGraphNode_t error_node;

    checkCudaErrors(cudaGraphInstantiate(&graph_exec, graph, cudaGraphInstantiateFlagDeviceLaunch));
    cudaEventRecord(start, stream);

    checkCudaErrors(cudaGraphLaunch(graph_exec, stream));

    cudaEventRecord(stop, stream);
    cudaStreamWaitEvent(stream, stop);

    // Finish Computations before this block
    checkCudaErrors(cudaMemcpy(result.data(), device_result, byte_size_output, cudaMemcpyDeviceToHost));

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    checkCudaErrors(cudaFree(device_signal));
    checkCudaErrors(cudaFree(device_result));
    checkCudaErrors(cudaFree(device_temp1));
    checkCudaErrors(cudaGraphDestroy(graph));
    checkCudaErrors(cudaGraphExecDestroy(graph_exec));
    checkCudaErrors(cudaStreamDestroy(stream));

    T scale = 1.0 / static_cast<T>(fft_size);
    for (int i = 0; i < result.size(); ++i) {
        result[i] *= fft_size;
    }
    return milliseconds;
}
template float CXTiming::device_fft_radix2_graphed<double>(const std::vector<double> &signal, std::vector<std::complex<double>> &result);
template float CXTiming::device_fft_radix2_graphed<float>(const std::vector<float> &signal, std::vector<std::complex<float>> &result);