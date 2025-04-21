#include <convexa.h>
#include <cxkernels.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cufft.h>
typedef enum CX_Conv_Type {OVERLAP_ADD, FFT_BASED} CXConv_t;

#define DEBUG false

template <typename T>
struct conv_ptrs
{
    cudaStream_t stream;
    CXConv_t conv_type;
    std::vector<T*> device_ptrs;
    std::vector<cufftComplex*> device_fft_ptrs;
    std::vector<cufftHandle> plans;
};
template <>
struct conv_ptrs<double>
{
    cudaStream_t stream;
    CXConv_t conv_type;
    std::vector<double*> device_ptrs;
    std::vector<cufftDoubleComplex*> device_fft_ptrs;
    std::vector<cufftHandle> plans;
};

template< typename T >
conv_ptrs<T> choose_convolution(const std::vector<T> &signal, const std::vector<T> &kernel)
{
    size_t signal_length = signal.size();
    size_t kernel_length = kernel.size();
    size_t classic_bigO = kernel_length * (kernel_length + signal_length - 1);
    size_t fftconv_bigO = 3 * signal_length * log2(signal_length) + signal_length;

    conv_ptrs<T> group;
    cudaStream_t stream{0};
    checkCudaErrors(cudaStreamCreate(&stream));
    group.stream = stream;

    T* d_signal, *d_kernel, *d_result;

    if (classic_bigO < fftconv_bigO)
    {
        group.conv_type = OVERLAP_ADD;
        
        checkCudaErrors(cudaMalloc(&d_signal, signal_length * sizeof(T)));
        checkCudaErrors(cudaMalloc(&d_kernel, kernel_length * sizeof(T)));
        checkCudaErrors(cudaMalloc(&d_result, (signal_length + kernel_length - 1) * sizeof(T)));

        checkCudaErrors(cudaMemcpyAsync(d_signal, signal.data(), signal_length * sizeof(T),
                        cudaMemcpyHostToDevice, group.stream));
        checkCudaErrors(cudaMemcpyAsync(d_kernel, kernel.data(), kernel_length * sizeof(T),
                        cudaMemcpyHostToDevice, group.stream));

        group.device_ptrs.push_back(d_signal);
        group.device_ptrs.push_back(d_kernel);
        group.device_ptrs.push_back(d_result);
    }
    else
    {
        group.conv_type = FFT_BASED;
        
        cufftComplex* d_signal_fft, *d_kernel_fft, *d_result_fft;
        uint32_t fft_size = 1;
        while (fft_size < (signal_length + kernel_length - 1)) fft_size <<= 1;
        
        checkCudaErrors(cudaMalloc(&d_signal, fft_size * sizeof(T)));
        checkCudaErrors(cudaMalloc(&d_kernel, fft_size * sizeof(T)));
        checkCudaErrors(cudaMalloc(&d_result, fft_size * sizeof(T)));

        checkCudaErrors(cudaMemset(d_signal, 0, fft_size * sizeof(T)));
        checkCudaErrors(cudaMemset(d_kernel, 0, fft_size * sizeof(T)));

        checkCudaErrors(cudaMemcpyAsync(d_signal, signal.data(), signal_length * sizeof(T),
                        cudaMemcpyHostToDevice, group.stream));
        checkCudaErrors(cudaMemcpyAsync(d_kernel, kernel.data(), kernel_length * sizeof(T),
                        cudaMemcpyHostToDevice, group.stream));


        group.device_ptrs.push_back(d_signal);
        group.device_ptrs.push_back(d_kernel);
        group.device_ptrs.push_back(d_result);

        checkCudaErrors(cudaMalloc(&d_signal_fft, fft_size * sizeof(cufftComplex)));
        checkCudaErrors(cudaMalloc(&d_kernel_fft, fft_size * sizeof(cufftComplex)));
        checkCudaErrors(cudaMalloc(&d_result_fft, fft_size * sizeof(cufftComplex)));
        group.device_fft_ptrs.push_back(d_signal_fft);
        group.device_fft_ptrs.push_back(d_kernel_fft);
        group.device_fft_ptrs.push_back(d_result_fft);

        cufftHandle planForward, planInverse;
        cufftPlan1d(&planForward, fft_size, CUFFT_R2C, 1);
        cufftPlan1d(&planInverse, fft_size, CUFFT_C2R, 1);
        group.plans.push_back(planForward);
        group.plans.push_back(planInverse);
    }

    return group;
}

template <typename T>
void launch_convolution(conv_ptrs<T> group, size_t signal_length, size_t kernel_length)
{
    
    checkCudaErrors(cudaStreamSynchronize(group.stream));

    uint32_t result_length = (signal_length + kernel_length - 1);

    if (group.conv_type == OVERLAP_ADD)
    {
        dim3 blockSize(1024);
        int blocks = signal_length / blockSize.x + 1;
        dim3 gridSize(blocks);
        size_t shmem = (1024 + kernel_length) * sizeof(T);
        CXKernels::overlap_save_full_convolve<T> <<< gridSize, blockSize, shmem, group.stream >>> (
            group.device_ptrs[0], group.device_ptrs[1], group.device_ptrs[2], 
            signal_length, kernel_length, result_length
        );

    } else if (group.conv_type == FFT_BASED) {

        dim3 blockSize(1024);
        uint32_t fft_size = 1;
        while (fft_size < result_length) fft_size <<= 1;
        if (fft_size < 1024)
            blockSize.x = fft_size;
        int blocks = signal_length / blockSize.x + 1;
        dim3 gridSize(blocks);
        size_t shmem = (1024 + kernel_length) * sizeof(T);

        cufftExecR2C(group.plans[0], group.device_ptrs[0], group.device_fft_ptrs[0]);
        cufftExecR2C(group.plans[0], group.device_ptrs[1], group.device_fft_ptrs[1]);

        CXKernels::vec_multiply_complex_f <<< gridSize, blockSize, 0, group.stream >>> (
            group.device_fft_ptrs[0], group.device_fft_ptrs[1], group.device_fft_ptrs[2], fft_size
        );

        cufftExecC2R(group.plans[1], group.device_fft_ptrs[2], group.device_ptrs[2]);
    }
}

template <typename T>
std::vector<std::vector<T>> ConvExa::batch_convolve(const std::vector<std::vector<T>> &signals, const std::vector<std::vector<T>> &kernels)
{
    std::vector<std::vector<T>> results;
    std::vector<conv_ptrs<T>> workspace;

    // Load the memory
    for (uint32_t idx = 0; idx < signals.size(); idx++)
    {
        conv_ptrs<T> group = choose_convolution(signals[idx], kernels[idx]);
        workspace.push_back(group);
    }
    if (DEBUG) {
        for (conv_ptrs<T> group: workspace) {
            printf("Call Type: %d \n", group.conv_type);
            // printf("", )
        }
    }

    for (uint32_t idx = 0; idx < signals.size(); idx++)
    {
        launch_convolution(workspace[idx], signals[idx].size(), kernels[idx].size());
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
        checkCudaErrors(cudaStreamSynchronize(workspace[idx].stream));
        checkCudaErrors(cudaMemcpyAsync(results[idx].data(), workspace[idx].device_ptrs[2], result_size,
                        cudaMemcpyDeviceToHost, workspace[idx].stream));

        if (workspace[idx].conv_type == FFT_BASED) {
            uint32_t fft_size = 1;
            while (fft_size < result_length) fft_size <<= 1;
            T scale = 1.0 / static_cast<T>(fft_size);
            for (int i = 0; i < results[idx].size(); i ++) {
                results[idx][i]  *= scale;
            }
        }

        // Memory copy is complete
        checkCudaErrors(cudaStreamSynchronize(workspace[idx].stream));
    }
    for (auto &group : workspace)
    {
        for (auto &ptr : group.device_ptrs)
        {
            checkCudaErrors(cudaFree(ptr));
        }
        for (auto &ptr : group.device_fft_ptrs)
        {
            checkCudaErrors(cudaFree(ptr));
        }
        for (auto &plan : group.plans)
        {
            cufftDestroy(plan);
        }
    }

    checkCudaErrors(cudaDeviceSynchronize());
    return results;
}
template std::vector<std::vector<float>> ConvExa::batch_convolve(const std::vector<std::vector<float>> &signals, const std::vector<std::vector<float>> &kernels);