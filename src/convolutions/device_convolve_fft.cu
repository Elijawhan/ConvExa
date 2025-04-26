#include <convexa.h>
#include <cxkernels.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <cufft.h>
#include <helper.h>

// Macro for checking cuFFT errors
#define CUFFT_CHECK(call) \
    do { \
        cufftResult err = call; \
        if (err != CUFFT_SUCCESS) { \
            std::cerr << "cuFFT Error: " << err << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


#define BLOCK_LEN 1024
namespace CXKernels
{
    // Kernel for pointwise complex multiplication

    __global__ void vec_multiply_complex_f(cufftComplex *A, cufftComplex *B, cufftComplex *C, unsigned int N)
    {
        unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
        for (int n = index; n < N; n += blockDim.x * gridDim.x) // scuttles down the signal
        {
            float ar = A[n].x;
            float ai = A[n].y;
            float br = B[n].x;
            float bi = B[n].y;
            // Complex multiplication: (ar + ai*i) * (br + bi*i) = (ar*br - ai*bi) + (ar*bi + ai*br)*i
            C[n].x = ar * br - ai * bi;
            C[n].y = ar * bi + ai * br;
        }
    }
    __global__ void vec_multiply_complex_d(cufftDoubleComplex *A, cufftDoubleComplex *B, cufftDoubleComplex *C, unsigned int N)
    {
        unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
        for (int n = index; n < N; n += blockDim.x * gridDim.x) // scuttles down the signal
        {
            float ar = A[n].x;
            float ai = A[n].y;
            float br = B[n].x;
            float bi = B[n].y;
            // Complex multiplication: (ar + ai*i) * (br + bi*i) = (ar*br - ai*bi) + (ar*bi + ai*br)*i
            C[n].x = ar * br - ai * bi;
            C[n].y = ar * bi + ai * br;
        }
    }
}
template <typename T = double>
float CXTiming::device_convolve_fft(const std::vector<T> &signal, const std::vector<T> &kernel, std::vector<T> &output)
{
    int output_size = signal.size() + kernel.size() -1;
    output.resize(output_size);
    int fft_size = 1;
    // Choose size of FFT (Nearest Pow 2)
    while (fft_size < output_size) fft_size <<= 1;
    
    T *d_signal, *d_kernel;
    cufftComplex *d_signalFFT, *d_kernelFFT, *d_productFFT;
    T *d_output;

    // printf("%d size, len: %d\n", fft_size, signal.size());

    checkCudaErrors(cudaMalloc(&d_signal, fft_size * sizeof(T)));
    checkCudaErrors(cudaMalloc(&d_kernel, fft_size * sizeof(T)));
    checkCudaErrors(cudaMalloc(&d_signalFFT, fft_size * sizeof(cufftComplex)));
    checkCudaErrors(cudaMalloc(&d_kernelFFT, fft_size * sizeof(cufftComplex)));
    checkCudaErrors(cudaMalloc(&d_productFFT, fft_size * sizeof(cufftComplex)));
    checkCudaErrors(cudaMalloc(&d_output, fft_size * sizeof(T)));


    checkCudaErrors(cudaMemset(d_signal, 0, fft_size * sizeof(T))); // Zero Pad
    checkCudaErrors(cudaMemcpy(d_signal, signal.data(), signal.size() * sizeof(T), cudaMemcpyHostToDevice));
    
    checkCudaErrors(cudaMemset(d_kernel, 0, fft_size * sizeof(T))); // Zero Pad
    checkCudaErrors(cudaMemcpy(d_kernel, kernel.data(), kernel.size() * sizeof(T), cudaMemcpyHostToDevice));

    // Create cuFFT plans
    cufftHandle planForward, planInverse;
    CUFFT_CHECK(cufftPlan1d(&planForward, fft_size, CUFFT_R2C, 1)); // Real-to-complex
    CUFFT_CHECK(cufftPlan1d(&planInverse, fft_size, CUFFT_C2R, 1)); // Complex-to-real


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    dim3 blockSize(BLOCK_LEN);
    if (fft_size < BLOCK_LEN)
        blockSize.x = fft_size;
    int blocks = signal.size() / blockSize.x + 1;
    dim3 gridSize(blocks);
    cudaEventRecord(start);
    // Memory Loaded, Perform Computations...
    // Compute FFT of input
    
    CUFFT_CHECK(cufftExecR2C(planForward, d_signal, d_signalFFT));
    // Compute FFT of kernel
    CUFFT_CHECK(cufftExecR2C(planForward, d_kernel, d_kernelFFT));

    CXKernels::vec_multiply_complex_f<<<gridSize, blockSize>>>(d_signalFFT, d_kernelFFT, d_productFFT, fft_size);

    CUFFT_CHECK(cufftExecC2R(planInverse, d_productFFT, d_output));
    
    cudaEventRecord(stop);

    cudaMemcpy(output.data(), d_output, output_size * sizeof(T), cudaMemcpyDeviceToHost);

    T scale = 1.0 / static_cast<T>(fft_size);
    for (int i = 0; i < output.size(); ++i) {
        output[i] *= scale;
    }

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    CUFFT_CHECK(cufftDestroy(planForward));
    CUFFT_CHECK(cufftDestroy(planInverse));
    checkCudaErrors(cudaFree(d_signal));
    checkCudaErrors(cudaFree(d_kernel));
    checkCudaErrors(cudaFree(d_signalFFT));
    checkCudaErrors(cudaFree(d_kernelFFT));
    checkCudaErrors(cudaFree(d_productFFT));
    checkCudaErrors(cudaFree(d_output));

    return milliseconds;
}

// template float CXTiming::device_convolve_fft<double>(const std::vector<double> &, const std::vector<double> &, std::vector<double> &);
template float CXTiming::device_convolve_fft<float>(const std::vector<float> &, const std::vector<float> &, std::vector<float> &);
// template float CXTiming::device_convolve_overlap_add<uint16_t>(const std::vector<uint16_t>&, const std::vector<uint16_t>&, std::vector<uint16_t>&);
// template float CXTiming::device_convolve_overlap_add<int16_t>(const std::vector<int16_t>&, const std::vector<int16_t>&, std::vector<int16_t>&);