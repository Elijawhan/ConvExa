#include <convexa.h>
#include <cxkernels.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cufft.h>
#include <helper.h>

template <typename T>
float CXTiming::cufft(const std::vector<T> &signal, std::vector<std::complex<T>> &result)
{
    std::vector<std::complex<T>> signal_complex = HELP::vec_cast<T, std::complex<T>>(signal);

    cufftComplex* device_a = nullptr;
    cufftComplex* device_c = nullptr;
    uint32_t length = signal.size();

    size_t byte_size = length * sizeof(cufftComplex);
    result.resize(length);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    checkCudaErrors(cudaMalloc(&device_a, byte_size));
    // C++ standard, I apologize sincerely for this.
    checkCudaErrors(cudaMemcpy(device_a, reinterpret_cast<cufftComplex*>(signal_complex.data()), byte_size, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc(&device_c, byte_size));

    cufftHandle plan;
    cufftPlan1d(&plan, length, CUFFT_C2C, 1);
    cudaEventRecord(start);
    cufftExecC2C(plan, device_a, device_c, CUFFT_FORWARD);
    cudaEventRecord(stop);

    // Finish Computations before this block
    checkCudaErrors(cudaMemcpy(result.data(), reinterpret_cast<std::complex<T>*>(device_c), byte_size, cudaMemcpyDeviceToHost));

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    checkCudaErrors(cudaFree(device_a));
    checkCudaErrors(cudaFree(device_c));
    cufftDestroy(plan);

    T scale = 1.0 / static_cast<T>(length);
    for (int i = 0; i < result.size(); ++i) {
        result[i] *= length;
    }

    return milliseconds;
}
template float CXTiming::cufft<float>(const std::vector<float> &signal, std::vector<std::complex<float>> &result);

template <>
float CXTiming::cufft<double>(const std::vector<double> &signal, std::vector<std::complex<double>> &result)
{
    std::vector<std::complex<double>> signal_complex = HELP::vec_cast<double, std::complex<double>>(signal);

    cufftDoubleComplex* device_a = nullptr;
    cufftDoubleComplex* device_c = nullptr;
    uint32_t length = signal.size();

    size_t byte_size = length * sizeof(cufftDoubleComplex);
    result.resize(length);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    checkCudaErrors(cudaMalloc(&device_a, byte_size));
    // C++ standard, I apologize sincerely for this.
    checkCudaErrors(cudaMemcpy(device_a, reinterpret_cast<cufftDoubleComplex*>(signal_complex.data()), byte_size, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc(&device_c, byte_size));

    cufftHandle plan;
    cufftPlan1d(&plan, length, CUFFT_Z2Z, 1);
    cudaEventRecord(start);
    cufftExecZ2Z(plan, device_a, device_c, CUFFT_FORWARD);
    cudaEventRecord(stop);

    // Finish Computations before this block
    checkCudaErrors(cudaMemcpy(result.data(), reinterpret_cast<std::complex<double>*>(device_c), byte_size, cudaMemcpyDeviceToHost));

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    checkCudaErrors(cudaFree(device_a));
    checkCudaErrors(cudaFree(device_c));
    cufftDestroy(plan);

    double scale = 1.0 / static_cast<double>(length);
    for (int i = 0; i < result.size(); ++i) {
        result[i] *= length;
    }
    
    return milliseconds;
}
