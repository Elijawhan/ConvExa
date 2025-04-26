#pragma once
#include <cuda/std/complex>
#include <math_constants.h>
#include <vector>
#include <cuda_runtime.h>
#include <cufft.h>
namespace ConvExa
{
    template <typename T>
    void device_dft(const T *signal, const uint32_t length, cuda::std::complex<T> *result);
    template <typename T>
    float device_fft_radix2(const std::vector<T> &signal, std::vector<std::complex<T>> &result);

    void device_convolve(const double *signal, const double *kernel, double *out, uint32_t sN, uint32_t kN, uint32_t oN);


    template <typename T>
    class Convexor
    {
    public:
        std::vector<T> m_signal{};
        std::vector<T> m_kernel{};
        std::vector<T> m_result{};

        Convexor() = default;
        Convexor(const Convexor &other) = default;
        Convexor &operator=(const Convexor &other) = default;
        Convexor(Convexor &&other) noexcept = default;
        Convexor &operator=(Convexor &&other) = default;
        ~Convexor() = default;

    private:
        T *d_signal = nullptr;
        T *d_kernel = nullptr;
        T *d_result = nullptr;
    };
}
namespace CXTiming
{
    template <typename T>
    float device_convolve(const std::vector<T> &signal, const std::vector<T> &kernel, std::vector<T> &output);
    template <typename T>
    float device_convolve_overlap_save(const std::vector<T> &signal, const std::vector<T> &kernel, std::vector<T> &output);
    template <typename T>
    float device_convolve_fft(const std::vector<T> &signal, const std::vector<T> &kernel, std::vector<T> &output);
    template <typename T>
    float device_dft(const std::vector<T> &signal, std::vector<std::complex<T>> &result);
    template <typename T>
    float device_fft_radix2(const std::vector<T> &signal, std::vector<std::complex<T>> &result);
    template <typename T>
    float cufft(const std::vector<T> &signal, std::vector<std::complex<T>> &result);
}

namespace CXKernels
{
template <typename T>
__global__ void overlap_save_full_convolve(T *A, T *B, T *C, unsigned int aN, unsigned int bN, unsigned int cN);
__global__ void vec_multiply_complex_f(cufftComplex *A, cufftComplex *B, cufftComplex *C, unsigned int N);
__global__ void vec_multiply_complex_d(cufftDoubleComplex *A, cufftDoubleComplex *B, cufftDoubleComplex *C, unsigned int N);
}