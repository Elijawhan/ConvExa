#pragma once
#include <cuda/std/complex>
#include <math_constants.h>

namespace ConvExa
{
    template <typename T>
    void device_dft(const T *signal, const uint32_t length, cuda::std::complex<T> *result);
    
    void device_convolve(const double *signal, const double *kernel, double *out, uint32_t sN, uint32_t kN, uint32_t oN);

    template <typename T = double>
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
    float device_dft(const std::vector<T> &signal, std::vector<std::complex<T>> &result);
    template <typename T>
    float device_fft_radix2(const std::vector<T> &signal, std::vector<std::complex<T>> &result);
}

namespace CXKernels
{
// Device-side constants used in DFT
// Try moving to constant memory?
__device__ constexpr cuda::std::complex<double> j_d(0.0, 1.0);
__device__ constexpr cuda::std::complex<float> j_f(0.0, 1.0);

template <typename T>
__global__ inline cuda::std::complex<T> twiddle_factor(int32_t k, size_t N) 
{
    T angle = -2.0 * CUDART_PI * k / static_cast<T>(N);
    return cuda::std::complex<T>(cos(angle), sin(angle));
}
}