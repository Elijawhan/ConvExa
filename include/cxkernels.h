#pragma once
#include <cuda/std/complex>


namespace ConvExa
{
    template <typename T = double>
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
    float device_convolve(const std::vector<double> &signal, const std::vector<double> &kernel, std::vector<double> &output);
    float device_dft(const std::vector<double> &signal, std::vector<std::complex<double>> &result);
}