#pragma once
#include <cuda/std/complex>
namespace ConvExa
{
constexpr cuda::std::complex<double> j_d(0.0, 1.0);
constexpr cuda::std::complex<float> j_f(0.0, 1.0);

template< typename T = double>
void device_dft(const T* signal, const uint32_t length, cuda::std::complex<T>* result);

template< typename T = double >
class Convexor
{
    public:
    std::vector<T> m_signal{};
    std::vector<T> m_kernel{};
    std::vector<T> m_result{};

    Convexer() = default;
    Convexer(const Convexer& other) = default;
    Convexer& operator=(const Convexer& other) = default;
    Convexer(Convexer&& other) noexcept = default;
    Convexer& operator=(Convexer&& other) = default;
    ~Convexer() = default;

    private:
    T* d_signal = nullptr;
    T* d_kernel = nullptr;
    T* d_result = nullptr;
}
}
