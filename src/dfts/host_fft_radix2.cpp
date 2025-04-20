#include <convexa.h>

namespace ConvExa
{
uint32_t reverse_bits(uint32_t x)
{
    x = ((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1);
    x = ((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2);
    x = ((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4);
    x = ((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8);
    return (x >> 16) | (x << 16);
}

template <typename T>
std::vector<std::complex<T>> host_fft_radix2(const std::vector<T> &signal)
{
    size_t N = signal.size();
    std::vector<std::complex<T>> result;
    result.resize(N);
    //std::cout << "RESULT SIZE = " << result.size() << std::endl;
    int logN = (int) log2(static_cast<T>(N));

    for (uint32_t i = 0; i < N; i++)
    { 
        uint32_t reverse = reverse_bits(i);
        reverse = reverse >> (32 - logN);
        result[i] = signal[reverse];
    }

    for (int step = 1; step <= logN; step++) 
    {
        int block = 1 << step;
        int half = block >> 1;

        std::complex<T> twiddle = std::exp(std::complex<T>(0,-2.0 * M_PI / static_cast<T>(block)));

        for (uint32_t k = 0; k < N; k += block) 
        {
            std::complex<T> twiddle_factor = 1;

            for (int j = 0; j < half; j++) 
            {
                std::complex<T> real = result[k + j];
                std::complex<T> imag = twiddle_factor * result[k + j + half];
                twiddle_factor *= twiddle;
                result[k + j] = real + imag;
                result[k + j + half] = real - imag;
            }
        }
    }
    return result;
}
}

template <typename T>
float CXTiming::host_fft_radix2(const std::vector<T>& signal, std::vector<std::complex<T>>& output)
{
    auto start = std::chrono::high_resolution_clock::now();

    output = ConvExa::host_fft_radix2(signal);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    if (duration.count() > 0) {
        return duration.count() / 1000.0;
    } else {
        return 0;
    }
}
template float CXTiming::host_fft_radix2<double>(const std::vector<double>& signal, std::vector<std::complex<double>>& output);
template float CXTiming::host_fft_radix2<float>(const std::vector<float>& signal, std::vector<std::complex<float>>& output);