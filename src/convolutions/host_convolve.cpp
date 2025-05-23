#include <convexa.h>

namespace ConvExa
{
    template <typename T>
    std::vector<T> host_convolve(const std::vector<T> &signal, const std::vector<T> &kernel)
    {
        size_t ol = signal.size() + kernel.size() - 1;
        std::vector<T> result;
        result.reserve(ol);

        for (size_t n = 0; n < ol; n++)
        {
            // for each output location
            double sum = 0;
            for (size_t i = 0; i < kernel.size(); i++)
            {
                if (n >= i && (n - i) < signal.size())
                    sum += kernel[i] * signal[n - i];
            }
            result.push_back(sum);
        }
        return result;
    }
    template std::vector<uint16_t> host_convolve(const std::vector<uint16_t> &signal, const std::vector<uint16_t> &kernel);
    template std::vector<int16_t> host_convolve(const std::vector<int16_t> &signal, const std::vector<int16_t> &kernel);
    template std::vector<double> host_convolve(const std::vector<double> &signal, const std::vector<double> &kernel);
    template std::vector<float> host_convolve(const std::vector<float> &signal, const std::vector<float> &kernel);
}
template <typename T>
float CXTiming::host_convolve(const std::vector<T> &signal, const std::vector<T> &kernel, std::vector<T> &output)
{

    auto start = std::chrono::high_resolution_clock::now();
    output = ConvExa::host_convolve(signal, kernel);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    if (duration.count() > 0)
    {
        return duration.count() / 1000.0;
    }
    else
    {
        return 0;
    }
}
template float CXTiming::host_convolve<uint16_t>(const std::vector<uint16_t> &signal, const std::vector<uint16_t> &kernel, std::vector<uint16_t> &output);
template float CXTiming::host_convolve<int16_t>(const std::vector<int16_t> &signal, const std::vector<int16_t> &kernel, std::vector<int16_t> &output);
template float CXTiming::host_convolve<double>(const std::vector<double> &signal, const std::vector<double> &kernel, std::vector<double> &output);
template float CXTiming::host_convolve<float>(const std::vector<float> &signal, const std::vector<float> &kernel, std::vector<float> &output);
