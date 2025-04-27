#include <convexa.h>

int nextPow2(size_t num)
{
    int rv = 1;
    while (rv < num)
        (rv << 1);
    return rv;
}
#define ITERS 5
template <typename T>
void threshold_solve()
{
    // TO solve for the threshold, we need to have a signal of size N, and a kernel of size M
    // In general the Traditional Convolutional Kernel time taken should scale with
    // M * (N + M - 1)
    // and the FFT Based Convolution should scale based on
    // l = max(nextPow2(M), nextPow2(N))
    // x * l log_2(l) + l
    // The objective of this function is to solve for x in the above equation by finding a point at which
    // the convolutions take approximately the same quantity of time.

    float t_reg_conv = -1;
    float t_fft_conv = 0;
    int N = 10000;
    int M = 400;
    std::vector<int> NMs;

    // 12 works for up to 8 million
    for (int status = 0; status < 10; status++)
    {
        auto signal = HELP::generate_random_vector<T>(N, -5, 5);
        std::vector<T> result;

        while ((M < N) && t_reg_conv < t_fft_conv)
        {
            M += 1;
            auto kernel = HELP::generate_random_vector<T>(M, -5, 5);

            (void)CXTiming::device_convolve_overlap_save(signal, kernel, result);
            for (int _ = 0; _ < ITERS; _++)
            {
                t_reg_conv += CXTiming::device_convolve_overlap_save(signal, kernel, result);
                result.clear();
            }
            t_reg_conv /= ITERS;

            (void)CXTiming::device_convolve_fft(signal, kernel, result);
            for (int _ = 0; _ < ITERS; _++)
            {
                t_fft_conv += CXTiming::device_convolve_fft(signal, kernel, result);
                result.clear();
            }
            t_fft_conv /= ITERS;

            
            
        }
        printf("Criticality Found: OS: %f, FFT: %f, M: %d, N: %d\n", t_reg_conv, t_fft_conv, M, N);
        M /= 2;
        t_reg_conv = -1;
        t_fft_conv = 0;
        N *= 2;
    }
}