#include <convexa.h>
#include <cxkernels.h>
#include "helper.h"
#define numRuns 10
void c_main()
{
    printf("\n******************************************\n");
    printf("Beginning Convolution Test:\n");
    std::vector<int16_t> data;
    HELP::wav_hdr hdr = HELP::read_wav("./cpp/audio/badadeedur.wav", &data);
    std::vector<float> fdata = HELP::vec_cast<int16_t, float>(data);
    std::vector<float> myKernel = {-0.004125211440833717, 0.0027955152777558783, 0.005850215740412419, 0.010502984638973641, 0.016040167214116205, 0.021505672324551077,
        0.025736586323270964, 0.027550780878680053, 0.025978633473695077, 0.020533655427109103, 0.011425562596231889, -0.00032773010346882366, -0.01294515064048871,
        -0.024088742338157703, -0.03121477574019287, -0.03201570748752679, -0.024875240870369457, -0.009253448209804964, 0.014089895495991024, 0.043081367018117746, 0.07457005785512794,
        0.10475610038853216, 0.12978293175046224, 0.14629763790897085, 0.15206954363540107, 0.14629763790897085, 0.12978293175046224, 0.10475610038853216, 0.07457005785512794,
        0.043081367018117746, 0.014089895495991024, -0.009253448209804964, -0.024875240870369457, -0.03201570748752679, -0.03121477574019287, -0.024088742338157703, -0.01294515064048871, -0.00032773010346882366,
        0.011425562596231889, 0.020533655427109103, 0.025978633473695077, 0.027550780878680053, 0.025736586323270964, 0.021505672324551077, 0.016040167214116205, 0.010502984638973641, 0.005850215740412419,
        0.0027955152777558783, -0.004125211440833717};

    std::vector<float> myResult_h;
    float base_convolve_h_timing = CXTiming::host_convolve(fdata, myKernel, myResult_h);
    std::vector<float> myResult_d;
    float base_convolve_d_timing = CXTiming::device_convolve(fdata, myKernel, myResult_d);

    double re = std::get<0>(HELP::relative_error(myResult_d, myResult_h));
    printf("Relative Error device basic: %f ", re);
    if (re < HELP::MAX_RELATIVE_ERROR)
        printf("PASS\n");
    else
        printf("FAIL\n");

    base_convolve_h_timing = 0;
    for (int i = 0; i < numRuns; i++)
    {
        myResult_h.clear();
        base_convolve_h_timing += CXTiming::host_convolve(fdata, myKernel, myResult_h);
    }
    base_convolve_h_timing = base_convolve_h_timing / numRuns;
    base_convolve_d_timing = 0;
    for (int i = 0; i < numRuns; i++)
    {
        myResult_d.clear();
        base_convolve_d_timing += CXTiming::device_convolve(fdata, myKernel, myResult_d);
    }
    base_convolve_d_timing = base_convolve_d_timing / numRuns;

    printf("==== Timing for %d points over %d runs ====\n", data.size() + myKernel.size() - 1, numRuns);
    printf("Host Impl Timing: %f ms\n", base_convolve_h_timing);
    printf("Basic Convolution Kernel Timing: %f ms\n", base_convolve_d_timing);
    HELP::write_wav("./cpp/audio/badadeedu.wav", HELP::vec_cast<float, int16_t>(myResult_d), hdr);

    ////// SECOND ////
    data.clear();
    hdr = HELP::read_wav("./cpp/audio/classic_jam.wav", &data);
    fdata = HELP::vec_cast<int16_t, float>(data);
    base_convolve_h_timing = 0;
    for (int i = 0; i < numRuns; i++)
    {
        myResult_h.clear();
        base_convolve_h_timing += CXTiming::host_convolve(fdata, myKernel, myResult_h);
    }
    base_convolve_h_timing = base_convolve_h_timing / numRuns;
    base_convolve_d_timing = 0;
    for (int i = 0; i < numRuns; i++)
    {
        myResult_d.clear();
        base_convolve_d_timing += CXTiming::device_convolve(fdata, myKernel, myResult_d);
    }
    base_convolve_d_timing = base_convolve_d_timing / numRuns;
    printf("==== Timing for %d points over %d runs ====\n", data.size() + myKernel.size() - 1, numRuns);
    printf("Host Impl Timing: %f ms\n", base_convolve_h_timing);
    printf("Basic Convolution Kernel Timing: %f ms\n", base_convolve_d_timing);
    HELP::write_wav("./cpp/audio/unclassic_jam.wav", HELP::vec_cast<float, int16_t>(myResult_d), hdr);

    ////// SECOND ////
    data.clear();
    hdr = HELP::read_wav("./cpp/audio/subroutines.wav", &data);
    fdata = HELP::vec_cast<int16_t, float>(data);
    base_convolve_h_timing = 0;
    for (int i = 0; i < numRuns; i++)
    {
        myResult_h.clear();
        base_convolve_h_timing += CXTiming::host_convolve(fdata, myKernel, myResult_h);
    }
    base_convolve_h_timing = base_convolve_h_timing / numRuns;
    base_convolve_d_timing = 0;
    for (int i = 0; i < numRuns; i++)
    {
        myResult_d.clear();
        base_convolve_d_timing += CXTiming::device_convolve(fdata, myKernel, myResult_d);
    }
    base_convolve_d_timing = base_convolve_d_timing / numRuns;
    printf("==== Timing for %d points over %d runs ====\n", data.size() + myKernel.size() - 1, numRuns);
    printf("Host Impl Timing: %f ms\n", base_convolve_h_timing);
    printf("Basic Convolution Kernel Timing: %f ms\n", base_convolve_d_timing);
    HELP::write_wav("./cpp/audio/routines.wav", HELP::vec_cast<float, int16_t>(myResult_d), hdr);
}