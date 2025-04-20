#pragma once

#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <complex>
#include <cstdlib>
#include <type_traits>
#include <tuple>
#define numRuns 10

namespace HELP
{
    // WAV header information.
    // Pulled from https://stackoverflow.com/questions/13660777/c-reading-the-data-part-of-a-wav-file
    typedef struct WAV_HEADER
    {
        /* RIFF Chunk Descriptor */
        uint8_t RIFF[4];    // RIFF Header Magic header
        uint32_t ChunkSize; // RIFF Chunk Size
        uint8_t WAVE[4];    // WAVE Header
        /* "fmt" sub-chunk */
        uint8_t fmt[4];         // FMT header
        uint32_t Subchunk1Size; // Size of the fmt chunk
        uint16_t AudioFormat;   // Audio format 1=PCM,6=mulaw,7=alaw,     257=IBM Mu-Law, 258=IBM A-Law, 259=ADPCM
        uint16_t NumOfChan;     // Number of channels 1=Mono 2=Sterio
        uint32_t SamplesPerSec; // Sampling Frequency in Hz
        uint32_t bytesPerSec;   // bytes per second
        uint16_t blockAlign;    // 2=16-bit mono, 4=16-bit stereo
        uint16_t bitsPerSample; // Number of bits per sample
        /* "data" sub-chunk */
        uint8_t Subchunk2ID[4]; // "data"  string
        uint32_t Subchunk2Size; // Sampled data length
    } wav_hdr;

    inline int tokenize_csv(
        char *fname,
        std::vector<float> &matrix) // Must pass in vectors so that the memory stays alive.
    {
        std::ifstream in_file;
        in_file.open(fname);

        std::string line, token;
        int row_count = 0, cell_count = 0;

        if (!in_file.is_open())
        {
            return -2;
        }
        row_count = 0;
        while (std::getline(in_file, line))
        {
            row_count++;
            std::stringstream line_stream(line);
            std::vector<float> row;
            while (std::getline(line_stream, token, ','))
            {
                cell_count++;
                if (token != "," && token != "\n")
                    row.push_back(std::stof(token));
                else
                    std::cout << "BAD TOKEN\n";
            }
            matrix.insert(matrix.end(), row.begin(), row.end());
        }
        in_file.close();
        if (row_count)
            return row_count;
        else
            return 1;
    }
    template <typename T = double>
    inline void print_vec(std::vector<T> vec)
    {

        for (T i : vec)
        {
            if (std::is_floating_point<T>())
                printf("%f ", i);
            else
                printf("%d ", i);
        }
        printf("\n");
    }
    template <typename T = std::complex<double>>
    inline void print_vec_complex(std::vector<T> vec)
    {
        for (T i : vec)
        {
            printf("(%.10f, %.10f) ", i.real(), i.imag());
        }
        printf("\n");
    }

    inline wav_hdr read_wav(const char * fin, std::vector<int16_t>* out) {
        FILE* input;
        wav_hdr hdr;
        try {
        
            input = fopen(fin, "r");
            if (input == nullptr) {
                std::cout << "Input File '" << fin << "' Could not be opened\n";
                return hdr;
            }
            
        } catch (...) {
            std::cout << "Could Not Open files \n";
            return hdr;
        }

        size_t numThingsRead;
        int recordLength = 0;
        numThingsRead = fread(&hdr, sizeof hdr , 1, input);
        uint16_t bytesPerSample = hdr.bitsPerSample / 8;      //Number     of bytes per sample
        uint64_t numSamples = hdr.ChunkSize / bytesPerSample; //How many samples are in the wav file?
        out->reserve(numSamples); // slight optimization
        static const uint16_t BUFFER_SIZE = 4096 /bytesPerSample; 
        int16_t* buffer = new int16_t[BUFFER_SIZE];
        while ((numThingsRead = fread(buffer, sizeof buffer[0], BUFFER_SIZE / (sizeof buffer[0]), input)) > 0)
        {
            /** DO SOMETHING WITH THE WAVE DATA HERE **/
            // cout << "Read " << numThingsRead << " points." << endl;
            // Output some of Data Read here

            for (int i = 0; i < numThingsRead; i++) {
                int16_t nv = buffer[i];
                out->push_back(nv);
            }
            // Increment Time of Recording
            recordLength += numThingsRead;
        }
        return hdr;
    }
    inline void write_wav(const char * fout,const  std::vector<int16_t>& in, wav_hdr wav_details) {
        FILE* output;
        try {
        
            output = fopen(fout, "w+");
            if (output == nullptr) {
                std::cout << "Output File '" << fout << "' Could not be opened\n";
                return ;
            }
            
        } catch (...) {
            std::cout << "Could Not Open files \n";
            return ;
        }
        fwrite(&wav_details, sizeof wav_details, 1, output);
        fwrite(in.data(), sizeof(int16_t), in.size(), output);
        
    }

    constexpr long double MAX_RELATIVE_ERROR_FLOAT = 1e-5;
    constexpr long double MAX_RELATIVE_ERROR_DOUBLE = 1e-14;
    // MAKE SURE TO CHANGE BOTH TEMPLATES
    template <typename T>
    inline long double relative_error(std::vector<T> computed_vec, std::vector<T> reference_vec)
    {
            // calculate relative error
    // double errorNorm = 0.0;
    // double referenceNorm = 0.0;
    // for (size_t idx = 0; idx < myResult_h.size(); ++idx)
    // {
    //     auto const &reference = myResult_h[idx];
    //     auto const &computed = myResult_d[idx];
    //     double localError = reference - computed;
    //     errorNorm += localError * localError;
    //     referenceNorm += reference * reference;
    // }
    // double re  = std::sqrt(errorNorm / referenceNorm);

    // WISE'S WISDOM ^
        //printf("Size of long double: %d-bit\n", sizeof(long double) * 8);
        size_t length = computed_vec.size();
        if (length != reference_vec.size())
        {
            throw std::runtime_error("Relative error inputs must be the same size.");
        } 
        long double error_norm = 0.0;
        long double reference_norm = 0.0;
        for (size_t idx = 0; idx < length; ++idx)
        {
            const long double  &reference = reference_vec[idx];
            const long double  &computed = computed_vec[idx];
            long double localError = reference - computed;
            error_norm += localError * localError;
            reference_norm += reference * reference;
        }
        long double rel_error  = std::sqrt(error_norm / reference_norm);
        return rel_error;
    }

    template <typename T>
    inline long double relative_error(std::vector<std::complex<T>> computed_vec, std::vector<std::complex<T>> reference_vec)
    {
        //printf("Size of long double: %d-bit\n", sizeof(long double) * 8);
        size_t length = computed_vec.size();
        if (length != reference_vec.size())
        {
            std::string error_msg = "Relative error inputs must be the same size. (" + std::to_string(computed_vec.size()) + "), (" + std::to_string(reference_vec.size()) + ")";
            throw std::runtime_error(error_msg);
        }
        
        long double error_norm = 0.0;
        long double reference_norm = 0.0;
        for (size_t idx = 0; idx < length; ++idx)
        {
            const long double &reference_r = reference_vec[idx].real();
            const long double &computed_r = computed_vec[idx].real();
            const long double &reference_i = reference_vec[idx].imag();
            const long double &computed_i = computed_vec[idx].imag();
            long double local_error_r = (reference_r - computed_r);
            long double local_error_i = (reference_i - computed_i);
            /*
            if (local_error_r >= MAX_RELATIVE_ERROR_FLOAT)
            {
                std::cout << "Real Error Located at Index: " << idx << "  |  Magnitude: " << local_error_r << std::endl;
            }
            if (local_error_i >= MAX_RELATIVE_ERROR_FLOAT)
            {
                std::cout << "Imaginary Error Located at Index: " << idx << "  |  Magnitude: " << local_error_i << std::endl;
            }
            */
            error_norm += (local_error_r * local_error_r) + (local_error_i * local_error_i);
            reference_norm += (reference_r * reference_r) + (reference_i * reference_i);
        }
        long double rel_error  = std::sqrt(error_norm / reference_norm);
        return rel_error;
    }

    template <typename O, typename R>
    inline std::vector<R> vec_cast(const std::vector<O> &original) {
        std::vector<R> result;
        for (O item : original) {
            result.push_back(static_cast<R>(item));
        }
        return result;
    }

}
