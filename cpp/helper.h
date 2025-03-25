#pragma once

#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <complex>
#include <cstdlib>
#include <type_traits>
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

    int tokenize_csv(
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
    void print_vec(std::vector<T> vec)
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
    void print_vec_complex(std::vector<T> vec)
    {
        for (T i : vec)
        {
            printf("(%f, %f) ", i.real(), i.imag());
        }
        printf("\n");
    }

    wav_hdr read_wav(char * fin, std::vector<int16_t>* out) {
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
        printf("%d nsamp\n", numSamples);
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
    void write_wav(char * fout,const  std::vector<int16_t>& in, wav_hdr wav_details) {
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

}
