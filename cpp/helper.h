#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>

namespace HELP
{
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
    template <typename T=double>
    void print_vec(std::vector<T> vec) {
        for (T i: vec) {
            printf("%f ", i);
        }
        printf("\n");
    }
}
