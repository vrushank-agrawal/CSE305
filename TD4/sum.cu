#include <algorithm>
#include <iostream>
#include <chrono>
#include <math.h>

//------------------------------------------------

double Sum(double* arr, size_t N) {
    double result = 0.;
    for (size_t i = 0; i < N; ++i) {
        result += arr[i];
    }
    return result;
}

//-------------------------------------------------


/**
 * @brief Computes the sum of the array
 * @param arr - the pointer to the beginning of an array
 * @param N - the length of the array
 */
double SumGPU(double* arr, size_t N) {
    return 0.;
}

//---------------------------------------------------

int main(int argc, char* argv[]) {
    // setting the random seed to get the same result each time
    srand(42);

    // taking as input, which algo to run
    int alg_ind = std::stoi(argv[1]);

    // Generating data
    size_t N = 1 << 26;
    double* arr = (double*) malloc(N * sizeof(double));
    for (size_t i = 0; i < N; ++i) {
          arr[i] = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
    }
 
    double result = 0.;
    auto start = std::chrono::steady_clock::now();
    switch (alg_ind) {
        case 0: 
            result = Sum(arr, N);
            break;
        case 1:
            result = SumGPU(arr, N);
            break;
    }
    auto finish = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count(); 
    std::cout << "Elapsed time: " << elapsed << std::endl;
    std::cout << "Total result: " << result << std::endl;
    
    delete[] arr;
    return 0;
}
