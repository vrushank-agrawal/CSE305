#include <algorithm>
#include <iostream>
#include <chrono>
#include <math.h>

//------------------------------------------------

double Variance(double* arr, size_t N) {
    double sum = 0.;
    double sum_squares = 0.;
    for (size_t i = 0; i < N; ++i) {
        sum += arr[i];
        sum_squares += arr[i] * arr[i];
    }
    return (sum * sum  - sum_squares) / (1. * N);
}

//-------------------------------------------------


/**
 * @brief Computes the variance of the array
 * @param arr - the pointer to the beginning of an array
 * @param N - the length of the array
 */
double VarianceGPU(double* arr, size_t N) {
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
            result = Variance(arr, N);
            break;
        case 1:
            result = VarianceGPU(arr, N);
            break;
    }
    auto finish = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count(); 
    std::cout << "Elapsed time: " << elapsed << std::endl;
    std::cout << "Total result: " << result << std::endl;
 
    delete[] arr;   
    return 0;
}
