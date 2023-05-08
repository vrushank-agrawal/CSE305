#include <algorithm>
#include <iostream>
#include <chrono>
#include <math.h>

__device__
double DistKer(double* p, double* q, size_t dim) {
    double result = 0;
    for (size_t i = 0; i < dim; ++i) {
        result += (p[i] - q[i]) * (p[i] - q[i]);
    }
    return std::sqrt(result);
}

double Dist(double* p, double* q, size_t dim) {
    double result = 0;
    for (size_t i = 0; i < dim; ++i) {
        result += (p[i] - q[i]) * (p[i] - q[i]);
    }
    return std::sqrt(result);
}

//------------------------------------------------

double SumDistances(double* arr, size_t dim, size_t N) {
    double result = 0.;
    for (size_t i = 0; i < N; ++i) {
        double* p = arr + i * dim;
        for (size_t j = i + 1; j < N; ++j) {
            result += Dist(p, arr + j * dim, dim);
        }
    }
    return result;
}

//-------------------------------------------------


/**
 * @brief Computes the sum of pairwise distances between the points
 * @param arr - the pointer to the beginning of an array of length N * dim representing N points
 *        of dimension dim each (each point is represented by dim consecutive elements)
 * @param dim - dimension of the ambient space
 * @param N - the number of points
 */

double SumDistancesGPU(double* arr, size_t dim, size_t N) {
    return 0.;
}

//---------------------------------------------------

int main(int argc, char* argv[]) {
    // setting the random seed to get the same result each time
    srand(42);

    // taking as input, which algo to run
    int alg_ind = std::stoi(argv[1]);

    // Generating data
    size_t N = 5000;
    size_t dim = 5;
    double* arr = (double*) malloc(N * dim * sizeof(double));
    for (size_t i = 0; i < dim * N; ++i) {
          arr[i] = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
    }
 
    double result = 0.;
    auto start = std::chrono::steady_clock::now();
    switch (alg_ind) {
        case 0: 
            result = SumDistances(arr, dim, N);
            break;
        case 1:
            result = SumDistancesGPU(arr, dim, N);
            break;
    }
    auto finish = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count(); 
    std::cout << "Elapsed time: " << elapsed << std::endl;
    std::cout << "Total result: " << result << std::endl;
    
    delete[] arr;
    return 0;
}
