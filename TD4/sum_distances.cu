#include <algorithm>
#include <iostream>
#include <chrono>
#include <math.h>
#include <cfloat>
#include <climits>
#include <thread>
#include <numeric>
#include <iterator>
#include <vector>
#include <atomic>
#include <cmath>
#include <mutex>

__device__ double DistKer(double* p, double* q, size_t dim) {
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

void SumMapThread(double* arr, size_t begin, size_t end, double& result) {
    while (begin != end)
        result += arr[begin++];
}

void SumParallel(double* result, size_t N, size_t num_threads) {
    size_t block_size = N / num_threads;
    std::vector<double> results(num_threads, 0.0);

    std::vector<std::thread> workers(num_threads - 1);
    for (size_t i = 0; i < num_threads - 1; i++)
        workers[i] = std::thread(&SumMapThread,
                                result,
                                (i*block_size),
                                ((i+1)*block_size),
                                std::ref(results[i]));
    SumMapThread(result, (num_threads-1)*block_size, N, results[num_threads - 1]);

    for (auto& t : workers) t.join();
    result[0] = std::accumulate(results.begin(), results.end(), 0.0);
}

__global__ void SumDistancesGPUAux(double* arr, size_t dim, size_t N, double* result) {
    size_t curr_pt = blockIdx.x * blockDim.x + threadIdx.x;

    if (curr_pt >= N) return;

    double* p = arr + (curr_pt * dim);
    double sum = 0;
    for (size_t i = 0; i < N; ++i) {
        if (i == curr_pt) continue;
        sum += DistKer(p, arr + (i * dim), dim);
    }

    result[curr_pt] = sum;
}

/**
 * @brief Computes the sum of pairwise distances between the points
 * @param arr - the pointer to the beginning of an array of length N * dim representing N points
 *        of dimension dim each (each point is represented by dim consecutive elements)
 * @param dim - dimension of the ambient space
 * @param N - the number of points
 */

double SumDistancesGPU(double* arr, size_t dim, size_t N) {
    const size_t block_size = 64;
    const size_t threads_per_block = 512;
    const size_t total_threads = block_size * threads_per_block;

    double* result = new double[N];
    double* arrGPU;
    double* resultGPU;
    cudaMalloc(&arrGPU, N * dim * sizeof(double));
    cudaMalloc(&resultGPU, N * sizeof(double));
    cudaMemcpy(arrGPU, arr, N * dim * sizeof(double), cudaMemcpyHostToDevice);

    for (size_t i = 0; i < N; i += total_threads)
        SumDistancesGPUAux<<<block_size, threads_per_block>>>(arrGPU + (i * dim * total_threads), dim, N, resultGPU + (i * total_threads));
    cudaDeviceSynchronize();

    cudaMemcpy(result, resultGPU, N * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(arrGPU);
    cudaFree(resultGPU);

    SumParallel(result, N, 8);
    double sum = result[0];
    delete[] result;
    return sum/2;
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
