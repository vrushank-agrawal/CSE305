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

//------------------------------------------------

double Sum(double* arr, size_t N) {
    double result = 0.;
    for (size_t i = 0; i < N; ++i) {
        result += arr[i];
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

__global__ void PartialSumGPUAux(double* arr, double* partial_sums, size_t N, size_t chunk_size) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t begin = chunk_size * index;
    size_t end = chunk_size * (index + 1);
    double sum = 0.;

    if (end > N)
        end = N;

    for (size_t i = begin; i < end; ++i)
        sum += arr[i];

    partial_sums[index] = sum;
}

/**
 * @brief Computes the sum of the array
 * @param arr - the pointer to the beginning of an array
 * @param N - the length of the array
 */
double SumGPU(double* arr, size_t N) {
    // implement sum of the array on GPU using cuda
    const size_t BLOCKS_NUM = 64;
    const size_t THREADS_PER_BLOCK = 256;
    const size_t TOTAL_THREADS = BLOCKS_NUM  * THREADS_PER_BLOCK;

    // moving the data to device
    double* result = new double[TOTAL_THREADS];
    double* arrGPU;
    double* partialResGPU;
    cudaMalloc(&arrGPU, N * sizeof(double));
    cudaMalloc(&partialResGPU, TOTAL_THREADS * sizeof(double));
    cudaMemcpy(arrGPU, arr, N * sizeof(double), cudaMemcpyHostToDevice);

    // computing on GPU
    size_t chunk_size = (N + TOTAL_THREADS + 1) / TOTAL_THREADS;
    PartialSumGPUAux<<<BLOCKS_NUM, THREADS_PER_BLOCK>>>(arrGPU, partialResGPU, N, chunk_size);
    cudaDeviceSynchronize();

    // copying the result back
    cudaMemcpy(result, partialResGPU, TOTAL_THREADS * sizeof(double), cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(arrGPU);
    cudaFree(partialResGPU);

    SumParallel(result, TOTAL_THREADS, 8);
    double res = result[0];
    delete[] result;

    return res;
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

