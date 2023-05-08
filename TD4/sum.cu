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

__global__
void PartialSumGPUAux(double* arr, double* partial_sums, size_t N, size_t chunk_size) {
    // implement partial sum of the array on GPU using cuda
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    size_t start = tid * chunk_size;
    size_t end = std::min(start + chunk_size, N);
    double sum = 0.;
    for (size_t i = start; i < end; ++i) {
        sum += arr[i];
        partial_sums[tid] = sum;
    }
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
    double* arrGPU;
    cudaMalloc(&arrGPU, N * sizeof(double));
    cudaMemcpy(arr, arrGPU, N * sizeof(double), cudaMemcpyHostToDevice);

    // computing on GPU
    size_t chunk_size = (N + TOTAL_THREADS + 1) / TOTAL_THREADS;
    AddGPUAux<<<BLOCKS_NUM, THREADS_PER_BLOCK>>>(xd, yd, resd, N, chunk_size);
    cudaDeviceSynchronize();

    // copying the result back
    cudaMemcpy(res, resd, N * sizeof(double), cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(xd);
    cudaFree(yd);
    cudaFree(resd);

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
