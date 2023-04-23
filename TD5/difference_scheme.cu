#include <algorithm>
#include <iostream>
#include <chrono>
#include <math.h>

const double C = 0.5;

//------------------------------------------------

void SolvePDE(double* boundary_values, size_t N, double dx, double dt, size_t timesteps, double* result) {
    double* curr = (double*) malloc(N * sizeof(double));
    double* next = (double*) malloc(N * sizeof(double));
    memcpy(curr, boundary_values, N * sizeof(double));
    for (size_t i = 0; i < timesteps; ++i) {
        for (size_t j = 0; j < N; ++j) {
            if (j < N - 1) {
                next[j] = curr[j] + C * (dt / dx) * (curr[j + 1] - curr[j]);
            } else {
                next[j] = curr[j] + C * (dt / dx) * (curr[0] - curr[j]);
            }
        }
        std::swap(curr, next);
    }
    memcpy(result, curr, N * sizeof(double));
    free(curr);
    free(next);
}

//-------------------------------------------------

__global__ void SolvePDEKernel(double* curr, double* next, size_t N, double dx, double dt) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        next[i] = (i < N - 1) ? curr[i] + C * (dt / dx) * (curr[i + 1] - curr[i]) :
                                curr[i] + C * (dt / dx) * (curr[0] - curr[i]);
}

/**
 * @brief Solves a PDE u_t = C * u_x using a simple difference scheme
 * @param boundary_values - the pointer to the beginning of an array of values at t = 0
 * @param N - the length of the array
 * @param dx - step size for x coordinate
 * @param dt - step size for t coordinate
 * @param timesteps - number of steps in time to preform
 * @param result - pointer to yhe array for the value at the last time step
 */
void SolvePDEGPU(double* boundary_values, size_t N, double dx, double dt, size_t timesteps, double* result) {
    const size_t THREADS_PER_BLOCK = 256;
    const size_t NUM_BLOCKS = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    double* curr;
    double* next;
    cudaMalloc(&curr, N * sizeof(double));
    cudaMalloc(&next, N * sizeof(double));
    cudaMemcpy(curr, boundary_values, N * sizeof(double), cudaMemcpyHostToDevice);

    for (size_t i = 0; i < timesteps; ++i) {
        SolvePDEKernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(curr, next, N, dx, dt);
        std::swap(curr, next);
    }
    cudaDeviceSynchronize();

    cudaMemcpy(result, curr, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(curr);
    cudaFree(next);
}

//---------------------------------------------------

__global__ void SolvePDEKernelShared(double* curr, double* next, size_t N, double dx, double dt) {
    const size_t tid = threadIdx.x;
    const size_t i = blockIdx.x * blockDim.x + tid;
    __shared__ double shared_curr[256];
    if (i < N) {
        shared_curr[tid] = curr[i];
        __syncthreads();
        if (i < N-1)
            next[i] = (tid < 255) ? shared_curr[tid] + C * (dt / dx) * (shared_curr[tid + 1] - shared_curr[tid]) :
                                    shared_curr[tid] + C * (dt / dx) * (curr[i + 1] - shared_curr[tid]);
        else
            next[i] = shared_curr[tid] + C * (dt / dx) * (curr[0] - shared_curr[tid]);
    }

}

/**
 * @brief Solves a PDE u_t = C * u_x using a simple difference scheme
 * @param boundary_values - the pointer to the beginning of an array of values at t = 0
 * @param N - the length of the array
 * @param dx - step size for x coordinate
 * @param dt - step size for t coordinate
 * @param timesteps - number of steps in time to preform
 * @param result - pointer to yhe array for the value at the last time step
 */
void SolvePDEGPU2(double* boundary_values, size_t N, double dx, double dt, size_t timesteps, double* result) {
    const size_t THREADS_PER_BLOCK = 256;
    const size_t NUM_BLOCKS = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    double* curr;
    double* next;
    cudaMalloc(&curr, N * sizeof(double));
    cudaMalloc(&next, N * sizeof(double));
    cudaMemcpy(curr, boundary_values, N * sizeof(double), cudaMemcpyHostToDevice);

    for (size_t i = 0; i < timesteps; ++i) {
        SolvePDEKernelShared<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(curr, next, N, dx, dt);
        std::swap(curr, next);
    }
    cudaDeviceSynchronize();

    cudaMemcpy(result, curr, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(curr);
    cudaFree(next);
}

//---------------------------------------------------

int main(int argc, char* argv[]) {
    // setting the random seed to get the same result each time
    srand(42);

    // taking as input, which algo to run
    int alg_ind = std::stoi(argv[1]);

    // Generating data
    double length = 8 * atan(1.0); // 2 pi
    double dx = 0.0001;
    double dt = 0.0001;
    size_t N = int(length / dx);

    double* boundary = (double*) malloc(N * sizeof(double));
    double* result = (double*) malloc(N * sizeof(double));
    for (size_t i = 0; i < N; ++i) {
        boundary[i] = sin(i * dx);
    }

    size_t timesteps = 10000;
    auto start = std::chrono::steady_clock::now();
    switch (alg_ind) {
        case 0:
            SolvePDE(boundary, N, dx, dt, timesteps, result);
            break;
        case 1:
            SolvePDEGPU(boundary, N, dx, dt, timesteps, result);
            break;
        case 2:
            SolvePDEGPU2(boundary, N, dx, dt, timesteps, result);
            break;
    }
    auto finish = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count();

    for (size_t i = 0; i <  N; ++i) {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Elapsed time: " << elapsed << std::endl;

    free(boundary);
    free(result);
    return 0;
}
