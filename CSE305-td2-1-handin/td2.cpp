#pragma once
#include <cfloat>
#include <climits>
#include <thread>
#include <numeric>
#include <iterator>
#include <vector>
#include <atomic>
#include <cmath>
#include <mutex>

using namespace std;

/**
 * @brief Finds the maximum in the array in parallel
 * @param start - pointer to the beginning of the array
 * @param N - length of the array
 * @param num_threads - the number of threads to be used
 */
double MaxParallel(double* start, size_t N, size_t num_threads) {
    if (N == 0) return 0.0;

    size_t block_size = N / num_threads;

    int max = INT_MIN;
    std::mutex m;
    std::vector<std::thread> workers(num_threads-1);

    /* function calculates the maximum in an array avoiding
     * race conditions by using mutexes and locks
     */
    auto thread_func = [&](double* begin, size_t block_size){
        int local_max = INT_MIN;
        for (size_t i = 0; i < block_size; i++)
            if (*begin++ > local_max)
                local_max = *(begin-1);
        std::lock_guard<std::mutex> lock(m);
            if (local_max > max)
                max = local_max;
    };

    for (size_t i = 0; i < num_threads-1; i++)
        workers[i] = (std::thread(thread_func, start + (i*block_size), block_size));
    thread_func(start + ((num_threads-1)*block_size), N - ((num_threads-1)*block_size));

    for (auto& t : workers) t.join();
    return max;
}

//-----------------------------------------------------------------------------

/**
 * @brief Sets the maximums of all the prefixes of the array start in the result array
 * @param start - pointer to the beginning of the array
 * @param N - number of elements
 * @param res_start - pointer to the beginning of the result array
 * @param offset - starting value for the maximums
 */
void PartialMax(double* start, size_t N, double* res_start, double offset) {
    res_start[0] = start[0] > offset ? start[0] : offset;
    for (size_t i = 1; i < N; i++)
        res_start[i] = start[i] > res_start[i-1] ? start[i] : res_start[i-1];
}

/**
 * @brief Checks if array prefixes are less than offset and sets them to offset
 * @param start - pointer to the beginning of the array
 * @param N - number of elements
 * @param offset - maximum value from the previous array chunk
 */
void CheckOffset(double* start, size_t N, double offset) {
    for (size_t i = 0; i < N; i++)
        if (start[i] < offset) start[i] = offset;
        else break;
}

/**
 * @brief Computes the maximums of all the prefixes of the array start
 * @param start - pointer to the beginning of the array
 * @param N - number of elements
 * @param num_threads - number of threads to be used
 * @param res_start - pointer to the beginning of the result array
 */
void PrefixMaximums(double* start, size_t N, size_t num_threads, double* res_start) {
    size_t chunk_length = N / (num_threads+1);
    std::thread workers[num_threads-1];

    /* We compute the maximums of the chunks until N-1 chuncks where N is total number of chunks */
    for (size_t i = 0; i < num_threads - 1; i++)
        workers[i] = std::thread(&PartialMax, start + (i*chunk_length), chunk_length, res_start + (i*chunk_length), INT_MIN);
    PartialMax(start + ((num_threads-1)*chunk_length), chunk_length, res_start + ((num_threads-1)*chunk_length), INT_MIN);

    for (auto& t : workers) t.join();

    /* We compute the maximums of the N-1 chunks in the offsets array */
    double offsets[num_threads];
    offsets[0] = res_start[chunk_length-1];
    for (size_t i = 1; i < num_threads; i++)
        offsets[i] = offsets[i-1] > res_start[chunk_length*(i+1) - 1] ? offsets[i-1] : res_start[chunk_length*(i+1) - 1];

    /* We compare and set the offsets from previous chunks to the current chunk */
    for (size_t i = 0; i < num_threads - 1; ++i)
        workers[i] = std::thread(&CheckOffset, res_start + ((i+1)*chunk_length), chunk_length, offsets[i]);
    /* We finally compute the maximums of the last chunk */
    PartialMax(start + (chunk_length*num_threads), N - (chunk_length*num_threads), res_start + (chunk_length*num_threads), offsets[num_threads-1]);

    for (auto& t : workers) t.join();
}

//-----------------------------------------------------------------------------
