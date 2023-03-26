#pragma once
#include <climits>
#include <thread>
#include <numeric>
#include <iterator>
#include <optional>
#include <vector>
#include <atomic>
#include <cmath>
#include <mutex>
#include <condition_variable>

typedef long double Num;
typedef std::vector<long double>::const_iterator NumIter;
typedef std::vector<int>::const_iterator IntIter;

//-----------------------------------------------------------------------------

void SumMapThread(NumIter begin, NumIter end, Num f(Num), long double& result) {
    while (begin != end)
        result += f(*begin++);
}

/**
 * @brief Sums f(x) for x in [begin, end)
 * @param begin Start iterator
 * @param end End iterator
 * @param f Function to apply
 * @param result Reference to the result variable
 */
Num SumParallel(NumIter begin, NumIter end, Num f(Num), size_t num_threads) {
    if (end == begin) return 0.0;
    size_t block_size = (end - begin) / num_threads;
    std::vector<Num> results(num_threads, 0.0);

    std::vector<std::thread> workers(num_threads - 1);
    for (size_t i = 0; i < num_threads - 1; i++)
        workers[i] = std::thread(&SumMapThread,
                                begin + (i*block_size),
                                begin + ((i+1)*block_size),
                                f,
                                std::ref(results[i]));
    SumMapThread(begin + ((num_threads-1)*block_size), end, f, results[num_threads - 1]);

    for (auto& t : workers) t.join();
    return std::accumulate(results.begin(), results.end(), 0.0);
}

//-----------------------------------------------------------------------------

/**
 * @brief Computes the mean of the numbers in [begin, end)
 * @param begin Start iterator
 * @param end End iterator
 * @param num_threads The number of threads to use
 * @return The mean in the range
*/
Num MeanParallel(NumIter begin, NumIter end, size_t num_threads) {
    if (end == begin) return 0.0;
    return (SumParallel(begin, end, [](Num x) -> Num {return abs(x);}, num_threads))/(end-begin);
}

//-----------------------------------------------------------------------------

/**
 * @brief Computes the variance of the numbers in [begin, end)
 * @param begin Start iterator
 * @param end End iterator
 * @param num_threads The number of threads to use
 * @return The variance in the range
*/
Num VarianceParallel(NumIter begin, NumIter end, size_t num_threads) {
    if (end == begin) return 0.0;
    return (SumParallel(begin, end, [](Num x) -> Num {return x*x;}, num_threads))/(end-begin) 
            - pow(MeanParallel(begin, end, num_threads), 2);
}

//-----------------------------------------------------------------------------


/**
 * @brief Computs the occurences of the minimal value in [begin, end)
 * @param begin Start iterator
 * @param end End iterator
 * @param num_threads The number of threads to use
 * @return the number of occurences of the minimal value in [begin, end)
*/
int CountMinsParallel(IntIter begin, IntIter end, size_t num_threads) {
    int block_size = (end - begin) / num_threads;

    std::atomic<int> min(INT_MAX);
    std::vector<std::thread> workers(num_threads-1);

    auto thread_func = [&](IntIter begin, IntIter end){
        while (begin != end)
            if (*begin++ < min)
                min = *(begin-1);
    };

    for (size_t i = 0; i < num_threads-1; i++)
        workers[i] = (std::thread(thread_func, begin + (i*block_size), begin + ((i+1)*block_size)));
    thread_func(begin + ((num_threads-1)*block_size), end);

    for (auto& t : workers) t.join();

    auto thread_func2 = [&](IntIter begin, IntIter end, int& count){
        while (begin != end)
            if (*begin++ == min)
                count++;
    };

    std::vector<int> count(num_threads, 0);
    for (size_t i = 0; i < num_threads-1; i++)
        workers[i] = (std::thread(thread_func2, begin + (i*block_size), begin + ((i+1)*block_size), std::ref(count[i])));
    thread_func2(begin + ((num_threads-1)*block_size), end, count[num_threads-1]);

    for (auto& t : workers) t.join();
    return std::accumulate(count.begin(), count.end(), 0);
}

//-----------------------------------------------------------------------------


/**
 * @brief Finds target in [begin, end)
 * @param begin Start iterator
 * @param end End iterator
 * @param target The target to search for
 * @param num_threads The number of threads to use
 * @return The sum in the range
*/
template <typename Iter, typename T>
bool FindParallel(Iter begin, Iter end, T target, size_t num_threads) {
    if (end == begin) return false;

    std::atomic<bool> found(false);
    size_t block_size = (end - begin) / num_threads;
    std::vector<std::thread> workers(num_threads-1);

    auto thread_func = [&](Iter start, Iter end, T target) {
        while (start != end){
            if (found) break;
            if (*start++ == target) found = true;
        }
    };

    for (size_t i = 0; i < num_threads-1; i++)
        workers[i] = std::thread(thread_func,
                                begin + (i*block_size),
                                begin + ((i+1)*block_size),
                                target);

    thread_func(begin + ((num_threads-1)*block_size), end, target);

    for (auto& t : workers) t.join();
    return found;
}

//-----------------------------------------------------------------------------


/**
 * @brief Runs a function and checks whether it finishes within a timeout
 * @param f Function to run
 * @param arg Arguments to pass to the function 
 * @param timeout The timeout
 * @return std::optional with result (if the function finishes) and empty (if timeout)
 */
template <typename ArgType, typename ReturnType>
std::optional<ReturnType> RunWithTimeout(ReturnType f(ArgType), ArgType arg, size_t timeout) {

    // Inspired by https://stackoverflow.com/questions/40550730/how-to-implement-timeout-for-function-in-c
    std::mutex m;
    std::optional<ReturnType> result;
    std::condition_variable cv;
    bool finished = false;

    std::thread t([&](){
        result = f(arg);
        cv.notify_one();
    });

    t.detach();

    std::unique_lock<std::mutex> lock(m);
    if (cv.wait_for(lock, std::chrono::milliseconds(timeout), [&](){return finished;}))
        return {};
    else
        return result;
}

//-----------------------------------------------------------------------------
