#include <chrono>
#include <climits>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "CoarseSetList.cpp"
#include "SetList.cpp"

template <typename T>
int benchmark_single_thread(T& set, int count) {
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < count; ++i) {
        set.add(std::to_string(rand()));
    }
    auto finish = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count();
    return elapsed;
}

template <typename T>
int benchmark_multiple_threads(T& set, int count, int threads) {
    std::vector<std::thread> workers;
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < threads; ++i) {
        workers.push_back(std::thread([&set, count, threads, i]() {
            for (int j = 0; j < count / threads; ++j)
                set.add(std::to_string(rand()));
        }));
    }

    for (auto& t : workers) t.join();

    auto finish = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count();
    return elapsed;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: ./set_benchmarker num_thread num_insertions" << std::endl;
        return 0;
    }

    int num_threads = std::stoi(argv[1]);
    int num_insertions = std::stoi(argv[2]);

    if (num_threads == 1) {
        // Timing for coarse-grained
        CoarseSetList CL;
        int elapsed = benchmark_single_thread(CL, num_insertions);
        std::cout << "Time for coare-grained version is " << elapsed << " microseconds" << std::endl;

        // Timing for fine-grained
        SetList L;
        elapsed = benchmark_single_thread(L, num_insertions);
        std::cout << "Time for fine-grained version is " << elapsed << " microseconds" << std::endl;
    } else {
        // Timing for coarse-grained
        CoarseSetList CL;
        int elapsed = benchmark_multiple_threads(CL, num_threads, num_insertions);
        std::cout << "Time for coare-grained version is " << elapsed << " microseconds" << std::endl;

        // Timing for fine-grained
        SetList L;
        elapsed = benchmark_multiple_threads(L, num_threads, num_insertions);
        std::cout << "Time for fine-grained version is " << elapsed << " microseconds" << std::endl;
    }
}

/*

SPACE TO REPORT AND ANALYZE THE RUNTIMES

1.  1 thread, 1000 insertions
    Time for coare-grained version is 742 microseconds
    Time for fine-grained version is 3026 microseconds

    8 threads, 1000 insertions
    Time for coare-grained version is 55929 microseconds
    Time for fine-grained version is 43873 microseconds

2.  1 thread, 10000 insertions
    Time for coare-grained version is 131488 microseconds
    Time for fine-grained version is 338471 microseconds

    8 threads, 10000 insertions
    Time for coare-grained version is 356591 microseconds
    Time for fine-grained version is 308806 microseconds

3.  1 thread, 30000 insertions
    Time for coare-grained version is 2395682 microseconds
    Time for fine-grained version is 3783280 microseconds

    8 threads, 30000 insertions
    Time for coare-grained version is 1050977 microseconds
    Time for fine-grained version is 851559 microseconds

4.  for more than 32743 insertions on multiple threads my computer complains

Explanation:

    From the above run-times, we can see that the coare-grained version
    is faster than the fine-grained version for the single thread case.
    This is because the fine-grained version has a much smaller critical
    section than the coare-grained version, and hence, the fine-grained
    has significantly more overhead than the coare-grained version for
    acquiring and releasing locks and thus the synchronization costs.

    However, the fine-grained version is faster than the coare-grained
    version for the multi-threaded case. This is because the fine-grained
    version has a much smaller critical section than the coare-grained
    version, and hence, the fine-grained version can concurrently execute
    more insertions than the coare-grained version where the threads
    have to wait for the lock to be released before they can execute
    their insertions.

    Further, the coare-grained single thread version is faster than the
    multi-threaded version for a relatively small number of insertions
    around 22000 after which the multi-threaded version is faster. This is
    because the overhead of creating threads is more than the time saved by
    running the threads in parallel.

 */
