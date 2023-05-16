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
    }
}

/*

SPACE TO REPORT AND ANALYZE THE RUNTIMES


 */
