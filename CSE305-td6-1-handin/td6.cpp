#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <future>
#include <iostream>
#include <cmath>
#include <mutex>
#include <numeric>
#include <thread>
#include <queue>

//----------------------------------------------------------------------------

class OrderedVec {
        std::vector<int> data;
        std::condition_variable cv;
        std::mutex m;
        std::mutex search_m;
        // You can add some mutexes/atomics here is you want
    public:
        OrderedVec() {
        }

        void insert(int val) {
            std::lock_guard<std::mutex> lock(m);
            cv.wait(lock, [this](){return true;});
            if (data.empty())
                data.push_back(val);
            else
                for (auto& v : data)
                    if (v > val) {
                        // insert the data and extend the vector
                        data.insert(v, val);
                        break;
                    }
            cv.notify_all();
        }

        bool search(int val) {
            std::lock_guard<std::mutex> lock(search_m);
            cv.wait(lock, [this](){return true;});
            if (std::find(data.begin(), data.end(), val) != data.end())
                return true;
            return false;
        }

        // used for testing
        std::vector<int> get_data() {
            return data;
        }
};

//----------------------------------------------------------------------------

void DivideOnceEven(std::condition_variable& iseven, std::mutex& m, int& n) {
}

//-----------------------------------------------------------------------------

template <class E>
class SafeUnboundedQueue {
        std::queue<E> elements;
        std::mutex lock;
        std::condition_variable not_empty;
    public:
        SafeUnboundedQueue<E>(){}
        void push(const E& element);
        E pop ();
        bool is_empty() const {return this->elements.empty();}
};

template <class E>
void SafeUnboundedQueue<E>::push(const E& element) {
}

template <class E>
E SafeUnboundedQueue<E>::pop() {
}

//-----------------------------------------------------------------------------

