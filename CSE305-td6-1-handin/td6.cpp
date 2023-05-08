#include <algorithm>
#include <condition_variable>
#include <future>
#include <mutex>
#include <thread>
#include <queue>

//----------------------------------------------------------------------------
class OrderedVec {
    std::vector<int> data;
    std::mutex m;
    std::atomic<bool> inserting{false};

public:
    void insert(int val) {
        // Set inserting to true
        inserting.store(true, std::memory_order_release);

        std::lock_guard<std::mutex> lock(m);
        if (data.empty()) {
            data.push_back(val);
        } else {
            // if the vector is not sorted, sort it
            if (!std::is_sorted(data.begin(), data.end()))
                std::sort(data.begin(), data.end());

            // find the first element that is greater than val and insert val before it
            auto greater_val = std::upper_bound(data.begin(), data.end(), val);
            data.insert(greater_val, val);
        }

        // Set inserting to false
        inserting.store(false, std::memory_order_release);
    }

    bool search(int val) {
        // If an insertion is happening don't start a search
        if (inserting.load(std::memory_order_acquire))
            return false;

        // Search the data
        std::lock_guard<std::mutex> lock(m);
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
    std::unique_lock<std::mutex> lock(m);
    while (n % 2 != 0)
        iseven.wait(lock);
    n /= 2;
    iseven.notify_all();
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
    std::unique_lock<std::mutex> lock(this->lock);
    this->elements.push(element);
    if (is_empty()) this->not_empty.notify_all();
}

template <class E>
E SafeUnboundedQueue<E>::pop() {
    std::unique_lock<std::mutex> lock(this->lock);
    this->not_empty.wait(lock, [this](){return !this->is_empty();});
    E element = this->elements.front();
    this->elements.pop();
    return element;
}

//-----------------------------------------------------------------------------

