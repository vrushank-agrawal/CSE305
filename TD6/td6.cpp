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
    std::atomic<int> searching{0};

public:
    void insert(int val) {
        // Set inserting to true
        inserting.store(true, std::memory_order_release);

        while (searching.load(std::memory_order_acquire) > 0)
            ;

        std::lock_guard<std::mutex> lock(m);
        // find the first element that is greater than val and insert val before it
        data.insert(std::upper_bound(data.begin(), data.end(), val), val);

        // Set inserting to false
        inserting.store(false, std::memory_order_release);
    }

    bool search(int val) {
        // If an insertion is happening don't start a search
        while (inserting.load(std::memory_order_acquire))
            ;

        searching.fetch_add(1, std::memory_order_relaxed);

        // Search the data
        bool found = std::find(data.begin(), data.end(), val) != data.end();

        searching.fetch_sub(1, std::memory_order_relaxed);

        return found;
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

