#include <iostream>
#include <mutex>
#include <thread>
#include <utility>

class DoubleCounter {
        int val_first;
        int val_second;
        std::mutex m_first;
        std::mutex m_second;

    public:
        DoubleCounter(): val_first(0), val_second(0) {}

        void increment_first() {
            m_first.lock();
            ++val_first;
            m_first.unlock();
        }

        void increment_second() {
            m_second.lock();
            ++val_second;
            m_second.unlock();
        }

        void increment_both() {
            m_first.lock();
            m_second.lock();
            ++val_first;
            ++val_second;
            m_first.unlock();
            m_second.unlock();
        }

        std::pair<int, int> get() const {
            return std::make_pair(val_first, val_second);
        }
};

void test() {
    DoubleCounter counter;

    /* Thread to increment both values */
    std::thread t1([&counter]() {
        for (int i = 0; i < 1000000; ++i)
            counter.increment_both();
    });

    /* Thread to check that the values are equal when read */
    std::thread t2([&counter]() {
        for (int i = 0; i < 1000000; ++i) {
            std::pair<int, int> vals = counter.get();
            if (vals.first != vals.second)
                std::cout << "Difference found:" << vals.first << " " << vals.second << std::endl;
        }
    });
    t1.join();
    t2.join();
}

int main() {
    test();
    return 0;
}
