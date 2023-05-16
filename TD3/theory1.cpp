#include <thread>
#include <mutex>
#include <iostream>

class BoundedAtomicCounter {
        int val;
        int bound;
        std::mutex m;

    public:
        BoundedAtomicCounter(): val(0), bound(1000) {}

        void increment() {
            m.lock();
            if (val >= bound) {
                return;
            }
            ++val;
            m.unlock();
        }

        int get() const {
            return val;
        }
};

int main() {
    BoundedAtomicCounter counter;
    /* We run one thread until 1000 iterations
     * and it exits successfully as we see the output
     */
    std::thread t1([&counter](){
        for (int i = 0; i < 1000; ++i) {
            counter.increment();
        }
    });
    t1.join();
    std::cout << counter.get() << std::endl;

    /* We run another thread until 1001 iterations but
     * it does not exit because of the infinite lock
     */
    std::thread t2([&counter](){
        for (int i = 0; i < 1001; ++i) {
            counter.increment();
        }
    });
    t2.join();
    std::cout << counter.get() << std::endl;
}