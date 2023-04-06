#pragma once
#include <cfloat>
#include <climits>
#include <thread>
#include <numeric>
#include <iterator>
#include <vector>
#include <atomic>
#include <mutex>

//-----------------------------------------------------------------------------

template <typename T>
void FindThread(T* arr, size_t block_size, T target, unsigned int count, std::atomic<unsigned int>& occurences) {
}

/**
 * @brief Checks if there are at least `count` occurences of targert in the array
 * @param arr - pointer to the first element of the array
 * @param N - the length of the array
 * @param target - the target to search for
 * @param count - the number of occurences to stop after
 * @param num_threads - the number of threads to use
 * @return if there are at least `count` occurences
*/
template <typename T>
bool FindParallel(T* arr, size_t N, T target, size_t count, size_t num_threads) {
    if (N==0) return false;

    std::atomic<size_t> exists(0);
    size_t block_size = N / num_threads;
    std::vector<std::thread> workers(num_threads-1);

    auto thread_func = [&](T* start, T* end) {
        while (start != end)
            if (*start++ == target)
                if (++exists >= count)
                    return;
    };

    for (size_t i = 0; i < num_threads-1; i++)
        workers[i] = std::thread(thread_func, arr + (i*block_size), arr + (i+1)*block_size);
    thread_func((arr + (num_threads-1)*block_size), arr+N);

    for (auto& t : workers) t.join();
    return exists >= count;
}

//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------

class Account {
        unsigned int amount;
        unsigned int account_id;
        static std::mutex lock;

        static std::atomic<unsigned int> max_account_id;
    public:
        Account() {
            this->lock.lock();
            this->amount = 0;
            this->account_id = ++max_account_id;
            this->lock.unlock();
        }

        Account(unsigned int amount) {
            this->lock.lock();
            this->amount = amount;
            this->account_id = ++max_account_id;
            this->lock.unlock();
        }

        // copy-contructor and assignment are deleted to make the id's really unique
        Account(const Account& other) = delete;

        Account& operator = (const Account& other) = delete;

        unsigned int get_amount() const {
            this->lock.lock();
            unsigned int amt = this->amount;
            this->lock.unlock();
            return amt;
        }

        unsigned int get_id() const {
            return this->account_id;
        }

        // withdrwas deduction if the current amount is at least deduction
        // returns whether the withdrawal took place
        bool withdraw(unsigned int deduction) {
            this->lock.lock();
            if (this->amount >= deduction) {
                this->amount -= deduction;
                this->lock.unlock();
                return true;
            }
            this->lock.unlock();
            return false;
        }

        // adds the prescribed amount of money to the account
        void add(unsigned int to_add) {
            this->lock.lock();
            this->amount += to_add;
            this->lock.unlock();
        }

        // transfers amount from from to to if there are enough money on from
        // returns whether the transfer happened
        static bool transfer(unsigned int amount, Account& from, Account& to) {
            if (from.get_id() == to.get_id()) return false;
            if (from.get_amount() < amount) return false;
            if (from.get_id() > to.get_id()) {
                to.add(amount);
                from.withdraw(amount);
                return true;
            }
            from.withdraw(amount);
            to.add(amount);
            return true;
        }
};

std::mutex Account::lock;
std::atomic<unsigned int> Account::max_account_id(0);

//-----------------------------------------------------------------------------


