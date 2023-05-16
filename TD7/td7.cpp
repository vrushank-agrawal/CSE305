#include <algorithm>
#include <atomic>
#include <climits>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "CoarseSetList.cpp"
#include "SetList.cpp"

//-----------------------------------------------------------------------------

class SetListTrans: public SetList {
public:
    template <typename F>
    void transform(F f);
};

template <typename F>
void SetListTrans::transform(F f) {
}

//-----------------------------------------------------------------------------


class BoundedSetList: public SetList {
    std::condition_variable not_full;
    size_t capacity;
    size_t count;
    std::mutex count_lock;
public:
    BoundedSetList(size_t capacity) {
        this->capacity = capacity;
        this->count = 0;
    }

    size_t get_capacity() const {return this->capacity;}
    size_t get_count() const {return this->count;}

    bool add(const std::string& val);
    bool remove(const std::string& val);
};

bool BoundedSetList::add(const std::string& val) {
}

bool BoundedSetList::remove(const std::string& val) {
}

//-----------------------------------------------------------------------------
