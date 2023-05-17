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
    Node* current = this->head->next;
    std::string newItem;
    unsigned long newKey;

    // create a tuple array for the new keys and items
    std::vector<std::tuple<unsigned long, std::string>> items;

    // iterate through the list and apply the function to each item
    while (current->key != LARGEST_KEY) {
        std::lock_guard<std::mutex> lock(current->lock);
        newItem = f(current->item);
        newKey = std::hash<std::string>{}(newItem);
        items.push_back(std::make_tuple(newKey, newItem));
        current = current->next;
    }

    // sort the array based on the keys
    std::sort(items.begin(), items.end());

    // remove duplicates
    items.erase(std::unique(items.begin(), items.end()), items.end());

    // replace the items in the list with the new items
    current = this->head;
    for (auto& item : items) {
        {
            std::lock_guard<std::mutex> lock(current->lock);
            current->next = new Node(std::get<1>(item));
            current = current->next;
        }
        std::lock_guard<std::mutex> lock2(current->lock);
        current->next = new Node(SetList::LARGEST_KEY);
    }
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
    // unique lock allows fine-grained insertion
    std::unique_lock<std::mutex> countLock(count_lock);

    // Wait until the set has available space
    while (count >= capacity)
        not_full.wait(countLock);

    Node* pred = this->head;
    Node* curr = pred->next;

    while (true) {
        std::lock_guard<std::mutex> predLock(pred->lock);
        std::lock_guard<std::mutex> currLock(curr->lock);

        if (curr->key == LARGEST_KEY || curr->key > std::hash<std::string>{}(val)) {
            Node* newNode = new Node(val);
            newNode->next = curr;
            pred->next = newNode;
            ++count;
            break;
        }

        pred = curr;
        curr = curr->next;
    }

    count_lock.unlock();
    not_full.notify_one();

    return true;
}

bool BoundedSetList::remove(const std::string& val) {
    Node* pred = this->head;
    Node* curr = pred->next;

    while (true) {
        // Lock the current node
        std::lock_guard<std::mutex> predLock(pred->lock);
        std::lock_guard<std::mutex> currLock(curr->lock);

        if (curr->key == LARGEST_KEY)
            break;

        if (curr->key == std::hash<std::string>{}(val)) {
            pred->next = curr->next;
            delete curr;
            std::lock_guard<std::mutex> countLock(count_lock);
            --count;
            not_full.notify_one();
            return true;
        }
        pred = curr;
        curr = curr->next;
    }

    return false;
}

//-----------------------------------------------------------------------------
