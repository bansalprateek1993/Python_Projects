#include <thread>
#include <queue>
#include <chrono>
#include <mutex>
#include <atomic>

using namespace std::chrono;
using namespace std;
class Task {
public:
    virtual void run() = 0;
};
template<typename T, typename = enable_if<std::is_base_of<Task, T>::value>>
class SchedulerItem {
public:
    T task;
    time_point<steady_clock> startTime;
    int delay;
    SchedulerItem(T t, time_point<steady_clock> s, int d) : task(t), startTime(s), delay(d){}
};
template<typename T, typename = enable_if<std::is_base_of<Task, T>::value>>
class Scheduler {
public:
    queue<SchedulerItem<T>> pool;
    mutex mtx;
    atomic<bool> running;
    Scheduler() : running(false){}
    void add(T task, double delayMsToRun) {
        lock_guard<mutex> lock(mtx);
        pool.push(SchedulerItem<T>(task, high_resolution_clock::now(), delayMsToRun));
        if (running == false) runNext();
    }
    void runNext(void) {
        running = true;
        auto th = [this]() {
            mtx.lock();
            auto item = pool.front();
            pool.pop();
            mtx.unlock();
            auto remaining = (item.startTime + milliseconds(item.delay)) - high_resolution_clock::now();
            if(remaining.count() > 0) this_thread::sleep_for(remaining);
            item.task.run();
            if(pool.size() > 0) 
                runNext();
            else
                running = false;
        };
        thread t(th);
        t.detach();
    }
};


class MyTask : Task {
public:
    virtual void run() override {
        printf("mytask \n");
    };
};

int main()
{
    Scheduler<MyTask> s;

    s.add(MyTask(), 0);
    s.add(MyTask(), 2000);
    s.add(MyTask(), 2500);
    s.add(MyTask(), 6000);
    std::this_thread::sleep_for(std::chrono::seconds(10));

}
