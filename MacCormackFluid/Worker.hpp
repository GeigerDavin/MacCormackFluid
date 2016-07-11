#ifndef WORKER_HPP
#define WORKER_HPP

class WorkerPrivate;

class Worker {
    DISABLE_COPY(Worker)

public:
    Worker(int rank, int worldSize);
    ~Worker();

public:
    bool initialize(const uint3& volumeSize);
    void destroy();
    void update();

    void getData(float** data) const;

private:
    DECLARE_PRIVATE(Worker)

    WorkerPrivate* dPtr;
};

#endif
