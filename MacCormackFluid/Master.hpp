#ifndef MASTER_HPP
#define MASTER_HPP

class MasterPrivate;

class Master {
    DISABLE_COPY(Master)

public:
    Master(int rank, int worldSize);
    ~Master();

public:
    bool initialize(const uint3& volumeSize);
    void destroy();
    void update(float* data);
    void render();

private:
    DECLARE_PRIVATE(Master)

    MasterPrivate* dPtr;
};

#endif
