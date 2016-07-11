#include "../StdAfx.hpp"
#include "CudaEventTimer.hpp"

#include "Modules/ErrorHandling.hpp"

namespace CUDA {

class CudaEventTimerPrivate {
public:
    CudaEventTimerPrivate(CudaEventTimer::EventFlag flag)
        : eventFlag(flag)
        , running(false)
        , timeDiff(0)
        , totalTime(0)
        , averageTime(0)
        , timeSessions(0)
        , startEvent(nullptr)
        , stopEvent(nullptr) {}

    ~CudaEventTimerPrivate() {
        destroy();
    }

    void create();
    void destroy();

    void start();
    void stop();
    void reset();

    bool running;
    float timeDiff;
    float totalTime;
    float averageTime;
    uint timeSessions;

    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;
    CudaEventTimer::EventFlag eventFlag;

    cudaStream_t stream = nullptr;
};

void CudaEventTimerPrivate::create() {
    if (Ctx->isCreated() && !startEvent && !stopEvent) {
        checkCudaError(cudaEventCreateWithFlags(&startEvent, eventFlag));
        checkCudaError(cudaEventCreateWithFlags(&stopEvent, eventFlag));
    }
}

void CudaEventTimerPrivate::destroy() {
    if (Ctx->isCreated() && startEvent && stopEvent) {
        checkCudaError(cudaEventSynchronize(startEvent));
        checkCudaError(cudaEventSynchronize(stopEvent));
        checkCudaError(cudaEventDestroy(startEvent));
        checkCudaError(cudaEventDestroy(stopEvent));
        startEvent = nullptr;
        stopEvent = nullptr;
        running = false;
        reset();
    }
}

void CudaEventTimerPrivate::start() {
    if (Ctx->isCreated() && startEvent && stopEvent) {
        checkCudaError(cudaEventRecord(startEvent, stream));
        checkCudaError(cudaEventSynchronize(startEvent));
        running = true;
    }
}

void CudaEventTimerPrivate::stop() {
    if (Ctx->isCreated() && startEvent && stopEvent) {
        checkCudaError(cudaEventRecord(stopEvent));
        checkCudaError(cudaEventSynchronize(stopEvent));
        checkCudaError(cudaEventElapsedTime(&timeDiff, startEvent, stopEvent));
        totalTime += timeDiff;
        timeSessions++;
        running = false;
    }
}

void CudaEventTimerPrivate::reset() {
    timeDiff = 0;
    totalTime = 0;
    averageTime = 0;
    if (running) {
        start();
    }
}

CudaEventTimer::CudaEventTimer()
    : dPtr(new CudaEventTimerPrivate(CudaEventTimer::Default)) {

    D(CudaEventTimer);
    d->create();
}

CudaEventTimer::CudaEventTimer(CudaEventTimer::EventFlag flag)
    : dPtr(new CudaEventTimerPrivate(flag)) {

    D(CudaEventTimer);
    d->create();
}

CudaEventTimer::~CudaEventTimer() {
    _delete(dPtr);
}

void CudaEventTimer::setFlag(CudaEventTimer::EventFlag flag) {
    D(CudaEventTimer);
    d->eventFlag = flag;
}

CudaEventTimer::EventFlag CudaEventTimer::getFlag() const {
    D(const CudaEventTimer);
    return d->eventFlag;
}

void CudaEventTimer::start() {
    D(CudaEventTimer);
    d->start();
}

void CudaEventTimer::stop() {
    D(CudaEventTimer);
    d->stop();
}

void CudaEventTimer::reset() {
    D(CudaEventTimer);
    d->reset();
}

float CudaEventTimer::getTime() const {
    D(const CudaEventTimer);
    return d->timeDiff;
}

float CudaEventTimer::getTotalTime() const {
    D(const CudaEventTimer);
    return d->totalTime;
}

float CudaEventTimer::getAverageTime() const {
    D(const CudaEventTimer);
    return (d->timeSessions > 0) ? (d->totalTime / d->timeSessions) : (0.0f);
}

} // namespace CUDA
