#ifndef CUDA_EVENT_TIMER_HPP
#define CUDA_EVENT_TIMER_HPP

#include "../Utils/Timer.hpp"

namespace CUDA {

class CudaEventTimerPrivate;

class CudaEventTimer : public Utils::Timer {
    DISABLE_COPY(CudaEventTimer)

public:
    enum EventFlag {
        Default             = cudaEventDefault,
        BlockingSync        = cudaEventBlockingSync,
        DisableTiming       = cudaEventDisableTiming,
        EventInterprocess   = cudaEventInterprocess
    };

public:
    CudaEventTimer();
    explicit CudaEventTimer(CudaEventTimer::EventFlag flag);
    virtual ~CudaEventTimer() NOEXCEPT;

public:
    void setFlag(CudaEventTimer::EventFlag flag);
    CudaEventTimer::EventFlag getFlag() const;

public:
    void start() override;
    void stop() override;
    void reset() override;
    float getTime() const override;
    float getTotalTime() const override;
    float getAverageTime() const override;

private:
    DECLARE_PRIVATE(CudaEventTimer)

    CudaEventTimerPrivate* dPtr;
};

} // namespace CUDA

#endif