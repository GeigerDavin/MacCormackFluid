#ifndef TIMER_HPP
#define TIMER_HPP

namespace Utils {

/**
 * Timer abstract interface class
 */
class Timer {
    DISABLE_COPY(Timer)

public:
    Timer() = default;
    virtual ~Timer() = default;

public:
    virtual void start() = 0;
    virtual void stop() = 0;
    virtual void reset() = 0;
    virtual float getTime() const = 0;
    virtual float getTotalTime() const = 0;
    virtual float getAverageTime() const = 0;
};

namespace TimerSDK {
    bool createTimer(Timer** timer);
    bool destroyTimer(Timer** timer);
    bool startTimer(Timer** timer);
    bool stopTimer(Timer** timer);
    bool resetTimer(Timer** timer);
    float getTimerValue(Timer** timer);
    float getTotalTimerValue(Timer** timer);
    float getAverageTimerValue(Timer** timer);
} // TimerSDK

} // namespace Utils

#endif