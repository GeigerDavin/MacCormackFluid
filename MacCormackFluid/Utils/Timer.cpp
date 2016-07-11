#include "../StdAfx.hpp"
#include "Timer.hpp"

#include "LinuxTimer.hpp"
#include "WindowsTimer.hpp"
#include "../CUDA/CudaEventTimer.hpp"

namespace Utils {
namespace TimerSDK {

bool createTimer(Timer** timer) {
    if (Ctx->isCreated()) {
        *timer = (Timer *) new CUDA::CudaEventTimer;
    } else {
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)

#else

#endif
    }
    return (*timer != nullptr) ? (true) : (false);
}

bool destroyTimer(Timer** timer) {
    if (*timer) {
        delete *timer;
        *timer = nullptr;
        return true;
    }
    return false;
}

bool startTimer(Timer** timer) {
    if (*timer) {
        (*timer)->start();
        return true;
    }
    return false;
}

bool stopTimer(Timer** timer) {
    if (*timer) {
        (*timer)->stop();
        return true;
    }
    return false;
}

bool resetTimer(Timer** timer) {
    if (*timer) {
        (*timer)->reset();
        return true;
    }
    return false;
}

float getTimerValue(Timer** timer) {
    if (*timer) {
        return (*timer)->getTime();
    } else {
        return 0.0f;
    }
}

float getTotalTimerValue(Timer** timer) {
    if (*timer) {
        return (*timer)->getTotalTime();
    } else {
        return 0.0f;
    }
}

float getAverageTimerValue(Timer** timer) {
    if (*timer) {
        return (*timer)->getAverageTime();
    } else {
        return 0.0f;
    }
}

} // namespace TimerSDK
} // namespace Utils