#include "../StdAfx.hpp"
#include "CudaDeviceContext.hpp"

namespace CUDA {

const size_t bytesInMegaBytes = 1048576;

class CudaDeviceContextPrivate {
public:
    CudaDeviceContextPrivate()
        : activeDevice(-1)
        , activeStream(nullptr) {}

    ~CudaDeviceContextPrivate() {
        destroy();
    }

    int activeDevice;
    cudaStream_t activeStream;

    bool create(int device);
    void destroy();
    bool isValid() const;

    int gpuDeviceInit(int device);
    int gpuGetMaxGflopsDeviceId();
    int findCudaDevice(int argc, const char** argv);
    bool checkCudaCapabilities(int major, int minor);
};

bool CudaDeviceContextPrivate::create(int device) {
    if (isValid()) {
        return true;
    }

    if (device < 0) {
        std::cout << "Using CPU only" << std::endl;
        activeDevice = -1;
        return false;
    }

    int count = 0;
    checkCudaError(cudaGetDeviceCount(&count));
    if (count < 1) {
        std::cout << "No CUDA devices found; using CPU only" << std::endl;
        activeDevice = -1;
        return false;
    }

    if (device >= 0 && device < count) {
        checkCudaError(cudaSetDevice(device));
        cudaDeviceProp prop = {0};
        checkCudaError(cudaGetDeviceProperties(&prop, device));
        if (prop.major < 2) {
            device = -1;
        }
    } else {
        size_t freeMax = 0;
        for (int i = 0; i < count; i++) {
            checkCudaError(cudaSetDevice(i));
            cudaDeviceProp prop = {0};
            checkCudaError(cudaGetDeviceProperties(&prop, i));
            if (prop.major < 2) {
                continue;
            }
            size_t free = 0, total = 0;
            checkCudaError(cudaMemGetInfo(&free, &total));
            if (free > freeMax) {
                freeMax = free;
                device = i;
            }
        }
    }

    if (device < 0) {
        std::cout << "Compute capability 2.0 support required; using CPU only" << std::endl;
        activeDevice = -1;
        return false;
    }

    checkCudaError(cudaSetDevice(device));
    checkCudaError(cudaFree(0));

    cudaDeviceProp prop = {0};
    checkCudaError(cudaGetDeviceProperties(&prop, device));
    size_t free = 0, total = 0;
    checkCudaError(cudaMemGetInfo(&free, &total));

    printf("_________________________________________\n");
    printf("%s (%lluMB free of %lluMB total)\n", prop.name, (unsigned long long) (free / bytesInMegaBytes), (unsigned long long) (total / bytesInMegaBytes));
    printf("Using %d multiprocessors\n", prop.multiProcessorCount);
    printf("Max threads per processor: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Max threads per dim: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("_________________________________________\n");
    printf("\n");

    activeDevice = device;
    return isValid();
}

void CudaDeviceContextPrivate::destroy() {
    if (isValid()) {
        std::cout << "Reset cuda device" << std::endl;
        checkCudaError(cudaDeviceReset());
        activeDevice = -1;
        activeStream = nullptr;
    }
}

bool CudaDeviceContextPrivate::isValid() const {
    return (activeDevice >= 0);
}

CudaDeviceContext::CudaDeviceContext()
    : dPtr(new CudaDeviceContextPrivate) {}

CudaDeviceContext::~CudaDeviceContext() {
    destroy();
}

bool CudaDeviceContext::create(int device) {
    D(CudaDeviceContext);
    return d->create(device);
}

bool CudaDeviceContext::isCreated() const {
    D(const CudaDeviceContext);
    return d->isValid();
}

void CudaDeviceContext::destroy() {
    _delete(dPtr);
}

void CudaDeviceContext::reset() const {
    checkCudaError(cudaDeviceReset());
}

void CudaDeviceContext::synchronize() const {
    checkCudaError(cudaDeviceSynchronize());
}

bool CudaDeviceContext::setDevice(int device) {
    if (device > 0) {
        D(CudaDeviceContext);
        checkCudaError(cudaSetDevice(device));
        d->activeDevice = device;
        return true;
    }
    return false;
}

int CudaDeviceContext::getDevice() const {
    int device = -1;
    checkCudaError(cudaGetDevice(&device));
    return device;
}

int CudaDeviceContext::getDeviceCount() const {
    int count = 0;
    checkCudaError(cudaGetDeviceCount(&count));
    return count;
}

size_t CudaDeviceContext::getFreeDeviceMemoryBytes() const {
    size_t free = 0, total = 0;
    checkCudaError(cudaMemGetInfo(&free, &total));
    return free;
}

size_t CudaDeviceContext::getTotalDeviceMemoryBytes() const {
    size_t free = 0, total = 0;
    checkCudaError(cudaMemGetInfo(&free, &total));
    return total;
}

size_t CudaDeviceContext::getFreeDeviceMemoryMegaBytes() const {
    size_t free = 0, total = 0;
    checkCudaError(cudaMemGetInfo(&free, &total));
    return (free / bytesInMegaBytes);
}

size_t CudaDeviceContext::getTotalDeviceMemoryMegaBytes() const {
    size_t free = 0, total = 0;
    checkCudaError(cudaMemGetInfo(&free, &total));
    return (total / bytesInMegaBytes);
}

cudaDeviceProp CudaDeviceContext::getDeviceProperties() const {
    D(const CudaDeviceContext);
    cudaDeviceProp prop = {0};
    checkCudaError(cudaGetDeviceProperties(&prop, d->activeDevice));
    return prop;
}

} // namespace CUDA