#ifndef CUDA_DEVICE_CONTEXT_HPP
#define CUDA_DEVICE_CONTEXT_HPP

#include "../Utils/Singularity.hpp"

namespace CUDA {

class CudaDeviceContextPrivate;

class CudaDeviceContext : public Utils::Singularity<CudaDeviceContext> {
    DISABLE_COPY(CudaDeviceContext)

public:
    CudaDeviceContext();
    ~CudaDeviceContext();

public:
    bool create(int device = 0);
    bool isCreated() const;

    void destroy();

    void reset() const;
    void synchronize() const;

    bool setDevice(int device);
    int getDevice() const;
    int getDeviceCount() const;

    size_t getFreeDeviceMemoryBytes() const;
    size_t getTotalDeviceMemoryBytes() const;
    size_t getFreeDeviceMemoryMegaBytes() const;
    size_t getTotalDeviceMemoryMegaBytes() const;

    cudaDeviceProp getDeviceProperties() const;

private:
    DECLARE_PRIVATE(CudaDeviceContext)

    CudaDeviceContextPrivate* dPtr;
};

} // namespace CUDA

#define Ctx CUDA::CudaDeviceContext::getSingularityPtr()
#define CtxSync checkCudaError(cudaDeviceSynchronize());

#endif