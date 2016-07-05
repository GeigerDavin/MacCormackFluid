#ifndef CUDA_DEVICE_POINTER_HPP
#define CUDA_DEVICE_POINTER_HPP

#include "Modules/MemoryManagement.hpp"

namespace CUDA {

template <class T>
struct CudaDevicePointer {
    CudaDevicePointer(size_t count)
        : size(count), ptr(nullptr) {

        ptr = MemoryManagement::mallocDevice(size);
    }

    ~CudaDevicePointer() {
        MemoryManagement::freeDevice(ptr);
        ptr = nullptr;
        size = 0;
    }

    T* ptr;
    size_t size;
};

} // namespace CUDA

#endif