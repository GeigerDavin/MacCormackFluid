#ifndef CUDA_HOST_POINTER_HPP
#define CUDA_HOST_POINTER_HPP

#include "Modules/MemoryManagement.hpp"

namespace CUDA {

template <class T>
struct CudaHostPointer {
    CudaHostPointer(size_t count)
        : size(count), ptr(nullptr) {

        ptr = MemoryManagement::mallocHost(size);
    }

    ~CudaHostPointer() {
        MemoryManagement::freeHost(ptr);
        ptr = nullptr;
        size = 0;
    }

    T* ptr;
    size_t size;
};

} // namespace CUDA

#endif