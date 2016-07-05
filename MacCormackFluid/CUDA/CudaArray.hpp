#ifndef CUDA_ARRAY_HPP
#define CUDA_ARRAY_HPP

#include "Modules/MemoryManagement.hpp"
#include "Modules/TextureReferenceManagement.hpp"
#include "Modules/ErrorHandling.hpp"

namespace CUDA {

template <class T>
struct CudaArray {
    CudaArray(size_t w, size_t h = 1, uint f = 0)
        : width(w), height(h), flags(f), array(nullptr) {

        desc = TextureReferenceManagement::createChannelDesc<T>();
        array = MemoryManagement::mallocArray(&desc, width, height, flags);
    }

    ~CudaArray() {
        MemoryManagement::freeArray(array);
        array = nullptr;
        width = height = 0;
        flags = 0;
    }

    void setData(const void* data) {
        checkCudaError(cudaMemcpyToArray(array, 0, 0, data, width * height
                        * sizeof(T), cudaMemcpyHostToDevice));
    }

    uint flags;
    size_t width;
    size_t height;
    cudaArray_t array;
    cudaChannelFormatDesc desc;
};

} // namespace CUDA

#endif