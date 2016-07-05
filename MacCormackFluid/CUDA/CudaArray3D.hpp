#ifndef CUDA_ARRAY_3D_HPP
#define CUDA_ARRAY_3D_HPP

#include "Modules/MemoryManagement.hpp"
#include "Modules/TextureReferenceManagement.hpp"
#include "Modules/ErrorHandling.hpp"

namespace CUDA {

template <class T>
struct CudaArray3D {
    CudaArray3D(cudaExtent ext, uint f = 0)
        : extent(ext), flags(f), array(nullptr) {

        desc = TextureReferenceManagement::createChannelDesc<T>();
        array = MemoryManagement::malloc3DArray(&desc, extent, flags);
    }

    ~CudaArray3D() {
        MemoryManagement::free3DArray(array);
        array = nullptr;
        extent = { 0 };
        flags = 0;
    }

    void setData(void* data) {
        cudaMemcpy3DParms copyParams = { 0 };
        copyParams.srcPtr = MemoryManagement::createCudaPitchedPtr
            (data, extent.width * sizeof(T), extent.width, extent.height);
        copyParams.dstArray = array;
        copyParams.extent = extent;
        copyParams.kind = cudaMemcpyHostToDevice;
        checkCudaError(cudaMemcpy3D(&copyParams));
    }

    void setData(cudaArray_const_t data) {

    }

    void* getData() {
        return nullptr;
    }

    uint flags;
    cudaExtent extent;
    cudaArray_t array;
    cudaChannelFormatDesc desc;
};

} // namespace CUDA

#endif