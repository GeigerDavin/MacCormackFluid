#ifndef CUDA_ARRAY_3D_HPP
#define CUDA_ARRAY_3D_HPP

#include "Modules/MemoryManagement.hpp"
#include "Modules/TextureReferenceManagement.hpp"
#include "Modules/ErrorHandling.hpp"

namespace CUDA {

enum class CudaArrayFlag {
    Default             = cudaArrayDefault,
    SurfaceLoadStore    = cudaArraySurfaceLoadStore,
    TextureGather       = cudaArrayTextureGather
};

template <class T>
class CudaArray3D {
    DISABLE_COPY(CudaArray3D)

public:
    CudaArray3D()
        : array(nullptr)
        , flags(CudaArrayFlag::Default) {}

    explicit CudaArray3D(CudaArrayFlag flag)
        : array(nullptr)
        , flags(flag) {}

    ~CudaArray3D() {
        destroy();
    }

public:
    void setFlag(CudaArrayFlag flag) {
        flags = flag;
    }

    CudaArrayFlag getFlag() const {
        return flags;
    }

    cudaExtent getSize() const {
        return cudaExtent();
    }

    cudaArray_t get() {
        return array;
    }

    cudaArray_const_t get() const {
        return array;
    }

    cudaChannelFormatDesc getChannelFormatDesc() const {
        return cudaChannelFormatDesc();
    }

    void create(size_t width, size_t height, size_t depth) {
        if (isCreated()) {
            return;
        }

        T* data = (T *) MemoryManagement::mallocHost(width * height * depth * sizeof(T));
        extent = MemoryManagement::createCudaExtent(width, height, depth);
        desc = TextureReferenceManagement::createChannelDesc<T>();
        array = MemoryManagement::malloc3DArray(&desc, extent, static_cast<uint>(flags));
        cudaMemcpy3DParms copyParams = { 0 };
        copyParams.srcPtr = MemoryManagement::createCudaPitchedPtr
            (data, width * sizeof(T), width, height);
        copyParams.dstArray = array;
        copyParams.extent = extent;
        copyParams.kind = cudaMemcpyHostToDevice;
        checkCudaError(cudaMemcpy3D(&copyParams));
        MemoryManagement::freeHost(data);
    }

    void create(const T* data, size_t width, size_t height, size_t depth) {
        if (isCreated()) {
            return;
        }

        extent = MemoryManagement::createCudaExtent(width, height, depth);
        desc = TextureReferenceManagement::createChannelDesc<T>();
        array = MemoryManagement::malloc3DArray(&desc, extent, static_cast<uint>(flags));
        cudaMemcpy3DParms copyParams = { 0 };
        copyParams.srcPtr = MemoryManagement::createCudaPitchedPtr
            (data, width * sizeof(T), width, height);
        copyParams.dstArray = array;
        copyParams.extent = extent;
        copyParams.kind = cudaMemcpyHostToDevice;
        checkCudaError(cudaMemcpy3D(&copyParams));
    }

    void setData(void* data) {
        if (!isCreated()) {
            return;
        }

        cudaMemcpy3DParms copyParams = { 0 };
        copyParams.srcPtr = MemoryManagement::createCudaPitchedPtr
            (data, extent.width * sizeof(T), extent.width, extent.height);
        copyParams.dstArray = array;
        copyParams.extent = extent;
        copyParams.kind = cudaMemcpyHostToDevice;
        checkCudaError(cudaMemcpy3D(&copyParams));
    }

    bool isCreated() const {
        return (array != nullptr);
    }

    void destroy() {
        if (array) {
            std::cout << "Destroy CUDA array" << std::endl;
            MemoryManagement::free3DArray(array);
            array = nullptr;
        }
    }

    void getHostData(T** data) const {
        if (!isCreated()) {
            return;
        }

        cudaMemcpy3DParms copyParams = { 0 };
        copyParams.srcArray = array;
        copyParams.dstPtr.ptr = *data;
        copyParams.dstPtr.pitch = extent.width * sizeof(T);
        copyParams.dstPtr.xsize = extent.width;
        copyParams.dstPtr.ysize = extent.height;
        copyParams.extent = extent;
        copyParams.kind = cudaMemcpyDeviceToHost;
        checkCudaError(cudaMemcpy3D(&copyParams));
    }

private:
    cudaArray_t array;
    CudaArrayFlag flags;

    cudaExtent extent;
    cudaChannelFormatDesc desc;
};

} // namespace CUDA

#endif
