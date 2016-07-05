#include "../../StdAfx.hpp"
#include "MemoryManagement.hpp"
#include "ErrorHandling.hpp"
#include "DeviceManagement.hpp"

namespace CUDA {
namespace MemoryManagement {

#define cudaMemCheckHost(ptr)                                                   \
    ErrorHandling::_cudaMemCheck(ptr, "Host",               __FILE__, __LINE__)
#define cudaMemCheckDevice(ptr)                                                 \
    ErrorHandling::_cudaMemCheck(ptr, "Device",             __FILE__, __LINE__)
#define cudaMemCheckManaged(ptr)                                                \
    ErrorHandling::_cudaMemCheck(ptr, "Managed",            __FILE__, __LINE__)
#define cudaMemCheckPitch(ptr)                                                  \
    ErrorHandling::_cudaMemCheck(ptr, "Pitch",              __FILE__, __LINE__)
#define cudaMemCheck3D(ptr)                                                     \
    ErrorHandling::_cudaMemCheck(ptr, "3D",                 __FILE__, __LINE__)
#define cudaMemCheckArray(ptr)                                                  \
    ErrorHandling::_cudaMemCheck(ptr, "CudaArray",          __FILE__, __LINE__)
#define cudaMemCheck3DArray(ptr)                                                \
    ErrorHandling::_cudaMemCheck(ptr, "Cuda3DArray",        __FILE__, __LINE__)
#define cudaMemCheckMipmappedArray(ptr)                                         \
    ErrorHandling::_cudaMemCheck(ptr, "CudaMipmappedArray", __FILE__, __LINE__)

void getMemInfo(size_t* free, size_t* total) {
    if (useCuda) {
        cudaMemGetInfo(free, total);
    }
}

void* mallocHost(size_t size) {
    if (size == 0) {
        return nullptr;
    }
    void* ptr = nullptr;
    if (useCuda) {
        checkCudaError(cudaMallocHost(&ptr, size));
    } else {
        ptr = malloc(size);
    }
    cudaMemCheckHost(ptr);
    memset(ptr, 0, size);
    return ptr;
}

void* mallocHost(size_t size, uint flags) {
    if (size == 0) {
        return nullptr;
    }
    void* ptr = nullptr;
    if (useCuda) {
        checkCudaError(cudaHostAlloc(&ptr, size, flags));
    } else {
        ptr = malloc(size);
    }
    cudaMemCheckHost(ptr);
    memset(ptr, 0, size);
    return ptr;
}

void* reallocHost(void* ptr, size_t size) {
    freeHost(ptr);
    return mallocHost(size);
}

void* reallocHost(void* ptr, size_t oldSize, size_t newSize) {
    void* buffer = mallocHost(newSize);
    if (ptr) {
        moveHostToHost(buffer, ptr, oldSize);
        freeHost(ptr);
    }
    return buffer;
}

void freeHost(void* ptr) {
    if (!ptr) {
        return;
    }
    if (useCuda) {
        checkCudaError(cudaFreeHost(ptr));
    } else {
        free(ptr);
    }
}

void* mallocManaged(size_t size) {
    if (size == 0) {
        return nullptr;
    }
    void* ptr = nullptr;
    if (useCuda) {
        checkCudaError(cudaMallocManaged(&ptr, size));
        cudaMemCheckManaged(ptr);
    }
    return ptr;
}

void* reallocManaged(void* ptr, size_t size) {
    freeManaged(ptr);
    return mallocManaged(size);
}

void freeManaged(void* ptr) {
    if (!ptr) {
        return;
    }
    if (useCuda) {
        checkCudaError(cudaFree(ptr));
    }
}

void* mallocDevice(size_t size) {
    if (size == 0) {
        return nullptr;
    }
    void* ptr = nullptr;
    if (useCuda) {
        checkCudaError(cudaMalloc(&ptr, size));
        cudaMemCheckDevice(ptr);
        deviceMemset(ptr, 0, size);
    }
    return ptr;
}

void* reallocDevice(void* ptr, size_t size) {
    freeDevice(ptr);
    return mallocDevice(size);
}

void* reallocDevice(void* ptr, size_t oldSize, size_t newSize) {
    void* buffer = mallocDevice(newSize);
    if (ptr) {
        moveDeviceToDevice(buffer, ptr, oldSize);
        freeDevice(ptr);
    }
    return buffer;
}

void freeDevice(void* ptr) {
    if (!ptr) {
        return;
    }
    if (useCuda) {
        checkCudaError(cudaFree(ptr));
    }
}

void* mallocPitch(size_t* pitch, size_t width, size_t height) {
    void* ptr = nullptr;
    if (useCuda) {
        checkCudaError(cudaMallocPitch(&ptr, pitch, width, height));
        cudaMemCheckPitch(ptr);
        deviceMemset2D(ptr, *pitch, 0, width, height);
    }
    return ptr;
}

void* reallocPitch(void* ptr, size_t* pitch, size_t width, size_t height) {
    return nullptr;
}

void* reallocPitch(void* ptr, size_t* pitch, size_t oldWidth, size_t oldHeight,
                   size_t newWidth, size_t newHeight) {
    return nullptr;
}

void freePitch(void* ptr) {
    if (!ptr) {
        return;
    }
    if (useCuda) {
        checkCudaError(cudaFree(ptr));
    }
}

cudaPitchedPtr malloc3D(cudaExtent extent) {
    cudaPitchedPtr ptr;
    memset(&ptr, 0, sizeof(ptr));
    if (useCuda) {
        checkCudaError(cudaMalloc3D(&ptr, extent));
        cudaMemCheck3D(ptr.ptr);
        deviceMemset3D(ptr, 0, extent);
    }
    return ptr;
}

cudaPitchedPtr realloc3D(cudaPitchedPtr ptr, cudaExtent extent) {
    return cudaPitchedPtr();
}

cudaPitchedPtr realloc3D(cudaPitchedPtr ptr, cudaExtent oldExtent, cudaExtent newExtent) {
    return cudaPitchedPtr();
}

void free3D(cudaPitchedPtr ptr) {
    if (!ptr.ptr) {
        return;
    }
    if (useCuda) {
        checkCudaError(cudaFree(ptr.ptr));
    }
}

cudaArray_t mallocArray(const cudaChannelFormatDesc* desc, size_t width,
                        size_t height, uint flags) {
    if (!desc) {
        return nullptr;
    }
    cudaArray_t array = nullptr;
    if (useCuda) {
        checkCudaError(cudaMallocArray(&array, desc, width, height, flags));
        cudaMemCheckArray(array);
    }
    return array;
}

cudaArray_t reallocArray(cudaArray_t array, size_t width, size_t height) {
    return nullptr;
}

cudaArray_t reallocArray(cudaArray_t array, size_t oldWidth, size_t newWidth,
                         size_t oldHeight, size_t newHeight) {
    return nullptr;
}

void freeArray(cudaArray_t array) {
    if (!array) {
        return;
    }
    if (useCuda) {
        checkCudaError(cudaFreeArray(array));
    }
}

cudaArray_t malloc3DArray(const cudaChannelFormatDesc* desc, cudaExtent extent, uint flags) {
    if (!desc) {
        return nullptr;
    }
    cudaArray_t array = nullptr;
    if (useCuda) {
        checkCudaError(cudaMalloc3DArray(&array, desc, extent, flags));
        cudaMemCheck3DArray(array);
    }
    return array;
}

cudaArray_t realloc3DArray(cudaArray_t array, cudaExtent extent) {
    return nullptr;
}

cudaArray_t realloc3DArray(cudaArray_t array, cudaExtent oldExtent, cudaExtent newExtent) {
    return nullptr;
}

void free3DArray(cudaArray_t array) {
    if (!array) {
        return;
    }
    if (useCuda) {
        checkCudaError(cudaFreeArray(array));
    }
}

cudaMipmappedArray_t mallocMipmappedArray
    (const cudaChannelFormatDesc* desc, cudaExtent extent, uint numLevels, uint flags) {
    if (!desc) {
        return nullptr;
    }
    cudaMipmappedArray_t mipmappedArray = nullptr;
    if (useCuda) {
        checkCudaError(cudaMallocMipmappedArray(&mipmappedArray, desc, extent,
                                                numLevels, flags));
        cudaMemCheckMipmappedArray(mipmappedArray);
    }
    return mipmappedArray;
}

cudaMipmappedArray_t reallocMipmappedArray
    (cudaMipmappedArray_t mipmappedArray, cudaExtent extent) {
    return nullptr;
}

cudaMipmappedArray_t reallocMipmappedArray
    (cudaMipmappedArray_t mipmappedArray, cudaExtent oldExtent, cudaExtent newExtent) {
    return nullptr;
}

void freeMipmappedArray(cudaMipmappedArray_t mipmappedArray) {
    if (!mipmappedArray) {
        return;
    }
    if (useCuda) {
        checkCudaError(cudaFreeMipmappedArray(mipmappedArray));
    }
}

cudaArray_t getMipmappedArrayLevel(cudaMipmappedArray_const_t mipmappedArray, uint level) {
    if (!mipmappedArray) {
        return nullptr;
    }
    cudaArray_t levelArray = nullptr;
    checkCudaError(cudaGetMipmappedArrayLevel(&levelArray, mipmappedArray, level));
    return levelArray;
}

void getArrayInfo(cudaChannelFormatDesc* desc, cudaExtent* extent, uint* flags, cudaArray_t array) {
    if (array) {
        checkCudaError(cudaArrayGetInfo(desc, extent, flags, array));
    }
}

void moveHostToHost(void* dst, const void* src, size_t count) {
    if (!dst || !src) {
        return;
    }
    if (useCuda) {
        checkCudaError(cudaMemcpy(dst, src, count, cudaMemcpyHostToHost));
    } else {
        memcpy(dst, src, count);
    }
}

void moveHostToDevice(void* dst, const void* src, size_t count) {
    if (!useCuda || !dst || !src) {
        return;
    }
    checkCudaError(cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice));
}

void moveDeviceToHost(void* dst, const void* src, size_t count) {
    if (!useCuda || !dst || !src) {
        return;
    }
    checkCudaError(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost));
}

void moveDeviceToDevice(void* dst, const void* src, size_t count) {
    if (!useCuda || !dst || !src) {
        return;
    }
    checkCudaError(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToDevice));
}

void moveHostToHostAsync
    (void* dst, const void* src, size_t count, cudaStream_t stream) {
    if (!dst || !src) {
        return;
    }
    if (useCuda) {
        checkCudaError(cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToHost));
    } else {
        memcpy(dst, src, count);
    }
}

void moveHostToDeviceAsync
    (void* dst, const void* src, size_t count, cudaStream_t stream) {
    if (!useCuda || !dst || !src) {
        return;
    }
    checkCudaError(cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToDevice, stream));
}

void moveDeviceToHostAsync
    (void* dst, const void* src, size_t count, cudaStream_t stream) {
    if (!useCuda || !dst || !src) {
        return;
    }
    checkCudaError(cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToHost, stream));
}

void moveDeviceToDeviceAsync
    (void* dst, const void* src, size_t count, cudaStream_t stream) {
    if (!useCuda || !dst || !src) {
        return;
    }
    checkCudaError(cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice, stream));
}

void moveHostToHost2D
    (void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height) {
    if (!useCuda || !dst || !src) {
        return;
    }
    checkCudaError(cudaMemcpy2D(dst, dpitch, src, spitch, width, height, cudaMemcpyHostToHost));
}

void moveHostToDevice2D
    (void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height) {

}

void moveDeviceToHost2D
    (void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height) {

}

void moveDeviceToDevice2D
    (void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height) {

}

void moveHostToHost2DAsync
    (void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height,
     cudaStream_t stream) {

}

void moveHostToDevice2DAsync
    (void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height,
     cudaStream_t stream ) {

}

void moveDeviceToHost2DAsync
    (void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height,
     cudaStream_t stream) {

}

void moveDeviceToDevice2DAsync
    (void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height,
     cudaStream_t stream) {

}

void moveHostToSymbol(const void* symbol, const void* src, size_t count, size_t offset) {
    if (useCuda) {
        if (!symbol) {
            std::cerr << "Invalid symbol" << std::endl;
            return;
        }
        if (!src) {
            std::cerr << "Invalid source" << std::endl;
            return;
        }
        checkCudaError(cudaMemcpyToSymbol(symbol, src, count, offset, cudaMemcpyHostToDevice));
    }
}

void deviceMemset(void* ptr, int value, size_t count) {
    if (!useCuda || !ptr) {
        return;
    }
    checkCudaError(cudaMemset(ptr, value, count));
}

void deviceMemset2D(void* ptr, size_t pitch, int value, size_t width, size_t height) {
    if (!useCuda || !ptr) {
        return;
    }
    checkCudaError(cudaMemset2D(ptr, pitch, value, width, height));
}

void deviceMemset3D(cudaPitchedPtr ptr, int value, cudaExtent extent) {
    if (!useCuda || !ptr.ptr) {
        return;
    }
    checkCudaError(cudaMemset3D(ptr, value, extent));
}

void deviceMemsetAsync
    (void* ptr, int value, size_t count, cudaStream_t stream) {
    if (!useCuda || !ptr) {
        return;
    }
    checkCudaError(cudaMemsetAsync(ptr, value, count, stream));
}

void deviceMemset2DAsync
    (void* ptr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream) {
    if (!useCuda || !ptr) {
        return;
    }
    checkCudaError(cudaMemset2DAsync(ptr, pitch, value, width, height, stream));
}

void deviceMemset3DAsync
    (cudaPitchedPtr ptr, int value, cudaExtent extent, cudaStream_t stream) {
    if (!useCuda || !ptr.ptr) {
        return;
    }
    checkCudaError(cudaMemset3DAsync(ptr, value, extent, stream));
}

cudaPos createCudaPos(size_t x, size_t y, size_t z) {
    return make_cudaPos(x, y, z);
}

cudaExtent createCudaExtent(size_t width, size_t height, size_t depth) {
    return make_cudaExtent(width, height, depth);
}

cudaPitchedPtr createCudaPitchedPtr(void* ptr, size_t pitch, size_t width, size_t height) {
    return make_cudaPitchedPtr(ptr, pitch, width, height);
}

} // namespace MemoryManagement
} // namespace CUDA