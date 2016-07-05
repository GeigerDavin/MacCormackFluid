#include "../../StdAfx.hpp"
#include "GraphicsInteroperability.hpp"
#include "ErrorHandling.hpp"
#include "DeviceManagement.hpp"

namespace CUDA {
namespace GraphicsInteroperability {

void setGraphicsResourceMapFlags
    (cudaGraphicsResource_t resource, uint flags) {
    if (useCuda) {
        checkCudaError(cudaGraphicsResourceSetMapFlags(resource, flags));
    }
}

void* mapGraphicsResourcePointer
    (cudaGraphicsResource_t* resource, cudaStream_t stream) {
    void* ptr = nullptr;
    size_t numBytes = 0;
    if (useCuda) {
        checkCudaError(cudaGraphicsMapResources(1, resource, stream));
        checkCudaError(cudaGraphicsResourceGetMappedPointer((void **) &ptr,
                        &numBytes, *resource));
    }
    return ptr;
}

cudaMipmappedArray_const_t mapGraphicsResourceMipmappedArray
    (cudaGraphicsResource_t* resource, cudaStream_t stream) {
     cudaMipmappedArray_t mipmappedArray = nullptr;
    if (useCuda) {
        checkCudaError(cudaGraphicsMapResources(1, resource, stream));
        checkCudaError(cudaGraphicsResourceGetMappedMipmappedArray(&mipmappedArray,
                        *resource));
    }
    return mipmappedArray;
}

cudaArray_t mapGraphicsResourceSubArray
    (cudaGraphicsResource_t* resource, uint arrayIndex, uint mipLevel,
     cudaStream_t stream) {
     cudaArray_t array = nullptr;
     if (useCuda) {
         checkCudaError(cudaGraphicsMapResources(1, resource, stream));
         checkCudaError(cudaGraphicsSubResourceGetMappedArray(&array, *resource,
                        arrayIndex, mipLevel));
     }
     return array;
}

void unregisterGraphicsResource
    (cudaGraphicsResource_t* resource) {
    if (useCuda) {
        checkCudaError(cudaGraphicsUnregisterResource(*resource));
    }
    *resource = nullptr;
}

void unmapGraphicsResource
    (cudaGraphicsResource_t resource, cudaStream_t stream) {
    if (useCuda) {
        checkCudaError(cudaGraphicsUnmapResources(1, &resource, stream));
    }
}

} // namespace GraphicsInteroperability
} // namespace CUDA