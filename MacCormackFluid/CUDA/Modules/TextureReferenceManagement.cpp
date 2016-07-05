#include "../../StdAfx.hpp"
#include "TextureReferenceManagement.hpp"
#include "ErrorHandling.hpp"
#include "DeviceManagement.hpp"

namespace CUDA {
namespace TextureReferenceManagement {

void bindTexture(const textureReference* tex,
                 const void* devPtr,
                 const cudaChannelFormatDesc* desc,
                 size_t* offset,
                 size_t size) {
    if (useCuda && tex && devPtr && desc) {
        checkCudaError(cudaBindTexture(offset, tex, devPtr, desc, size));
    }
}

void bindTexture2D(const textureReference* tex,
                   const void* devPtr,
                   const cudaChannelFormatDesc* desc,
                   size_t* offset,
                   size_t width,
                   size_t height,
                   size_t pitch) {
    if (useCuda && tex && devPtr && desc) {
        checkCudaError(cudaBindTexture2D(offset, tex, devPtr, desc, width, height, pitch));
    }
}

void bindTextureToArray(const textureReference* tex,
                        cudaArray_const_t array,
                        const cudaChannelFormatDesc* desc) {
    if (useCuda && tex && array && desc) {
        checkCudaError(cudaBindTextureToArray(tex, array, desc));
    }
}

void bindTextureToMipmappedArray(const textureReference* tex,
                                 cudaMipmappedArray_const_t mipmappedArray,
                                 const cudaChannelFormatDesc* desc) {
    if (useCuda && tex && mipmappedArray && desc) {
        checkCudaError(cudaBindTextureToMipmappedArray(tex, mipmappedArray, desc));
    }
}

cudaChannelFormatDesc createChannelDesc(int x, int y, int z, int w,
                                        cudaChannelFormatKind f) {
    if (useCuda) {
        return cudaCreateChannelDesc(x, y, z, w, f);
    }
    return cudaChannelFormatDesc(); // ? Replace with pointer ?
}

cudaChannelFormatDesc getChannelDesc(cudaArray_const_t array) {
    cudaChannelFormatDesc desc;
    if (useCuda) {
        checkCudaError(cudaGetChannelDesc(&desc, array));
    }
    return desc;
}

size_t getTextureAlignmentOffset(const textureReference* tex) {
    size_t offset = 0;
    if (useCuda) {
        checkCudaError(cudaGetTextureAlignmentOffset(&offset, tex));
    }
    return offset;
}

const textureReference* getTextureReference(const void* symbol) {
    const textureReference* tex = nullptr;
    if (useCuda) {
        checkCudaError(cudaGetTextureReference(&tex, symbol));
    }
    return tex;
}

void unbindTexture(const textureReference* tex) {
    if (useCuda) {
        checkCudaError(cudaUnbindTexture(tex));
    }
}

} // namespace TextureReferenceManagement
} // namespace CUDA