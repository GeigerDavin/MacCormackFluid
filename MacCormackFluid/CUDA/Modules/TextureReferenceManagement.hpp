#ifndef CUDA_TEXTURE_REFERENCE_MANAGEMENT_HPP
#define CUDA_TEXTURE_REFERENCE_MANAGEMENT_HPP

#include "DeviceManagement.hpp"

namespace CUDA {
namespace TextureReferenceManagement {

void bindTexture(const textureReference* tex,
                 const void* devPtr,
                 const cudaChannelFormatDesc* desc,
                 size_t* offset,
                 size_t size);

void bindTexture2D(const textureReference* tex,
                   const void* devPtr,
                   const cudaChannelFormatDesc* desc,
                   size_t* offset,
                   size_t width,
                   size_t height,
                   size_t pitch);

void bindTextureToArray(const textureReference* tex,
                        cudaArray_const_t array,
                        const cudaChannelFormatDesc* desc);

void bindTextureToMipmappedArray(const textureReference* tex,
                                 cudaMipmappedArray_const_t mipmappedArray,
                                 const cudaChannelFormatDesc* desc);

template <class T>
cudaChannelFormatDesc createChannelDesc() {
    if (useCuda) {
        return cudaCreateChannelDesc<T>();
    }
    return cudaChannelFormatDesc(); // ? Replace with pointer ?
}

cudaChannelFormatDesc createChannelDesc(int x, int y, int z, int w,
                                        cudaChannelFormatKind f);

cudaChannelFormatDesc getChannelDesc(cudaArray_const_t array);

size_t getTextureAlignmentOffset(const textureReference* tex);

const textureReference* getTextureReference(const void* symbol);

void unbindTexture(const textureReference* tex);

} // namespace TextureReferenceManagement
} // namespace CUDA

#endif
