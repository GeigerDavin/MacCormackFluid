#ifndef CUDA_MIPMAPPED_ARRAY_HPP
#define CUDA_MIPMAPPED_ARRAY_HPP

#include "Modules/MemoryManagement.hpp"
#include "Modules/TextureReferenceManagement.hpp"

namespace CUDA {

template <class T>
struct CudaMipmappedArray {
    CudaMipmappedArray(cudaExtent ext, unsigned int levels, unsigned int f = 0)
        : extent(ext), numLevels(levels), flags(f)
        , desc(nullptr), mipmappedArray(nullptr) {

        desc = TextureReferenceManagement::createChannelDesc<T>();
        mipmappedArray = MemoryManagement::
    }


    cudaExtent extent;
    unsigned int flags;
    unsigned int numLevels;
    const cudaChannelFormatDesc* desc;
    cudaMipmappedArray_t mipmappedArray
};

} // namespace CUDA

#endif