#ifndef CUDA_GRAPHICS_INTEROPERABILITY_HPP
#define CUDA_GRAPHICS_INTEROPERABILITY_HPP

namespace CUDA {
namespace GraphicsInteroperability {

void setGraphicsResourceMapFlags
    (cudaGraphicsResource_t resource, uint flags);

void* mapGraphicsResourcePointer
    (cudaGraphicsResource_t* resource, cudaStream_t stream = nullptr);

cudaMipmappedArray_const_t mapGraphicsResourceMipmappedArray
    (cudaGraphicsResource_t* resource, cudaStream_t stream = nullptr);

cudaArray_t mapGraphicsResourceSubArray
    (cudaGraphicsResource_t* resource, uint arrayIndex, uint mipLevel,
     cudaStream_t stream = nullptr);

void unregisterGraphicsResource
    (cudaGraphicsResource_t* resource);

void unmapGraphicsResource
    (cudaGraphicsResource_t resource, cudaStream_t stream = nullptr);

} // namespace GraphicsInteroperability
} // namespace CUDA

#endif