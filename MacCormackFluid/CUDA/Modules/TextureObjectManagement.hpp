#ifndef CUDA_TEXTURE_OBJECT_MANAGEMENT_HPP
#define CUDA_TEXTURE_OBJECT_MANAGEMENT_HPP

namespace CUDA {
namespace TextureObjectManagement {

void createTextureObject(cudaTextureObject_t* tex,
                         const cudaResourceDesc* resDesc,
                         const cudaTextureDesc* texDesc,
                         const cudaResourceViewDesc* resViewDesc);

void destroyTextureObject(cudaTextureObject_t tex);

const cudaResourceDesc getResourceDesc(cudaTextureObject_t tex);

const cudaResourceViewDesc getResourceViewDesc(cudaTextureObject_t tex);

const cudaTextureDesc getTextureDesc(cudaTextureObject_t tex);

} // namespace TextureObjectManagement
} // namespace CUDA

#endif