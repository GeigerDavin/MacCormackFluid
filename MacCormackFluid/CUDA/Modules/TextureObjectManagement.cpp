#include "../../StdAfx.hpp"
#include "TextureObjectManagement.hpp"
#include "ErrorHandling.hpp"
#include "DeviceManagement.hpp"

namespace CUDA {
namespace TextureObjectManagement {

void createTextureObject(cudaTextureObject_t* tex,
                         const cudaResourceDesc* resDesc,
                         const cudaTextureDesc* texDesc,
                         const cudaResourceViewDesc* resViewDesc) {
    if (useCuda) {
        checkCudaError(cudaCreateTextureObject(tex, resDesc, texDesc, resViewDesc));
    }
}

void destroyTextureObject(cudaTextureObject_t tex) {
    if (useCuda) {
        checkCudaError(cudaDestroyTextureObject(tex));
    }
}

} // namespace TextureObjectManagement
} // namespace CUDA