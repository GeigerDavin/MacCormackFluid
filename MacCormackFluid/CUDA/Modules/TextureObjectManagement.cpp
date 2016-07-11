#include "../../StdAfx.hpp"
#include "TextureObjectManagement.hpp"
#include "ErrorHandling.hpp"

namespace CUDA {
namespace TextureObjectManagement {

void createTextureObject(cudaTextureObject_t* tex,
                         const cudaResourceDesc* resDesc,
                         const cudaTextureDesc* texDesc,
                         const cudaResourceViewDesc* resViewDesc) {
    if (Ctx->isCreated()) {
        checkCudaError(cudaCreateTextureObject(tex, resDesc, texDesc, resViewDesc));
    }
}

void destroyTextureObject(cudaTextureObject_t tex) {
    if (Ctx->isCreated()) {
        checkCudaError(cudaDestroyTextureObject(tex));
    }
}

} // namespace TextureObjectManagement
} // namespace CUDA