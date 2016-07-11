#include "../../StdAfx.hpp"
#include "OpenGLInteroperability.hpp"
#include "ErrorHandling.hpp"

namespace CUDA {
namespace OpenGLInteroperability {

void registerGraphicsBuffer
    (cudaGraphicsResource** graphicsResource, GLuint bufferId, unsigned int flags) {
    if (Ctx->isCreated()) {
        checkCudaError(cudaGraphicsGLRegisterBuffer(graphicsResource, bufferId, flags));
    }
}

void registerGraphicsImage
    (cudaGraphicsResource** resource, GLuint image, GLenum target, uint flags) {

}

} // namespace OpenGLInteroperability
} // namespace CUDA