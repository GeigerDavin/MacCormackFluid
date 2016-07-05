#ifndef CUDA_OPENGL_INTEROPERABILITY_HPP
#define CUDA_OPENGL_INTEROPERABILITY_HPP

namespace CUDA {
namespace OpenGLInteroperability {

void registerGraphicsBuffer
    (cudaGraphicsResource** resource, GLuint bufferId, uint flags);

void registerGraphicsImage
    (cudaGraphicsResource** resource, GLuint image, GLenum target, uint flags);

void getDevices
    (uint* deviceCounts, int* devices, uint deviceCount, cudaGLDeviceList deviceList);

} // namespace OpenGLInteroperability
} // namespace CUDA

#endif

