#include "OpenGLInterop.hpp"
#include "Auxiliary.hpp"

//void registerGraphicsResource
//    (cudaGraphicsResource** graphicsResource, unsigned int bufferId) {
//    if (useCuda) {
//        cudaGraphicsGLRegisterBuffer(graphicsResource, bufferId,
//                                     cudaGraphicsRegisterFlagsNone);
//        ERRORCHECK_CUDA();
//    }
//}
//
//void unregisterGraphicsResource
//    (cudaGraphicsResource** graphicsResource) {
//    if (useCuda) {
//        cudaGraphicsUnregisterResource(*graphicsResource);
//        ERRORCHECK_CUDA();
//    }
//    *graphicsResource = nullptr;
//}
//
//void* mapGraphicsResource
//    (cudaGraphicsResource** graphicsResource) {
//    void* ptr = nullptr;
//    size_t numBytes = 0;
//    if (useCuda) {
//        cudaGraphicsMapResources(1, graphicsResource, 0);
//        ERRORCHECK_CUDA();
//        cudaGraphicsResourceGetMappedPointer((void **) &ptr, &numBytes,
//                                             *graphicsResource);
//        ERRORCHECK_CUDA();
//    }
//    return ptr;
//}
//
//void unmapGraphicsResource
//    (cudaGraphicsResource* graphicsResource) {
//    if (useCuda) {
//        cudaGraphicsUnmapResources(1, &graphicsResource, 0);
//        ERRORCHECK_CUDA();
//    }
//}