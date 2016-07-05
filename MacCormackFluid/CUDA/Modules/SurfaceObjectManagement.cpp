#include "../../StdAfx.hpp"
#include "SurfaceObjectManagement.hpp"
#include "ErrorHandling.hpp"
#include "DeviceManagement.hpp"

namespace CUDA {
namespace SurfaceObjectManagement {

void createSurfaceObject(cudaSurfaceObject_t* surf, const cudaResourceDesc* resDesc) {
    if (useCuda) {
        checkCudaError(cudaCreateSurfaceObject(surf, resDesc));
    }
}

void destroySurfaceObject(cudaSurfaceObject_t surf) {
    if (useCuda) {
        checkCudaError(cudaDestroySurfaceObject(surf));
    }
}

const cudaResourceDesc getResourceDesc(cudaSurfaceObject_t surf) {
    return cudaResourceDesc();
}

} // namespace SurfaceObjectManagement 
} // namespace CUDA