#ifndef CUDA_SURFACE_OBJECT_MANAGEMENT_HPP
#define CUDA_SURFACE_OBJECT_MANAGEMENT_HPP

namespace CUDA {
namespace SurfaceObjectManagement {

void createSurfaceObject(cudaSurfaceObject_t* surf, const cudaResourceDesc* resDesc);

void destroySurfaceObject(cudaSurfaceObject_t surf);

const cudaResourceDesc getResourceDesc(cudaSurfaceObject_t surf);

} // namespace SurfaceObjectManagement 
} // namespace CUDA

#endif