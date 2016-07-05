#ifndef ADVECT_HPP
#define ADVECT_HPP

namespace Kernel {

void advect3D(cudaTextureObject_t speedIn,
              cudaSurfaceObject_t speedOut,
              cudaExtent extent);

void advectBackward3D(cudaTextureObject_t speedIn,
                      cudaSurfaceObject_t speedOut,
                      cudaExtent extent);

void advectMacCormack3D(cudaTextureObject_t speedIn,
                        cudaTextureObject_t speedAIn,
                        cudaSurfaceObject_t speedOut,
                        cudaExtent extent);

} // namespace Kernel

#endif