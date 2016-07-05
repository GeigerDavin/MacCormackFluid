#ifndef RENDER_HPP
#define RENDER_HPP

namespace Kernel {

struct Constant;

void renderSphere();
void renderVolume(cudaTextureObject_t speedSizeIn,
                  uint* rgbaOut,
                  uint dx, uint dy, uint dz);

void copyToConstant(const Constant& c);

void project3D(cudaTextureObject_t pressureIn,
               cudaSurfaceObject_t speedSizeOut,
               cudaTextureObject_t speedSize,
               uint dx, uint dy, uint dz);

} // namespace Kernel

#endif