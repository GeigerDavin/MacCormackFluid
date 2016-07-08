#ifndef JACOBI_HPP
#define JACOBI_HPP

namespace Kernel {

__global__ static void jacobi3D(cudaSurfaceObject_t pressureInSurf,
                                cudaSurfaceObject_t divergenceInSurf,
                                cudaSurfaceObject_t pressureOutSurf) {
    TID;
    if (tid.x >= deviceConstant.volumeSize.x / 4) {
        return;
    }

    uint3 j = make_uint3(0, 0, 0);

    j = make_uint3(max(tid.x - 1, 0), tid.y, tid.z);
    float4 pxm = make_float4(surf3Dread<float4>(pressureInSurf, j.x * sizeof(float4), j.y, j.z, CUDA_BOUNDARY_MODE).w,
                             surf3Dread<float4>(pressureInSurf, (tid.x + 0) * sizeof(float4), (tid.y + 0), (tid.z + 0), CUDA_BOUNDARY_MODE).x,
                             surf3Dread<float4>(pressureInSurf, (tid.x + 0) * sizeof(float4), (tid.y + 0), (tid.z + 0), CUDA_BOUNDARY_MODE).y,
                             surf3Dread<float4>(pressureInSurf, (tid.x + 0) * sizeof(float4), (tid.y + 0), (tid.z + 0), CUDA_BOUNDARY_MODE).z);
    j = make_uint3(min(tid.x + 1, (uint) deviceConstant.volumeSize.x - 1), tid.y, tid.z);
    float4 pxp = make_float4(surf3Dread<float4>(pressureInSurf, (tid.x + 0) * sizeof(float4), (tid.y + 0), (tid.z + 0), CUDA_BOUNDARY_MODE).y,
                             surf3Dread<float4>(pressureInSurf, (tid.x + 0) * sizeof(float4), (tid.y + 0), (tid.z + 0), CUDA_BOUNDARY_MODE).z,
                             surf3Dread<float4>(pressureInSurf, (tid.x + 0) * sizeof(float4), (tid.y + 0), (tid.z + 0), CUDA_BOUNDARY_MODE).w,
                             surf3Dread<float4>(pressureInSurf, j.x * sizeof(float4), j.y, j.z, CUDA_BOUNDARY_MODE).x);

    j = make_uint3(tid.x, max(tid.y - 1, 0), tid.z);
    float4 pym = surf3Dread<float4>(pressureInSurf, j.x * sizeof(float4), j.y, j.z, CUDA_BOUNDARY_MODE);
    j = make_uint3(tid.x, min(tid.y + 1, (uint) deviceConstant.volumeSize.y - 1), tid.z);
    float4 pyp = surf3Dread<float4>(pressureInSurf, j.x * sizeof(float4), j.y, j.z, CUDA_BOUNDARY_MODE);

    j = make_uint3(tid.x, tid.y, max(tid.z - 1, 0));
    float4 pzm = surf3Dread<float4>(pressureInSurf, j.x * sizeof(float4), j.y, j.z, CUDA_BOUNDARY_MODE);
    j = make_uint3(tid.x, tid.y, min(tid.z + 1, (uint) deviceConstant.volumeSize.z - 1));
    float4 pzp = surf3Dread<float4>(pressureInSurf, j.x * sizeof(float4), j.y, j.z, CUDA_BOUNDARY_MODE);

    float4 divergence = surf3Dread<float4>(divergenceInSurf, tid.x * sizeof(float4), tid.y, tid.z, CUDA_BOUNDARY_MODE);
    float4 pressure = (pxp + pxm + pyp + pym + pzp + pzm - divergence) / 6;

    surf3Dwrite(pressure, pressureOutSurf, tid.x * sizeof(float4), tid.y, tid.z, CUDA_BOUNDARY_MODE);
}

} // namespace Kernel

#endif