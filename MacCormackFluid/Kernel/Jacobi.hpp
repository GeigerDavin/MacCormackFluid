#ifndef JACOBI_HPP
#define JACOBI_HPP

namespace Kernel {

// Jacobi iteration to make gradient pressure equal to speed (before it has divergence 0)
__global__ static void jacobi3D(cudaSurfaceObject_t pressureInSurf,
                                cudaSurfaceObject_t divergenceInSurf,
                                cudaSurfaceObject_t pressureOutSurf) {
    TID_CONST;
    if (tid.x >= volumeSizeDev.x / 4) {
        return;
    }

    float4 p = surf3Dread<float4>(pressureInSurf, tid.x * sizeof(float4), tid.y, tid.z, CUDA_BOUNDARY_MODE);

    uint3 j = make_uint3(0, 0, 0);

    j = make_uint3(max(tid.x - 1, 0), tid.y, tid.z);
    float4 pxm = make_float4(surf3Dread<float4>(pressureInSurf, j.x * sizeof(float4), j.y, j.z, CUDA_BOUNDARY_MODE).w, p.x, p.y, p.z);
    j = make_uint3(min(tid.x + 1, (uint) volumeSizeDev.x - 1), tid.y, tid.z);
    float4 pxp = make_float4(p.y, p.z, p.w, surf3Dread<float4>(pressureInSurf, j.x * sizeof(float4), j.y, j.z, CUDA_BOUNDARY_MODE).x);

    j = make_uint3(tid.x, max(tid.y - 1, 0), tid.z);
    float4 pym = surf3Dread<float4>(pressureInSurf, j.x * sizeof(float4), j.y, j.z, CUDA_BOUNDARY_MODE);
    j = make_uint3(tid.x, min(tid.y + 1, (uint) volumeSizeDev.y - 1), tid.z);
    float4 pyp = surf3Dread<float4>(pressureInSurf, j.x * sizeof(float4), j.y, j.z, CUDA_BOUNDARY_MODE);

    j = make_uint3(tid.x, tid.y, max(tid.z - 1, 0));
    float4 pzm = surf3Dread<float4>(pressureInSurf, j.x * sizeof(float4), j.y, j.z, CUDA_BOUNDARY_MODE);
    j = make_uint3(tid.x, tid.y, min(tid.z + 1, (uint) volumeSizeDev.z - 1));
    float4 pzp = surf3Dread<float4>(pressureInSurf, j.x * sizeof(float4), j.y, j.z, CUDA_BOUNDARY_MODE);

    float4 divergence = surf3Dread<float4>(divergenceInSurf, tid.x * sizeof(float4), tid.y, tid.z, CUDA_BOUNDARY_MODE);
    float4 pressure = (pxp + pxm + pyp + pym + pzp + pzm - divergence) / 6;

    surf3Dwrite(pressure, pressureOutSurf, tid.x * sizeof(float4), tid.y, tid.z, CUDA_BOUNDARY_MODE);
}

} // namespace Kernel

#endif
