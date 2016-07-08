#ifndef DIVERGENCE_HPP
#define DIVERGENCE_HPP

namespace Kernel {

__global__ void divergence3D(cudaSurfaceObject_t speedInSurf,
                             cudaSurfaceObject_t divergenceOutSurf) {
    TID;
    if (tid.x >= deviceConstant.volumeSize.x / 4) {
        return;
    }
    
    float pxm, pxp, pym, pyp, pzm, pzp;
    float div[4];

    #pragma unroll
    for (int j = 0; j < 4; j++) {
        uint3 i4 = make_uint3(4 * tid.x + j, tid.y, tid.z);
        pxm = surf3Dread<float4>(speedInSurf, (i4.x - 1) * sizeof(float4), (i4.y + 0), (i4.z + 0), CUDA_BOUNDARY_MODE).x;
        pxp = surf3Dread<float4>(speedInSurf, (i4.x + 1) * sizeof(float4), (i4.y + 0), (i4.z + 0), CUDA_BOUNDARY_MODE).x;
        pym = surf3Dread<float4>(speedInSurf, (i4.x + 0) * sizeof(float4), (i4.y - 1), (i4.z + 0), CUDA_BOUNDARY_MODE).y;
        pyp = surf3Dread<float4>(speedInSurf, (i4.x + 0) * sizeof(float4), (i4.y + 1), (i4.z + 0), CUDA_BOUNDARY_MODE).y;
        pzm = surf3Dread<float4>(speedInSurf, (i4.x + 0) * sizeof(float4), (i4.y + 0), (i4.z - 1), CUDA_BOUNDARY_MODE).z;
        pzp = surf3Dread<float4>(speedInSurf, (i4.x + 0) * sizeof(float4), (i4.y + 0), (i4.z + 1), CUDA_BOUNDARY_MODE).z;
        div[j] = (pxp - pxm + pyp - pym + pzp - pzm) / 2;
    }

    float4 divergence = make_float4(div[0], div[1], div[2], div[3]);
    surf3Dwrite(divergence, divergenceOutSurf, tid.x * sizeof(float4), tid.y, tid.z, CUDA_BOUNDARY_MODE);
}

} // namespace Kernel

#endif