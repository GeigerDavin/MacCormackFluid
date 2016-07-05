#include "../StdAfx.hpp"
#include "Divergence.hpp"

namespace Kernel {

__constant__ float4 volumeSize;

__global__ void divergence3DKernel(cudaTextureObject_t speedIn,
                                   cudaSurfaceObject_t divergenceOut) {
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;
    uint z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= volumeSize.x / 4) {
        return;
    }

    float pxm, pxp, pym, pyp, pzm, pzp;
    float div[4];

    #pragma unroll
    for (int j = 0; j < 4; j++) {
        uint3 i4 = make_uint3(4 * x + j, y, z);
        pxm = tex3D<float4>(speedIn, i4.x - 1, i4.y + 0, i4.z + 0).x;
        pxp = tex3D<float4>(speedIn, i4.x + 1, i4.y + 0, i4.z + 0).x;
        pym = tex3D<float4>(speedIn, i4.x + 0, i4.y - 1, i4.z + 0).y;
        pyp = tex3D<float4>(speedIn, i4.x + 0, i4.y + 1, i4.z + 0).y;
        pzm = tex3D<float4>(speedIn, i4.x + 0, i4.y + 0, i4.z - 1).z;
        pzp = tex3D<float4>(speedIn, i4.x + 0, i4.y + 0, i4.z + 1).z;
        div[j] = (pxp - pxm + pyp - pym + pzp - pzm) / 2;
    }

    float4 divergence = make_float4(div[0], div[1], div[2], div[3]);
    surf3Dwrite(divergence, divergenceOut, x, y, z);
}

} // namespace Kernel