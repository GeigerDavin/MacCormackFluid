#include "../StdAfx.hpp"
#include "Jacobi.hpp"

namespace Kernel {

__constant__ float4 volumeSize;

__global__ void jacobi3DKernel(cudaTextureObject_t pressureIn,
                               cudaTextureObject_t divergenceIn,
                               cudaSurfaceObject_t pressureOut) {
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;
    uint z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= volumeSize.x / 4) {
        return;
    }

    uint3 j;

    j = make_uint3(max(x - 1, 0), y, z);
    float4 pxm = make_float4(tex3D<float4>(pressureIn, j.x, j.y, j.z).w,
                             tex3D<float4>(pressureIn, x + 0, y + 0, z + 0).x,
                             tex3D<float4>(pressureIn, x + 0, y + 0, z + 0).y,
                             tex3D<float4>(pressureIn, x + 0, y + 0, z + 0).z);
    //j = make_uint3(min(x + 1, volumeSize.x - 1), y, z);
    float4 pxp = make_float4(tex3D<float4>(pressureIn, x + 0, y + 0, z + 0).y,
                             tex3D<float4>(pressureIn, x + 0, y + 0, z + 0).z,
                             tex3D<float4>(pressureIn, x + 0, y + 0, z + 0).w,
                             tex3D<float4>(pressureIn, j.x, j.y, j.z).x);

    j = make_uint3(x, max(y - 1, 0), z);
    float4 pym = tex3D<float4>(pressureIn, j.x, j.y, j.z);
    //j = make_uint3(min(x + 1, volumeSize.x - 1), y, z);
    float4 pyp = tex3D<float4>(pressureIn, j.x, j.y, j.z);

    j = make_uint3(x, max(y - 1, 0), z);
    float4 pzm = tex3D<float4>(pressureIn, j.x, j.y, j.z);
    //j = make_uint3(min(x + 1, volumeSize.x - 1), y, z);
    float4 pzp = tex3D<float4>(pressureIn, j.x, j.y, j.z);

    float4 divergence = tex3D<float4>(divergenceIn, x, y, z);
    float4 pressure = (pxp + pxm + pyp + pym + pzp + pzm - divergence) / 6;

    surf3Dwrite(pressure, pressureOut, x, y, z);
}

} // namespace Kernel