#include "../StdAfx.hpp"
#include "Advect.hpp"

namespace Kernel {

__constant__ float4 volumeSize;

__global__ void advect3DKernel(cudaTextureObject_t speedIn,
                               cudaSurfaceObject_t speedOut,
                               cudaExtent extent);

__global__ void advectBackward3DKernel(cudaTextureObject_t speedIn,
                                       cudaSurfaceObject_t speedOut,
                                       cudaExtent extent);

__global__ void advectMacCormack3DKernel(cudaTextureObject_t speedIn,
                                         cudaTextureObject_t speedAIn,
                                         cudaSurfaceObject_t speedOut,
                                         cudaExtent extent);


__global__ void advect3DKernel(cudaTextureObject_t speedIn,
                               cudaSurfaceObject_t speedOut,
                               cudaExtent extent) {
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;
    uint z = blockIdx.z * blockDim.z + threadIdx.z;

    if ((x >= extent.width) || (y >= extent.height) || (z >= extent.depth)) {
        return;
    }

    float4 speed = tex3D<float4>(speedIn, x, y, z);
    float samX = x - speed.x;
    float samY = y - speed.y;
    float samZ = z - speed.z;
    float4 sam = make_float4(samX, samY, samZ, 0.0f);
    sam = (sam + 0.5f) / volumeSize;

    float4 sampledSpeed = tex3DLod<float4>(speedIn, sam.x, sam.y, sam.z, 0);

    surf3Dwrite(sampledSpeed, speedOut, x, y, z);
}

__global__ void advectBackward3DKernel(cudaTextureObject_t speedIn,
                                       cudaSurfaceObject_t speedOut,
                                       cudaExtent extent) {
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;
    uint z = blockIdx.z * blockDim.z + threadIdx.z;

    if ((x >= extent.width) || (y >= extent.height) || (z >= extent.depth)) {
        return;
    }

    float4 speed = tex3D<float4>(speedIn, x, y, z);
    float samX = x + speed.x;
    float samY = y + speed.y;
    float samZ = z + speed.z;
    float4 sam = make_float4(samX, samY, samZ, 0.0f);
    sam = (sam + 0.5f) / volumeSize;

    float4 sampledSpeed = tex3DLod<float4>(speedIn, sam.x, sam.y, sam.z, 0);

    surf3Dwrite(sampledSpeed, speedOut, x, y, z);
}

__global__ void advectMacCormack3DKernel(cudaTextureObject_t speedIn,
                                         cudaTextureObject_t speedAIn,
                                         cudaSurfaceObject_t speedOut,
                                         cudaExtent extent) {
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;
    uint z = blockIdx.z * blockDim.z + threadIdx.z;

    if ((x >= extent.width) || (y >= extent.height) || (z >= extent.depth)) {
        return;
    }

    float4 speed = tex3D<float4>(speedIn, x, y, z);
    float samX = x - speed.x;
    float samY = y - speed.y;
    float samZ = z - speed.z;
    float4 sam = make_float4(samX, samY, samZ, 0.0f);

    uint3 j = make_uint3(sam.x, sam.y, sam.z);

    sam = (sam + 0.5f) / volumeSize;

    float4 r0 = tex3D<float4>(speedIn, j.x + 0, j.y + 0, j.z + 0);
    float4 r1 = tex3D<float4>(speedIn, j.x + 1, j.y + 0, j.z + 0);
    float4 r2 = tex3D<float4>(speedIn, j.x + 0, j.y + 1, j.z + 0);
    float4 r3 = tex3D<float4>(speedIn, j.x + 1, j.y + 1, j.z + 0);
    float4 r4 = tex3D<float4>(speedIn, j.x + 0, j.y + 0, j.z + 1);
    float4 r5 = tex3D<float4>(speedIn, j.x + 1, j.y + 0, j.z + 1);
    float4 r6 = tex3D<float4>(speedIn, j.x + 0, j.y + 1, j.z + 1);
    float4 r7 = tex3D<float4>(speedIn, j.x + 1, j.y + 1, j.z + 1);

    float4 lmin = fminf(r0, fminf(r1, fminf(r2, fminf(r3, fminf(r4, fminf(r5, fminf(r6, r7)))))));
    float4 lmax = fmaxf(r0, fmaxf(r1, fmaxf(r2, fmaxf(r3, fmaxf(r4, fmaxf(r5, fmaxf(r6, r7)))))));

    float4 sampledSpeed0 = tex3DLod<float4>(speedIn, sam.x, sam.y, sam.z, 0);

    float4 sampledSpeed = sampledSpeed0 + 0.5f * (sampledSpeed0 -
        tex3DLod<float4>(speedAIn, sam.x, sam.y, sam.z, 0));

    sampledSpeed = clamp(sampledSpeed, lmin, lmax);
    sampledSpeed.w = max(sampledSpeed0.w - 0.001f, 0.0f);

    surf3Dwrite(sampledSpeed, speedOut, x, y, z);
}

} // namespace Kernel