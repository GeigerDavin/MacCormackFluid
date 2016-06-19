#include "Auxiliary.hpp"
#include "Allocator.hpp"
#include "KernelTextures.hpp"
#include "VectorTypes.hpp"

#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <device_launch_parameters.h>

__global__ void advect3D(unsigned int size) {
    uint3 tid;
    tid.x = blockIdx.x * blockDim.x + threadIdx.x;
    tid.y = blockIdx.y * blockDim.y + threadIdx.y;
    tid.z = blockIdx.z * blockDim.z + threadIdx.z;

    float4 voxel = tex3D(speed, tid.x, tid.y, tid.z);
    float3 sam = make_float3(tid.x - voxel.x, tid.y - voxel.y, tid.z - voxel.z);

    const float4 dim = make_float4(200, 200, 200, 0);
    sam.x += 0.5f;
    sam.y += 0.5f;
    sam.z += 0.5f;
    sam.x /= dim.x;
    sam.y /= dim.y;
    sam.z /= dim.z;

    float4 voxelLod = tex3D(speed, sam.x, sam.y, sam.z);
}