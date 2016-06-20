#include "../StdAfx.hpp"
#include "../CUDA/Auxiliary.hpp"
#include "../CUDA/KernelTextures.hpp"

namespace Kernel {

__global__ void cubeKernel(GLfloat* vertices, GLfloat* colors, size_t size) {
    TIDX(size);
    if (tidX >= size) {
        printf("%d\n", tidX);
    }
    if (colors[tidX] >= 1.0f) {
        colors[tidX] = 0.0f;
    }
    colors[tidX] += 0.01f;
}

void computeCube(Utils::Vector<GLfloat>& vertices, Utils::Vector<GLfloat>& colors) {
    if (vertices.getSize() != colors.getSize()) {
        std::cerr << "Buffer sizes don't match" << std::endl;
        return;
    }
    const size_t size = vertices.getSize();

    cubeKernel<<<CUDA::getGridDim1D(size), THREADS>>>(vertices.bindGraphicsResource(),
                 colors.bindGraphicsResource(),
                 vertices.getSize());

    vertices.unbindGraphicsResource();
    colors.unbindGraphicsResource();
    ERRORCHECK_CUDA();
}

__global__ void advect3D(unsigned int size) {
    uint3 tid;
    tid.x = blockIdx.x * blockDim.x + threadIdx.x;
    tid.y = blockIdx.y * blockDim.y + threadIdx.y;
    tid.z = blockIdx.z * blockDim.z + threadIdx.z;

    float4 voxel = tex3D(CUDA::speed, tid.x, tid.y, tid.z);
    float3 sam = make_float3(tid.x - voxel.x, tid.y - voxel.y, tid.z - voxel.z);

    const float4 dim = make_float4(200, 200, 200, 0);
    sam.x += 0.5f;
    sam.y += 0.5f;
    sam.z += 0.5f;
    sam.x /= dim.x;
    sam.y /= dim.y;
    sam.z /= dim.z;

    float4 voxelLod = tex3D(CUDA::speed, sam.x, sam.y, sam.z);
}

} // namespace Kernel