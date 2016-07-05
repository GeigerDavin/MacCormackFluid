#include "../StdAfx.hpp"
#include "Render.hpp"
#include "../CUDA/Modules/ErrorHandling.hpp"
#include "../CUDA/Modules/MemoryManagement.hpp"

#include "Constant.hpp"

namespace Kernel {

__device__ float3 mul(const float4x4& m, const float3& v) {
    float3 r;
    r.x = dot(v, make_float3(m.m[0]));
    r.y = dot(v, make_float3(m.m[1]));
    r.z = dot(v, make_float3(m.m[2]));
    return r;
}

__device__ float4 mul(const float4x4& m, const float4& v) {
    float4 r;
    r.x = dot(v, m.m[0]);
    r.y = dot(v, m.m[1]);
    r.z = dot(v, m.m[2]);
    r.w = 1.0f;
    return r;
}

__global__ void renderSphereKernel(cudaTextureObject_t speedIn,
                                   cudaSurfaceObject_t speedOut,
                                   cudaExtent extent) {
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;
    uint z = blockIdx.z * blockDim.z + threadIdx.z;

    if ((x >= extent.width) || (y >= extent.height) || (z >= extent.depth)) {
        return;
    }

    float3 eye = make_float3(0, 0, 4);
    float3 rayDir = make_float3(make_float2(2 * constDev.mouse.x - constDev.viewPort.x, 2 * constDev.mouse.y - constDev.viewPort.y)
                                / min(constDev.viewPort.x, constDev.viewPort.y) * constDev.zoom, 0) - eye;
    float3 viewDir = 0 - eye;

    float t = -dot(viewDir, eye) / dot(viewDir, rayDir);

    float3 ball = eye + t * rayDir;

    ball = mul(constDev.rotation, ball);
    float4 force = mul(constDev.rotation, constDev.dragDirection) * constDev.zoom * 4;

    // Texture coordinates of intersection
    ball.x = constDev.volumeSize.x * (ball.x + 1) / 2;
    ball.y = constDev.volumeSize.y * (ball.y + 1) / 2;
    ball.z = constDev.volumeSize.z * (ball.z + 1) / 2;

    // Volume coordinates relative to sphere center
    ball.x = x - ball.x;
    ball.y = y - ball.y;
    ball.z = z - ball.z;

    float r = dot(ball, ball); // Radius^2 square

    // Draw sphere
    float g = exp(-r / (constDev.mouse.w * constDev.mouse.w));
    float4 speed = tex3D<float4>(speedIn, x, y, z);
    surf3Dwrite(make_float4(speed.x + force.x * g,
                            speed.y + force.y * g,
                            speed.z + force.z * g,
                            speed.w + length(force) * g), speedOut, x, y, z);
}

__device__ uint rgbaFloatToInt(float4 rgba) {
    rgba.x = __saturatef(rgba.x);
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);
    return (uint(rgba.w * 255) << 24) | (uint(rgba.z * 255) << 16) | (uint(rgba.y * 255) << 8) | uint(rgba.x * 255);
}

__global__ void renderVolumeKernel(cudaTextureObject_t speedSizeIn,
                             uint* rgbaOut) {
    const uint3 tid = make_uint3(blockIdx.x * blockDim.x + threadIdx.x,
                                 blockIdx.y * blockDim.y + threadIdx.y,
                                 blockIdx.z * blockDim.z + threadIdx.z);

    float3 eye = make_float3(0, 0, 4);
    float testX = 2 * tid.x;
    float testY = 2 * tid.y;
    testX -= constDev.viewPort.x;
    testY -= constDev.viewPort.y;
    uint bla = min(constDev.viewPort.x, constDev.viewPort.y);
    testX /= bla;
    testY /= bla;
    testX *= constDev.zoom;
    testY *= constDev.zoom;
    float3 rayDir = make_float3(testX, testY, 0) - eye;

    eye = mul(constDev.rotation, eye);
    rayDir = mul(constDev.rotation, rayDir);

    float3 t1 = fmaxf(((-1 - eye) / rayDir), make_float3(0, 0, 0));
    float3 t2 = fmaxf((( 1 - eye) / rayDir), make_float3(0, 0, 0));

    float3 front = fminf(t1, t2);
    float3 back = fmaxf(t1, t2);

    float tfront = fmaxf(front.x, fmaxf(front.y, front.z));
    float tback = fminf(back.x, fminf(back.y, back.z));

    float3 texf = (eye + tfront * rayDir + 1) / 2;
    float3 texb = (eye + tback * rayDir + 1) / 2;

    float steps = floor(length(texf - texb) * constDev.volumeSize.x + 0.5f);

    float3 texDir = (texb - texf) / steps;

    steps = (tfront >= tback) ? 0 : steps;

    float m = 0;
    for (float i = 0.5f; i < steps; i++) {
        //float3 sam = make_float3(i, i, i);
        float s = tex3DLod<float>(speedSizeIn, 0.5f, 0.5f, 0.5f, 0.1f);
        m = fmaxf(m, s);
    }

    float4 color = lerp(make_float4(0, -1.41, -3, -0.4), make_float4(1.41, 1.41, 1, 1.41), m / 3);

    rgbaOut[tid.y * constDev.viewPort.x + tid.x] = rgbaFloatToInt(color);
}

void renderVolume(cudaTextureObject_t speedSizeIn,
                  uint* rgbaOut,
                  uint dx, uint dy, uint dz) {
    dim3 blockSize(16, 16, 1);
    dim3 gridSize(dx, dy, dz);

    renderVolumeKernel << <gridSize, blockSize >> >(speedSizeIn, rgbaOut);
    cudaDeviceSynchronize();
    getLastCudaError("volume kernel failed");
}

__global__ void project3DKernel(cudaTextureObject_t pressureIn,
                                cudaSurfaceObject_t speedSizeOut) {
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;
    uint z = blockIdx.z * blockDim.z + threadIdx.z;
    uint3 tid = make_uint3(x, y, z);

    if (tid.x >= constDev.volumeSize.x / 4) {
        return;
    }

    //if ((x >= constDev.volumeSize.x) || (y >= constDev.volumeSize.y) || (z >= constDev.volumeSize.z)) {
    //    return;
    //}

    //uint3 j;

    //j = make_uint3(max(x - 1, 0), y, z);
    //float4 pxm = make_float4(tex3D<float4>(pressureIn, j.x, j.y, j.z).w,
    //                         tex3D<float4>(pressureIn, x + 0, y + 0, z + 0).x,
    //                         tex3D<float4>(pressureIn, x + 0, y + 0, z + 0).y,
    //                         tex3D<float4>(pressureIn, x + 0, y + 0, z + 0).z);
    ////j = make_uint3(min(x + 1, volumeSize.x - 1), y, z);
    //float4 pxp = make_float4(tex3D<float4>(pressureIn, x + 0, y + 0, z + 0).y,
    //                         tex3D<float4>(pressureIn, x + 0, y + 0, z + 0).z,
    //                         tex3D<float4>(pressureIn, x + 0, y + 0, z + 0).w,
    //                         tex3D<float4>(pressureIn, j.x, j.y, j.z).x);

    //j = make_uint3(x, max(y - 1, 0), z);
    //float4 pym = tex3D<float4>(pressureIn, j.x, j.y, j.z);
    ////j = make_uint3(min(x + 1, volumeSize.x - 1), y, z);
    //float4 pyp = tex3D<float4>(pressureIn, j.x, j.y, j.z);

    //j = make_uint3(x, max(y - 1, 0), z);
    //float4 pzm = tex3D<float4>(pressureIn, j.x, j.y, j.z);
    ////j = make_uint3(min(x + 1, volumeSize.x - 1), y, z);
    //float4 pzp = tex3D<float4>(pressureIn, j.x, j.y, j.z);

    //pxp -= pxm;
    //pyp -= pym;
    //pzp -= pzm;

    //float4 s;
    //x *= 4;
    //bool borderYZ = any

    surf3Dwrite(1.0f, speedSizeOut, tid.x * sizeof(float), tid.y, tid.z);
}

__global__ void testKernel(cudaTextureObject_t speedSize) {
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;
    uint z = blockIdx.z * blockDim.z + threadIdx.z;
    uint3 tid = make_uint3(x, y, z);

    //if (x >= constDev.volumeSize.x / 4) {
    //    return;
    //}

    float s = tex3D<float>(speedSize, 0.5f, 0.5f, 0.5f);

    //if (x == 1 && y == 1 && z == 4) {
    //    printf("%f\n", s);
    //}

    if (s > 0) {
        printf("%f\n", s);
    }
}

void project3D(cudaTextureObject_t pressureIn,
               cudaSurfaceObject_t speedSizeOut,
               cudaTextureObject_t speedSize,
               uint dx, uint dy, uint dz) {
    dim3 blockSize(16, 4, 4);
    dim3 gridSize(dx, dy, dz);
    //std::cout << gridSize.x << " " << gridSize.y << " " << gridSize.z << std::endl;

    project3DKernel << <gridSize, blockSize >> >(pressureIn, speedSizeOut);
    cudaDeviceSynchronize();
    getLastCudaError("project kernel failed");

    //testKernel << <gridSize, blockSize >> >(speedSize);
    //getLastCudaError("test kernel failed");
    //cudaDeviceSynchronize();
}


void copyToConstant(const Constant& c) {
    CUDA::MemoryManagement::moveHostToSymbol(constDev, c);
}

} // namespace Kernel