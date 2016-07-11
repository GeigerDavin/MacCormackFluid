#ifndef ADVECT_HPP
#define ADVECT_HPP

namespace Kernel {

// Advect speed, by sampling speed at 'pos - dt * speed' position
__global__ static void advect3D(cudaTextureObject_t speedInTex,
                                cudaSurfaceObject_t speedInSurf,
                                cudaSurfaceObject_t speedOutSurf) {
    TID_CONST;
    float4 speed = surf3Dread<float4>(speedInSurf, tid.x * sizeof(float4), tid.y, tid.z, CUDA_BOUNDARY_MODE);
    float samX = tid.x - speed.x;
    float samY = tid.y - speed.y;
    float samZ = tid.z - speed.z;
    float4 sam = make_float4(samX, samY, samZ, 0.0f);
    sam = (sam + 0.5f) / volumeSizeDev;

    float4 sampledSpeed = tex3D<float4>(speedInTex, sam.x, sam.y, sam.z);

    surf3Dwrite(sampledSpeed, speedOutSurf, tid.x * sizeof(float4), tid.y, tid.z, CUDA_BOUNDARY_MODE);
}

// Advect speed, by sampling speed at 'pos - dt * speed' position
__global__ static void advectBackward3D(cudaTextureObject_t speedInTex,
                                        cudaSurfaceObject_t speedInSurf,
                                        cudaSurfaceObject_t speedOutSurf) {
    TID_CONST;
    float4 speed = surf3Dread<float4>(speedInSurf, tid.x * sizeof(float4), tid.y, tid.z, CUDA_BOUNDARY_MODE);
    float samX = tid.x + speed.x;
    float samY = tid.y + speed.y;
    float samZ = tid.z + speed.z;
    float4 sam = make_float4(samX, samY, samZ, 0.0f);
    sam = (sam + 0.5f) / volumeSizeDev;

    float4 sampledSpeed = tex3D<float4>(speedInTex, sam.x, sam.y, sam.z);

    surf3Dwrite(sampledSpeed, speedOutSurf, tid.x * sizeof(float4), tid.y, tid.z, CUDA_BOUNDARY_MODE);
}

// Advect speed, by sampling speed at 'pos - dt * speed' position
__global__ static void advectMacCormack3D(cudaTextureObject_t speedInTex,
                                          cudaSurfaceObject_t speedInSurf,
                                          cudaTextureObject_t speedInTexA,
                                          cudaSurfaceObject_t speedOutSurf) {
    TID_CONST;
    float4 speed = surf3Dread<float4>(speedInSurf, tid.x * sizeof(float4), tid.y, tid.z, CUDA_BOUNDARY_MODE);
    float samX = tid.x - speed.x;
    float samY = tid.y - speed.y;
    float samZ = tid.z - speed.z;
    float4 sam = make_float4(samX, samY, samZ, 0.0f);

    uint3 j = make_uint3(sam.x, sam.y, sam.z);

    sam = (sam + 0.5f) / volumeSizeDev;

    float4 r0 = surf3Dread<float4>(speedInSurf, (j.x + 0) * sizeof(float4), (j.y + 0), (j.z + 0), CUDA_BOUNDARY_MODE);
    float4 r1 = surf3Dread<float4>(speedInSurf, (j.x + 1) * sizeof(float4), (j.y + 0), (j.z + 0), CUDA_BOUNDARY_MODE);
    float4 r2 = surf3Dread<float4>(speedInSurf, (j.x + 0) * sizeof(float4), (j.y + 1), (j.z + 0), CUDA_BOUNDARY_MODE);
    float4 r3 = surf3Dread<float4>(speedInSurf, (j.x + 1) * sizeof(float4), (j.y + 1), (j.z + 0), CUDA_BOUNDARY_MODE);
    float4 r4 = surf3Dread<float4>(speedInSurf, (j.x + 0) * sizeof(float4), (j.y + 0), (j.z + 1), CUDA_BOUNDARY_MODE);
    float4 r5 = surf3Dread<float4>(speedInSurf, (j.x + 1) * sizeof(float4), (j.y + 0), (j.z + 1), CUDA_BOUNDARY_MODE);
    float4 r6 = surf3Dread<float4>(speedInSurf, (j.x + 0) * sizeof(float4), (j.y + 1), (j.z + 1), CUDA_BOUNDARY_MODE);
    float4 r7 = surf3Dread<float4>(speedInSurf, (j.x + 1) * sizeof(float4), (j.y + 1), (j.z + 1), CUDA_BOUNDARY_MODE);

    float4 lmin = fminf(r0, fminf(r1, fminf(r2, fminf(r3, fminf(r4, fminf(r5, fminf(r6, r7)))))));
    float4 lmax = fmaxf(r0, fmaxf(r1, fmaxf(r2, fmaxf(r3, fmaxf(r4, fmaxf(r5, fmaxf(r6, r7)))))));

    float4 sampledSpeed0 = tex3D<float4>(speedInTex, sam.x, sam.y, sam.z);

    float4 sampledSpeed = sampledSpeed0 + 0.5f * (sampledSpeed0 - tex3D<float4>(speedInTexA, sam.x, sam.y, sam.z));

    sampledSpeed = clamp(sampledSpeed, lmin, lmax);
    sampledSpeed.w = max(sampledSpeed0.w - 0.001f, 0.0f);

    surf3Dwrite(sampledSpeed, speedOutSurf, tid.x * sizeof(float4), tid.y, tid.z, CUDA_BOUNDARY_MODE);
}

} // namespace Kernel

#endif
