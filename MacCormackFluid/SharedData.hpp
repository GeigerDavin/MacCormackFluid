#ifndef SHARED_DATA_HPP
#define SHARED_DATA_HPP

#include "CUDA/CudaArray3D.hpp"
#include "CUDA/CudaSurfaceObject.hpp"
#include "CUDA/CudaTextureObject.hpp"

typedef struct {
    float4 m[4];
} float4x4;

struct __align__(128) SharedDataGPU {
    uint2 viewPort;
    uint viewSlice;
    uint viewOrientation;

    float4 mouse;
    float4 dragDirection;

    float4x4 rotation;

    float zoom;
    int smoky;
};

__constant__ static int workerId;
__constant__ static float4 volumeSizeDev;

struct SharedDataCPU {
    bool running;

    float timeDiff;
    float totalTime;
    float timeAverage;
    float elapsedTimeSinceSecond;
};

__constant__ static SharedDataGPU g;

typedef struct MPI_BROADCASTDATA
{
	SharedDataGPU sharedDataGPUHost;
	SharedDataCPU sharedDataCPUHost;
} MPI_BROADCASTDATA;

extern MPI_BROADCASTDATA mpiBoardCastData;

#endif
