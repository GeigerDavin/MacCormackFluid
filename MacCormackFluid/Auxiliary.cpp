#include "Auxiliary.hpp"
#include <iostream>
#include <math.h>
#include <curand_kernel.h>

bool useCuda = false;

void initializeCuda(bool useGpu, int device) {
    useCuda = true;
    if (!useGpu) {
        std::cout << "Using CPU only" << std::endl;
        useCuda = false;
        return;
    }

    int count = 0;
    cudaGetDeviceCount(&count);
    if (count < 1) {
        std::cout << "No CUDA devices found; using CPU only" << std::endl;
        useCuda = false;
    }

    if (useCuda) {
        if (device >= 0 && device < count) {
            cudaSetDevice(device);
            ERRORCHECK_CUDA();
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, device);
            if (prop.major < 2) {
                device = -1;
            }
        } else {
            size_t freeMax = 0;
            for (int i = 0; i < count; i++) {
                cudaSetDevice(i);
                ERRORCHECK_CUDA();
                cudaDeviceProp prop;
                cudaGetDeviceProperties(&prop, i);
                if (prop.major < 2) {
                    continue;
                }
                size_t free, total;
                cudaMemGetInfo(&free, &total);
                if (free > freeMax) {
                    freeMax = free;
                    device = i;
                }
            }
        }
        if (device < 0) {
            std::cout << "Compute capability 2.0 support required; using CPU only" << std::endl;
            useCuda = false;
        }
    }

    if (useCuda) {
        cudaSetDevice(device);
        cudaFree(0);

        ERRORCHECK_CUDA();

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        size_t free, total;
        cudaMemGetInfo(&free, &total);

        printf("_________________________________________\n");
        printf("%s (%lluMB free)\n", prop.name, (unsigned long long) free / 1048576);
        printf("Using %d multiprocessors\n", prop.multiProcessorCount);
        printf("Max threads per processor: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("Max threads per dim: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("_________________________________________\n");
        printf("\n");
        ERRORCHECK_CUDA();
    }
}

void errorCheckCuda(const char* file, int line) {
    if (!useCuda) {
        return;
    }
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error @ %s (line %d): %s\n", file, line, cudaGetErrorString(err));
        exit(-1);
    }
}

void memCheck(const void* ptr, const char* location, const char* file, int line) {
    if (!ptr) {
        printf("%s malloc returned null @ %s (line %d)\n", location, file, line);
        exit(-1);
    }
}

void deviceReset() {
    if (useCuda) {
        cudaDeviceReset();
        ERRORCHECK_CUDA();
    }
}

void deviceSync() {
    if (useCuda) {
        cudaDeviceSynchronize();
        ERRORCHECK_CUDA();
    }
}
