#include "../../StdAfx.hpp"
#include "DeviceManagement.hpp"
#include "ErrorHandling.hpp"
#include "MemoryManagement.hpp"

namespace CUDA {

bool useCuda = false;

namespace DeviceManagement {

void initializeCuda(bool useGpu, int device) {
    useCuda = true;
    if (!useGpu) {
        std::cout << "Using CPU only" << std::endl;
        useCuda = false;
        return;
    }

    int count = 0;
    getDeviceCount(&count);
    if (count < 1) {
        std::cout << "No CUDA devices found; using CPU only" << std::endl;
        useCuda = false;
    }

    if (useCuda) {
        if (device >= 0 && device < count) {
            setDevice(device);
            cudaDeviceProp prop;
            getDeviceProperties(&prop, device);
            if (prop.major < 2) {
                device = -1;
            }
        } else {
            size_t freeMax = 0;
            for (int i = 0; i < count; i++) {
                setDevice(i);
                cudaDeviceProp prop;
                getDeviceProperties(&prop, i);
                if (prop.major < 2) {
                    continue;
                }
                size_t free = 0, total = 0;
                MemoryManagement::getMemInfo(&free, &total);
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
        setDevice(device);
        MemoryManagement::freeDevice(0);

        cudaDeviceProp prop;
        getDeviceProperties(&prop, device);
        size_t free = 0, total = 0;
        MemoryManagement::getMemInfo(&free, &total);

        printf("_________________________________________\n");
        printf("%s (%lluMB free)\n", prop.name, (unsigned long long) free / 1048576);
        printf("Using %d multiprocessors\n", prop.multiProcessorCount);
        printf("Max threads per processor: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("Max threads per dim: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("_________________________________________\n");
        printf("\n");
    }
}

void deviceReset() {
    if (useCuda) {
        checkCudaError(cudaDeviceReset());
    }
}

void deviceSync() {
    if (useCuda) {
        checkCudaError(cudaDeviceSynchronize());
    }
}

void setDevice(int device) {
    if (useCuda) {
        checkCudaError(cudaSetDevice(device));
    }
}

void getDevice(int* device) {
    if (useCuda) {
        checkCudaError(cudaGetDevice(device));
    }
}

void getDeviceCount(int* count) {
    if (useCuda) {
        checkCudaError(cudaGetDeviceCount(count));
    }
}

void getDeviceProperties(cudaDeviceProp* prop, int device) {
    if (useCuda) {
        checkCudaError(cudaGetDeviceProperties(prop, device));
    }
}

} // namespace DeviceManagement
} // namespace CUDA