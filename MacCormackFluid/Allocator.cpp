#include "Allocator.hpp"
#include "Auxiliary.hpp"

#include <stdlib.h>
#include <cstring>
#include <iostream>

#define MEMCHECK_HOST(ptr) memCheck(ptr, "Host", __FILE__, __LINE__)
#define MEMCHECK_DEVICE(ptr) memCheck(ptr, "Device", __FILE__, __LINE__)

void* mallocHost(size_t size) {
    if (size == 0) {
        return nullptr;
    }
    void* ptr = nullptr;
    if (useCuda) {
        cudaMallocHost(&ptr, size);
        ERRORCHECK_CUDA();
    } else {
        ptr = malloc(size);
    }
    MEMCHECK_HOST(ptr);
    memset(ptr, 0, size);
    return ptr;
}

void* reallocHost(void* ptr, size_t size) {
    freeHost(ptr);
    return mallocHost(size);
}

void* reallocHost(void* ptr, size_t oldSize, size_t newSize) {
    void* buffer = mallocHost(newSize);
    if (ptr) {
        moveHostToHost(buffer, ptr, oldSize);
        freeHost(ptr);
    }
    return buffer;
}

void freeHost(void* ptr) {
    if (!ptr) {
        return;
    }
    if (useCuda) {
        cudaFreeHost(ptr);
        ERRORCHECK_CUDA();
    } else {
        free(ptr);
    }
}

void* mallocDevice(size_t size) {
    if (size == 0) {
        return nullptr;
    }
    void* ptr = nullptr;
    if (useCuda) {
        cudaMalloc(&ptr, size);
        ERRORCHECK_CUDA();
        MEMCHECK_DEVICE(ptr);
        deviceMemset(ptr, 0, size);
    }
    return ptr;
}

void* reallocDevice(void* ptr, size_t size) {
    freeDevice(ptr);
    return mallocDevice(size);
}

void* reallocDevice(void* ptr, size_t oldSize, size_t newSize) {
    void* buffer = mallocDevice(newSize);
    if (ptr) {
        moveDeviceToDevice(buffer, ptr, oldSize);
        freeDevice(ptr);
    }
    return buffer;
}

void freeDevice(void* ptr) {
    if (!ptr) {
        return;
    }
    if (useCuda) {
        cudaFree(ptr);
        ERRORCHECK_CUDA();
    }
}

void moveHostToHost(void* dst, const void* src, size_t count) {
    if (count == 0) {
        return;
    }
    if (useCuda) {
        cudaMemcpy(dst, src, count, cudaMemcpyHostToHost);
        ERRORCHECK_CUDA();
    } else {
        memcpy(dst, src, count);
    }
}

void moveHostToDevice(void* dst, const void* src, size_t count) {
    if (!useCuda || count == 0) {
        return;
    }
    cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice);
    ERRORCHECK_CUDA();
}

void moveDeviceToHost(void* dst, const void* src, size_t count) {
    if (!useCuda || count == 0) {
        return;
    }
    cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost);
    ERRORCHECK_CUDA();
}

void moveDeviceToDevice(void* dst, const void* src, size_t count) {
    if (!useCuda || count == 0) {
        return;
    }
    cudaMemcpy(dst, src, count, cudaMemcpyDeviceToDevice);
    ERRORCHECK_CUDA();
}

void moveHostToDeviceAsync(void* dst, const void* src, size_t count, cudaStream_t stream) {
    if (!useCuda || count == 0) {
        return;
    }
    cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToDevice, stream);
    ERRORCHECK_CUDA();
}

void moveDeviceToHostAsync(void* dst, const void* src, size_t count, cudaStream_t stream) {
    if (!useCuda || count == 0) {
        return;
    }
    cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToHost, stream);
    ERRORCHECK_CUDA();
}

void moveDeviceToDeviceAsync(void* dst, const void* src, size_t count, cudaStream_t stream) {
    if (!useCuda || count == 0) {
        return;
    }
    cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice, stream);
    ERRORCHECK_CUDA();
}

void deviceMemset(void* ptr, int value, size_t count) {
    if (!useCuda || count == 0) {
        return;
    }
    cudaMemset(ptr, value, count);
    ERRORCHECK_CUDA();
}
