#ifndef ALLOCATOR_HPP
#define ALLOCATOR_HPP

#include <cstdio>
#include <cuda_runtime.h>

void* mallocHost(size_t size);
void* reallocHost(void* ptr, size_t size);
void* reallocHost(void* ptr, size_t oldSize, size_t newSize);
void freeHost(void* ptr);

void* mallocDevice(size_t size);
void* reallocDevice(void* ptr, size_t size);
void* reallocDevice(void* ptr, size_t oldSize, size_t newSize);
void freeDevice(void* ptr);

void moveHostToHost(void* dst, const void* src, size_t count);
void moveHostToDevice(void* dst, const void* src, size_t count);
void moveDeviceToHost(void* dst, const void* src, size_t count);
void moveDeviceToDevice(void* dst, const void* src, size_t count);

void moveHostToDeviceAsync(void* dst, const void* src, size_t count, cudaStream_t stream = nullptr);
void moveDeviceToHostAsync(void* dst, const void* src, size_t count, cudaStream_t stream = nullptr);
void moveDeviceToDeviceAsync(void* dst, const void* src, size_t count, cudaStream_t stream = nullptr);

#endif

