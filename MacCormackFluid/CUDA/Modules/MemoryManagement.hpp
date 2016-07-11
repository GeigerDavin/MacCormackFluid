#ifndef ALLOCATOR_HPP
#define ALLOCATOR_HPP

namespace CUDA {
namespace MemoryManagement {

void getMemInfo(size_t* free, size_t* total);

void registerHost(void* ptr, size_t size, uint flags);
void unregisterHost(void* ptr);
void* getDevicePointerHost(void* hostPtr, uint flags);
void getHostFlags(uint* flags, void* hostPtr);

/**
 * Host (Pin-Locked) Memory
 */
void* mallocHost(size_t size);
void* mallocHost(size_t size, uint flags);
void* reallocHost(void* ptr, size_t size);
void* reallocHost(void* ptr, size_t oldSize, size_t newSize);
void freeHost(void* ptr);

/**
 * Managed (Unified) Memory
 */
void* mallocManaged(size_t size);
void* reallocManaged(void* ptr, size_t size);
void* reallocManaged(void* ptr, size_t oldSize, size_t newSize);
void freeManaged(void* ptr);

/**
 * Device Memory
 */
void* mallocDevice(size_t size);
void* reallocDevice(void* ptr, size_t size);
void* reallocDevice(void* ptr, size_t oldSize, size_t newSize);
void freeDevice(void* ptr);

/**
 * Device 2D (Pitched) Memory
 */
void* mallocPitch(size_t* pitch, size_t width, size_t height);
void* reallocPitch(void* ptr, size_t* pitch, size_t width, size_t height);
void* reallocPitch(void* ptr, size_t* pitch, size_t oldWidth, size_t oldHeight,
                   size_t newWidth, size_t newHeight);
void freePitch(void* ptr);

/**
 * Device 3D (Pitched) Memory
 */
cudaPitchedPtr malloc3D(cudaExtent extent);
cudaPitchedPtr realloc3D(cudaPitchedPtr ptr, cudaExtent extent);
cudaPitchedPtr realloc3D(cudaPitchedPtr ptr, cudaExtent oldExtent, cudaExtent newExtent);
void free3D(cudaPitchedPtr ptr);

/**
 * Device (Array) Memory
 */
cudaArray_t mallocArray(const cudaChannelFormatDesc* desc, size_t width,
                        size_t height = 0, uint flags = 0);
cudaArray_t reallocArray(cudaArray_t array, size_t width, size_t height = 0);
cudaArray_t reallocArray(cudaArray_t array, size_t oldWidth, size_t newWidth,
                         size_t oldHeight = 0, size_t newHeight = 0);
void freeArray(cudaArray_t array);

/**
 * Device 3D (Array) Memory
 */
cudaArray_t malloc3DArray(const cudaChannelFormatDesc* desc, cudaExtent extent, uint flags = 0);
cudaArray_t realloc3DArray(cudaArray_t array, cudaExtent extent);
cudaArray_t realloc3DArray(cudaArray_t array, cudaExtent oldExtent, cudaExtent newExtent);
void free3DArray(cudaArray_t array);

/**
 * Device (Mipmapped Array) Memory
 */
cudaMipmappedArray_t mallocMipmappedArray
    (const cudaChannelFormatDesc* desc, cudaExtent extent, uint numLevels, uint flags = 0);
cudaMipmappedArray_t reallocMipmappedArray
    (cudaMipmappedArray_t mipmappedArray, cudaExtent extent);
cudaMipmappedArray_t reallocMipmappedArray
    (cudaMipmappedArray_t mipmappedArray, cudaExtent oldExtent, cudaExtent newExtent);
void freeMipmappedArray(cudaMipmappedArray_t mipmappedArray);
cudaArray_t getMipmappedArrayLevel(cudaMipmappedArray_const_t mipmappedArray, uint level);

void getArrayInfo(cudaChannelFormatDesc* desc, cudaExtent* extent, uint* flags, cudaArray_t array);

/**
 * 1D Copy
 */
void moveHostToHost(void* dst, const void* src, size_t count);
void moveHostToDevice(void* dst, const void* src, size_t count);
void moveDeviceToHost(void* dst, const void* src, size_t count);
void moveDeviceToDevice(void* dst, const void* src, size_t count);

/**
 * 1D Async Copy
 */
void moveHostToHostAsync
    (void* dst, const void* src, size_t count, cudaStream_t stream = nullptr);
void moveHostToDeviceAsync
    (void* dst, const void* src, size_t count, cudaStream_t stream = nullptr);
void moveDeviceToHostAsync
    (void* dst, const void* src, size_t count, cudaStream_t stream = nullptr);
void moveDeviceToDeviceAsync
    (void* dst, const void* src, size_t count, cudaStream_t stream = nullptr);

/**
 * 2D Copy
 */
void moveHostToHost2D
    (void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height);
void moveHostToDevice2D
    (void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height);
void moveDeviceToHost2D
    (void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height);
void moveDeviceToDevice2D
    (void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height);

/**
 * 2D Async Copy
 */
void moveHostToHost2DAsync
    (void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height,
     cudaStream_t stream = nullptr);
void moveHostToDevice2DAsync
    (void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height,
     cudaStream_t stream = nullptr);
void moveDeviceToHost2DAsync
    (void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height,
     cudaStream_t stream = nullptr);
void moveDeviceToDevice2DAsync
    (void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height,
     cudaStream_t stream = nullptr);

/**
 * 2D Pointer to Array Copy
 */
void moveHost2DToArray
    (cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch,
     size_t width, size_t height);
void moveDevice2DToArray
    (cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch,
     size_t width, size_t height);

/**
 * 2D Pointer to Array Async Copy
 */
void moveHost2DToArrayAsync
    (cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch,
     size_t width, size_t height, cudaStream_t stream = nullptr);
void moveDevice2DToArrayAsync
    (cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch,
     size_t width, size_t height, cudaStream_t stream = nullptr);

/**
 * 2D Array to Pointer Copy
 */
void moveArrayToHost2D
    (void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset,
     size_t width, size_t height);
void moveArrayToDevice2D
    (void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset,
     size_t width, size_t height);

/**
* 2D Array to Pointer Async Copy
*/
void moveArrayToHost2D
    (void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset,
     size_t width, size_t height, cudaStream_t stream = nullptr);
void moveArrayToDevice2D
    (void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset,
     size_t width, size_t height, cudaStream_t stream = nullptr);

/**
 * 2D Array to Array Copy
 */
void moveArrayToArray2D
    (cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src,
     size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height);

//void moveHostToSymbol(const void* symbol, const void* src, size_t count, size_t offset = 0);

void deviceMemset(void* ptr, int value, size_t count);
void deviceMemset2D(void* ptr, size_t pitch, int value, size_t width, size_t height);
void deviceMemset3D(cudaPitchedPtr ptr, int value, cudaExtent extent);

void deviceMemsetAsync
    (void* ptr, int value, size_t count, cudaStream_t stream = nullptr);
void deviceMemset2DAsync
    (void* ptr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream = nullptr);
void deviceMemset3DAsync
    (cudaPitchedPtr ptr, int value, cudaExtent extent, cudaStream_t stream = nullptr);

cudaPos createCudaPos(size_t x, size_t y, size_t z);
cudaExtent createCudaExtent(size_t width, size_t height, size_t depth);
cudaPitchedPtr createCudaPitchedPtr(void* ptr, size_t pitch, size_t width, size_t height);

template <class Symbol>
void moveHostToSymbol(Symbol& symbol, const Symbol& source) {
    void* target = nullptr;
    cudaGetSymbolAddress(&target, symbol);
    moveHostToDevice(target, &source, sizeof(source));
}

size_t getSymbolSize(const void* symbol);
void* getSymbolAddress(const void* symbol);

} // namespace MemoryManagement
} // namespace CUDA

#endif