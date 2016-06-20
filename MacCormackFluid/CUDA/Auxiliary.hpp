#ifndef AUXILIARY_HPP
#define AUXILIARY_HPP

namespace CUDA {

extern bool useCuda;
void initializeCuda(bool useGpu, int device = 0);

void errorCheckCuda(const char* file, int line);
void memCheck(const void* ptr, const char* location, const char* file, int line);

dim3 getGridDim1D(uint count, uint threads = THREADS);

void deviceMemset(void* ptr, int value, size_t count);
void deviceSync();
void deviceReset();

} // namespace CUDA

#define ERRORCHECK_CUDA() CUDA::errorCheckCuda(__FILE__, __LINE__)

#endif
