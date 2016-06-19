#ifndef AUXILIARY_HPP
#define AUXILIARY_HPP

extern bool useCuda;
void initializeCuda(bool useGpu, int device = 0);

void errorCheckCuda(const char* file, int line);
void memCheck(const void* ptr, const char* location, const char* file, int line);

void deviceMemset(void* ptr, int value, size_t count);
void deviceSync();
void deviceReset();

#define ERRORCHECK_CUDA() errorCheckCuda(__FILE__, __LINE__)

#endif
