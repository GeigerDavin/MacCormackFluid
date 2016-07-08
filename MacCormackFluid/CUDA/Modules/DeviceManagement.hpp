#ifndef CUDA_DEVICE_MANAGEMENT_HPP
#define CUDA_DEVICE_MANAGEMENT_HPP

namespace CUDA {

extern bool useCuda;

namespace DeviceManagement {

void initializeCuda(bool useGpu, int device = 0);

void deviceSync();
void deviceReset();
void setDevice(int device);
void getDevice(int* device);
void getDeviceCount(int* count);
void getDeviceProperties(cudaDeviceProp* prop, int device);

int gpuDeviceInit(int device);
int gpuGetMaxGflopsDeviceId();
int findCudaDevice(int argc, const char** argv);
bool checkCudaCapabilities(int major, int minor);

} // namespace DeviceManagement
} // namespace CUDA

#endif
