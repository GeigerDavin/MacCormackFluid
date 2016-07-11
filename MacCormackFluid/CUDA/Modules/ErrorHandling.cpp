#include "../../StdAfx.hpp"
#include "ErrorHandling.hpp"

namespace CUDA {
namespace ErrorHandling {

void _getLastCudaError(const char* errorMessage, const char* file, int line) {
    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess) {
        fprintf(stderr, "%s(%i): getLastCudaError(): CUDA error: %s: (%d) %s\n",
                file, line, errorMessage, static_cast<unsigned int>(err),
                cudaGetErrorString(err));
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

void _cudaMemCheck(const void* ptr, const char* location, const char* file, int line) {
    if (!ptr) {
        fprintf(stderr, "%s malloc returned null @ %s (line %d)\n", location, file, line);
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

} // namespace ErrorHandling
} // namespace CUDA