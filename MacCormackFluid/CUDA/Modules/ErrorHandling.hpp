#ifndef CUDA_ERROR_HANDLING_HPP
#define CUDA_ERROR_HANDLING_HPP

namespace CUDA {
namespace ErrorHandling {

static const char* _cudaGetErrorEnum(cudaError_t error) {
    switch (error) {
    case cudaSuccess:
        return "cudaSuccess";
    case cudaErrorMissingConfiguration:
        return "cudaErrorMissingConfiguration";
    case cudaErrorMemoryAllocation:
        return "cudaErrorMemoryAllocation";
    case cudaErrorInitializationError:
        return "cudaErrorInitializationError";
    case cudaErrorLaunchFailure:
        return "cudaErrorLaunchFailure";
    case cudaErrorPriorLaunchFailure:
        return "cudaErrorPriorLaunchFailure";
    case cudaErrorLaunchTimeout:
        return "cudaErrorLaunchTimeout";
    case cudaErrorLaunchOutOfResources:
        return "cudaErrorLaunchOutOfResources";
    case cudaErrorInvalidDeviceFunction:
        return "cudaErrorInvalidDeviceFunction";
    case cudaErrorInvalidConfiguration:
        return "cudaErrorInvalidConfiguration";
    case cudaErrorInvalidDevice:
        return "cudaErrorInvalidDevice";
    case cudaErrorInvalidValue:
        return "cudaErrorInvalidValue";
    case cudaErrorInvalidPitchValue:
        return "cudaErrorInvalidPitchValue";
    case cudaErrorInvalidSymbol:
        return "cudaErrorInvalidSymbol";
    case cudaErrorMapBufferObjectFailed:
        return "cudaErrorMapBufferObjectFailed";
    case cudaErrorUnmapBufferObjectFailed:
        return "cudaErrorUnmapBufferObjectFailed";
    case cudaErrorInvalidHostPointer:
        return "cudaErrorInvalidHostPointer";
    case cudaErrorInvalidDevicePointer:
        return "cudaErrorInvalidDevicePointer";
    case cudaErrorInvalidTexture:
        return "cudaErrorInvalidTexture";
    case cudaErrorInvalidTextureBinding:
        return "cudaErrorInvalidTextureBinding";
    case cudaErrorInvalidChannelDescriptor:
        return "cudaErrorInvalidChannelDescriptor";
    case cudaErrorInvalidMemcpyDirection:
        return "cudaErrorInvalidMemcpyDirection";
    case cudaErrorAddressOfConstant:
        return "cudaErrorAddressOfConstant";
    case cudaErrorTextureFetchFailed:
        return "cudaErrorTextureFetchFailed";
    case cudaErrorTextureNotBound:
        return "cudaErrorTextureNotBound";
    case cudaErrorSynchronizationError:
        return "cudaErrorSynchronizationError";
    case cudaErrorInvalidFilterSetting:
        return "cudaErrorInvalidFilterSetting";
    case cudaErrorInvalidNormSetting:
        return "cudaErrorInvalidNormSetting";
    case cudaErrorMixedDeviceExecution:
        return "cudaErrorMixedDeviceExecution";
    case cudaErrorCudartUnloading:
        return "cudaErrorCudartUnloading";
    case cudaErrorUnknown:
        return "cudaErrorUnknown";
    case cudaErrorNotYetImplemented:
        return "cudaErrorNotYetImplemented";
    case cudaErrorMemoryValueTooLarge:
        return "cudaErrorMemoryValueTooLarge";
    case cudaErrorInvalidResourceHandle:
        return "cudaErrorInvalidResourceHandle";
    case cudaErrorNotReady:
        return "cudaErrorNotReady";
    case cudaErrorInsufficientDriver:
        return "cudaErrorInsufficientDriver";
    case cudaErrorSetOnActiveProcess:
        return "cudaErrorSetOnActiveProcess";
    case cudaErrorInvalidSurface:
        return "cudaErrorInvalidSurface";
    case cudaErrorNoDevice:
        return "cudaErrorNoDevice";
    case cudaErrorECCUncorrectable:
        return "cudaErrorECCUncorrectable";
    case cudaErrorSharedObjectSymbolNotFound:
        return "cudaErrorSharedObjectSymbolNotFound";
    case cudaErrorSharedObjectInitFailed:
        return "cudaErrorSharedObjectInitFailed";
    case cudaErrorUnsupportedLimit:
        return "cudaErrorUnsupportedLimit";
    case cudaErrorDuplicateVariableName:
        return "cudaErrorDuplicateVariableName";
    case cudaErrorDuplicateTextureName:
        return "cudaErrorDuplicateTextureName";
    case cudaErrorDuplicateSurfaceName:
        return "cudaErrorDuplicateSurfaceName";
    case cudaErrorDevicesUnavailable:
        return "cudaErrorDevicesUnavailable";
    case cudaErrorInvalidKernelImage:
        return "cudaErrorInvalidKernelImage";
    case cudaErrorNoKernelImageForDevice:
        return "cudaErrorNoKernelImageForDevice";
    case cudaErrorIncompatibleDriverContext:
        return "cudaErrorIncompatibleDriverContext";
    case cudaErrorPeerAccessAlreadyEnabled:
        return "cudaErrorPeerAccessAlreadyEnabled";
    case cudaErrorPeerAccessNotEnabled:
        return "cudaErrorPeerAccessNotEnabled";
    case cudaErrorDeviceAlreadyInUse:
        return "cudaErrorDeviceAlreadyInUse";
    case cudaErrorProfilerDisabled:
        return "cudaErrorProfilerDisabled";
    case cudaErrorProfilerNotInitialized:
        return "cudaErrorProfilerNotInitialized";
    case cudaErrorProfilerAlreadyStarted:
        return "cudaErrorProfilerAlreadyStarted";
    case cudaErrorProfilerAlreadyStopped:
        return "cudaErrorProfilerAlreadyStopped";
    case cudaErrorAssert:
        return "cudaErrorAssert";
    case cudaErrorTooManyPeers:
        return "cudaErrorTooManyPeers";
    case cudaErrorHostMemoryAlreadyRegistered:
        return "cudaErrorHostMemoryAlreadyRegistered";
    case cudaErrorHostMemoryNotRegistered:
        return "cudaErrorHostMemoryNotRegistered";
    case cudaErrorOperatingSystem:
        return "cudaErrorOperatingSystem";
    case cudaErrorPeerAccessUnsupported:
        return "cudaErrorPeerAccessUnsupported";
    case cudaErrorLaunchMaxDepthExceeded:
        return "cudaErrorLaunchMaxDepthExceeded";
    case cudaErrorLaunchFileScopedTex:
        return "cudaErrorLaunchFileScopedTex";
    case cudaErrorLaunchFileScopedSurf:
        return "cudaErrorLaunchFileScopedSurf";
    case cudaErrorSyncDepthExceeded:
        return "cudaErrorSyncDepthExceeded";
    case cudaErrorLaunchPendingCountExceeded:
        return "cudaErrorLaunchPendingCountExceeded";
    case cudaErrorNotPermitted:
        return "cudaErrorNotPermitted";
    case cudaErrorNotSupported:
        return "cudaErrorNotSupported";
    case cudaErrorHardwareStackError:
        return "cudaErrorHardwareStackError";
    case cudaErrorIllegalInstruction:
        return "cudaErrorIllegalInstruction";
    case cudaErrorMisalignedAddress:
        return "cudaErrorMisalignedAddress";
    case cudaErrorInvalidAddressSpace:
        return "cudaErrorInvalidAddressSpace";
    case cudaErrorInvalidPc:
        return "cudaErrorInvalidPc";
    case cudaErrorIllegalAddress:
        return "cudaErrorIllegalAddress";
    case cudaErrorInvalidPtx:
        return "cudaErrorInvalidPtx";
    case cudaErrorInvalidGraphicsContext:
        return "cudaErrorInvalidGraphicsContext";
    case cudaErrorStartupFailure:
        return "cudaErrorStartupFailure";
    case cudaErrorApiFailureBase:
        return "cudaErrorApiFailureBase";
    }

    return "<unknown>";
}

template <typename T>
void _checkCudaError(const T& result, const char* func, const char* file, int line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
                file, line, static_cast<unsigned int>(result),
                _cudaGetErrorEnum(result), func);
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

void _getLastCudaError(const char* errorMessage, const char* file, int line);
void _cudaMemCheck(const void* ptr, const char* location, const char* file, int line);

} // namespace ErrorHandling
} // namespace CUDA

#define checkCudaError(val) CUDA::ErrorHandling::_checkCudaError((val), #val, __FILE__, __LINE__)
#define getLastCudaError(msg) CUDA::ErrorHandling::_getLastCudaError(msg, __FILE__, __LINE__)

#endif