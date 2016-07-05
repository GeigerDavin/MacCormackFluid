#ifndef CUDA_SURFACE_OBJECT_HPP
#define CUDA_SURFACE_OBJECT_HPP

namespace CUDA {

class CudaSurfaceObjectPrivate;

class CudaSurfaceObject {
    DISABLE_COPY(CudaSurfaceObject)

public:
    enum class ResourceType {
        Array           = cudaResourceTypeArray,
        Mipmap          = cudaResourceTypeMipmappedArray,
        Linear          = cudaResourceTypeLinear,
        Pitch2D         = cudaResourceTypePitch2D
    };

public:
    CudaSurfaceObject();
    explicit CudaSurfaceObject(ResourceType type);
    ~CudaSurfaceObject();

public:
    void setResourceType(ResourceType type);
    ResourceType getResourceType() const;

    bool create(void* data);
    bool isCreated() const;

    void destroy();

    cudaSurfaceObject_t getSurf() const;

private:
    DECLARE_PRIVATE(CudaSurfaceObject)

    CudaSurfaceObjectPrivate* dPtr;
};

} // namespace CUDA

#endif