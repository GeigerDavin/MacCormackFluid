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
    CudaSurfaceObject(CudaSurfaceObject&& other);
    ~CudaSurfaceObject();

    CudaSurfaceObject& operator = (CudaSurfaceObject&& other);

public:
    void setResourceType(ResourceType type);
    ResourceType getResourceType() const;

    bool create(void* data);
    bool isCreated() const;

    void destroy();

    /* Return opaque surface object to pass into the CUDA kernel */
    cudaSurfaceObject_t getId() const;

private:
    DECLARE_PRIVATE(CudaSurfaceObject)

    CudaSurfaceObjectPrivate* dPtr;
};

} // namespace CUDA

#endif