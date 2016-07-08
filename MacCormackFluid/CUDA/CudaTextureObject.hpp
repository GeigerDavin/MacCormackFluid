#ifndef CUDA_TEXTURE_OBECT_HPP
#define CUDA_TEXTURE_OBECT_HPP

namespace CUDA {

class CudaTextureObjectPrivate;

class CudaTextureObject {
    DISABLE_COPY(CudaTextureObject)

public:
    enum class ResourceType {
        Array           = cudaResourceTypeArray,
        Mipmap          = cudaResourceTypeMipmappedArray,
        Linear          = cudaResourceTypeLinear,
        Pitch2D         = cudaResourceTypePitch2D
    };

    enum class ReadMode {
        ElementType     = cudaReadModeElementType,
        NormalizedFloat = cudaReadModeNormalizedFloat
    };

    enum class AddressMode {
        Wrap            = cudaAddressModeWrap,
        Clamp           = cudaAddressModeClamp,
        Mirror          = cudaAddressModeMirror,
        Border          = cudaAddressModeBorder
    };

    enum class FilterMode {
        PointFilter     = cudaFilterModePoint,
        LinearFilter    = cudaFilterModeLinear
    };

public:
    CudaTextureObject();
    explicit CudaTextureObject(ResourceType type);
    CudaTextureObject(CudaTextureObject&& other);
    ~CudaTextureObject();

    CudaTextureObject& operator = (CudaTextureObject&& other);

public:
    void setResourceType(ResourceType type);
    ResourceType getResourceType() const;

    void setReadMode(ReadMode mode);
    ReadMode getReadMode() const;

    void setFilterMode(FilterMode mode);
    FilterMode getFilterMode() const;

    void setAddressMode0(AddressMode mode);
    AddressMode getAddressMode0() const;

    void setAddressMode1(AddressMode mode);
    AddressMode getAddressMode1() const;

    void setAddressMode2(AddressMode mode);
    AddressMode getAddressMode2() const;

    void setNormalized(bool normalized);
    bool isNormalized() const;

    bool create(void* data);
    bool isCreated() const;

    void destroy();

    /* Return opaque texture object to pass into the CUDA kernel */
    cudaTextureObject_t getId() const;

private:
    DECLARE_PRIVATE(CudaTextureObject)

    CudaTextureObjectPrivate* dPtr;
};

} // namespace CUDA

#endif