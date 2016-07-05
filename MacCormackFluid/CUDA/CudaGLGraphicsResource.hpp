#ifndef CUDA_GL_GRAPHICS_RESOURCE_HPP
#define CUDA_GL_GRAPHICS_RESOURCE_HPP

namespace CUDA {

class CudaGLGraphicsResourcePrivate;

class CudaGLGraphicsResource {
    DISABLE_COPY(CudaGLGraphicsResource)

public:
    enum class Type {
        Image               = 1,
        Buffer              = 2
    };

    enum class RegisterFlag {
        None                = cudaGraphicsRegisterFlagsNone,
        ReadOnly            = cudaGraphicsRegisterFlagsReadOnly,
        WriteDiscard        = cudaGraphicsRegisterFlagsWriteDiscard,
        SurfaceLoadStore    = cudaGraphicsRegisterFlagsSurfaceLoadStore,
        TextureGather       = cudaGraphicsRegisterFlagsTextureGather
    };

    enum class MapFlag {
        None                = cudaGraphicsMapFlagsNone,
        ReadOnly            = cudaGraphicsMapFlagsReadOnly,
        Discard             = cudaGraphicsMapFlagsWriteDiscard
    };

    enum class ImageTarget {
        Texture2D           = GL_TEXTURE_2D,
        TextureRectangle    = GL_TEXTURE_RECTANGLE,
        TextureCubeMap      = GL_TEXTURE_CUBE_MAP,
        Texture3D           = GL_TEXTURE_3D,
        Texture2DArray      = GL_TEXTURE_2D_ARRAY,
        Renderbuffer        = GL_RENDERBUFFER
    };

public:
    CudaGLGraphicsResource();
    explicit CudaGLGraphicsResource(Type type);
    ~CudaGLGraphicsResource();

public:
    void setType(Type type);
    Type getType() const;

    void setRegisterFlag(RegisterFlag flag);
    RegisterFlag getRegisterFlag() const;

    void setImageTarget(ImageTarget target);
    ImageTarget getImageTarget() const;

    void setStream(cudaStream_t stream);
    cudaStream_t getStream() const;

    bool create(GLuint id);
    bool isCreated() const;

    void destroy();

    int getSize() const;

    void* map(MapFlag flag = MapFlag::None);
    cudaArray_t mapArray(MapFlag flag = MapFlag::None) const;
    cudaMipmappedArray_t mapMipmappedArray(MapFlag flag = MapFlag::None) const;
    void unmap() const;

private:
    DECLARE_PRIVATE(CudaGLGraphicsResource)

    CudaGLGraphicsResourcePrivate* dPtr;
};

} // namespace CUDA

#endif