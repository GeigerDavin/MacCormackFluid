#ifndef OPENGL_BUFFER_HPP
#define OPENGL_BUFFER_HPP

namespace OpenGL {

class OpenGLBufferPrivate;

class OpenGLBuffer {
    DISABLE_COPY(OpenGLBuffer)

public:
    enum Type {
        VertexBuffer          = GL_ARRAY_BUFFER,
        IndexBuffer           = GL_ELEMENT_ARRAY_BUFFER,
        StorageBuffer         = GL_SHADER_STORAGE_BUFFER,
        UniformBuffer         = GL_UNIFORM_BUFFER,
        PixelPackBuffer       = GL_PIXEL_PACK_BUFFER,
        PixelUnpackBuffer     = GL_PIXEL_UNPACK_BUFFER
    };

    enum UsagePattern {
        StreamDraw            = GL_STREAM_DRAW,
        StreamRead            = GL_STREAM_READ,
        StreamCopy            = GL_STREAM_COPY,
        StaticDraw            = GL_STATIC_DRAW,
        StaticRead            = GL_STATIC_READ,
        StaticCopy            = GL_STATIC_COPY,
        DynamicDraw           = GL_DYNAMIC_DRAW,
        DynamicRead           = GL_DYNAMIC_READ,
        DynamicCopy           = GL_DYNAMIC_COPY
    };

    enum Access {
        ReadOnly              = GL_READ_ONLY,
        WriteOnly             = GL_WRITE_ONLY,
        ReadWrite             = GL_READ_WRITE
    };

    enum RangeAccessFlag {
        RangeRead             = GL_MAP_READ_BIT,
        RangeWrite            = GL_MAP_WRITE_BIT,
        RangeInvalidate       = GL_MAP_INVALIDATE_RANGE_BIT,
        RangeInvalidateBuffer = GL_MAP_INVALIDATE_BUFFER_BIT,
        RangeFlushExplicit    = GL_MAP_FLUSH_EXPLICIT_BIT,
        RangeUnsynchronized   = GL_MAP_UNSYNCHRONIZED_BIT
    };

public:
    OpenGLBuffer();
    explicit OpenGLBuffer(OpenGLBuffer::Type type);
    ~OpenGLBuffer();

public:
    void setType(OpenGLBuffer::Type value);
    OpenGLBuffer::Type getType() const;

    void setUsagePattern(OpenGLBuffer::UsagePattern value);
    OpenGLBuffer::UsagePattern getUsagePattern() const;

    bool create();
    bool isCreated() const;

    void destroy();

    bool bind();
    bool bindUniform(GLuint program, const char* uniformName);
    inline bool bindUniform(GLuint program, const std::string& uniformName) {
        return bindUniform(program, uniformName.c_str());
    }
    void release();

    static void release(OpenGLBuffer::Type type);

    GLuint getId() const;

    int getSize() const;

    bool read(int offset, void* data, int count) const;
    void write(int offset, const void* data, int count);

    void allocate(const void* data, int count);
    inline void allocate(int count) {
        allocate(nullptr, count);
    }

    void* map(OpenGLBuffer::Access access) const;
    void* mapRange(int offset, int count, OpenGLBuffer::RangeAccessFlag access) const;
    bool unmap();

private:
    DECLARE_PRIVATE(OpenGLBuffer)

    OpenGLBufferPrivate* dPtr;
};

} // namespace OpenGL

#endif