#ifndef OPENGL_VERTEX_ARRAY_OBJECT_HPP
#define OPENGL_VERTEX_ARRAY_OBJECT_HPP

namespace OpenGL {

class OpenGLVertexArrayObjectPrivate;

class OpenGLVertexArrayObject {
public:
    explicit OpenGLVertexArrayObject();
    ~OpenGLVertexArrayObject();

public:
    bool create();
    bool isCreated() const;

    void destroy();

    bool bind();
    void release();

    GLuint getId() const;

public:
    class Binder {
    public:
        inline Binder(OpenGLVertexArrayObject* v)
            : vao(v) {

            _assert(vao);
            if (vao->isCreated() || vao->create()) {
                vao->bind();
            }
        }

        inline ~Binder() {
            release();
        }

        inline void release() {
            vao->release();
        }

        inline void rebind() {
            vao->bind();
        }

    private:
        DISABLE_COPY(Binder)
        OpenGLVertexArrayObject* vao;
    };

private:
    DISABLE_COPY(OpenGLVertexArrayObject)
    DECLARE_PRIVATE(OpenGLVertexArrayObject)

    OpenGLVertexArrayObjectPrivate* dPtr;
};

} // namespace OpenGL

#endif