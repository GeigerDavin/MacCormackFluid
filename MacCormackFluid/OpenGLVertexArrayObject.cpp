#include "StdAfx.hpp"
#include "OpenGLVertexArrayObject.hpp"

#include "ResourceGuard.hpp"

class OpenGLVertexArrayObjectPrivate {
public:
    OpenGLVertexArrayObjectPrivate()
        : vaoGuard(nullptr) {}

    ~OpenGLVertexArrayObjectPrivate() {
        destroy();
    }

    bool create();
    void destroy();
    bool bind();
    void release();
    bool isValid() const;

    ResourceGuard<GLuint>* vaoGuard;
};

namespace {
    void freeVaoFunc(GLuint id) {
        glDeleteVertexArrays(1, &id);
    }
}

bool OpenGLVertexArrayObjectPrivate::create() {
    if (isValid()) {
        return true;
    }

    GLuint vao = 0;
    glGenVertexArrays(1, &vao);
    if (vao) {
        destroy();
        vaoGuard = new ResourceGuard<GLuint>(vao, freeVaoFunc);
        return isValid();
    } else {
        std::cout << "Could not create VAO" << std::endl;
        return false;
    }
}

void OpenGLVertexArrayObjectPrivate::destroy() {
    if (vaoGuard) {
        delete vaoGuard;
        vaoGuard = nullptr;
    }
}

bool OpenGLVertexArrayObjectPrivate::bind() {
    if (isValid()) {
        glBindVertexArray(vaoGuard->get());
        return true;
    } else {
        std::cout << "VAO not created" << std::endl;
        return false;
    }
}

void OpenGLVertexArrayObjectPrivate::release() {
    if (!isValid()) {
        std::cout << "VAO not created" << std::endl;
    }
    glBindVertexArray(0);
}

bool OpenGLVertexArrayObjectPrivate::isValid() const {
    return (vaoGuard) && (vaoGuard->get() != 0);
}

OpenGLVertexArrayObject::OpenGLVertexArrayObject()
    : dPtr(new OpenGLVertexArrayObjectPrivate) {}

OpenGLVertexArrayObject::~OpenGLVertexArrayObject() {
    destroy();
}

bool OpenGLVertexArrayObject::create() {
    D(OpenGLVertexArrayObject);
    return d->create();
}

bool OpenGLVertexArrayObject::isCreated() const {
    D(const OpenGLVertexArrayObject);
    return d->isValid();
}

void OpenGLVertexArrayObject::destroy() {
    if (dPtr) {
        delete dPtr;
        dPtr = nullptr;
    }
}

bool OpenGLVertexArrayObject::bind() {
    D(OpenGLVertexArrayObject);
    return d->bind();
}

void OpenGLVertexArrayObject::release() {
    D(OpenGLVertexArrayObject);
    d->release();
}

GLuint OpenGLVertexArrayObject::getId() const {
    D(const OpenGLVertexArrayObject);
    return (d->vaoGuard) ? (d->vaoGuard->get()) : (0);
}