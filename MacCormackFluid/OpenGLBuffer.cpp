#include "StdAfx.hpp"
#include "OpenGLBuffer.hpp"

#include "ResourceGuard.hpp"

class OpenGLBufferPrivate {
public:
    OpenGLBufferPrivate(OpenGLBuffer::Type t)
        : type(t)
        , usagePattern(OpenGLBuffer::StaticDraw)
        , bufferGuard(nullptr) {}

    ~OpenGLBufferPrivate() {
        destroy();
    }

    bool create();
    void destroy();
    bool bind();
    void release();
    bool isValid() const;
    size_t getBindingPoint() const;

    OpenGLBuffer::Type type;
    OpenGLBuffer::UsagePattern usagePattern;
    ResourceGuard<GLuint>* bufferGuard;
    static std::vector<size_t> bindingPoints;
};

namespace {
    void freeBufferFunc(GLuint id) {
        glDeleteBuffers(1, &id);
    }
}

bool OpenGLBufferPrivate::create() {
    if (isValid()) {
        return true;
    }

    GLuint bufferId = 0;
    glGenBuffers(1, &bufferId);
    if (bufferId) {
        destroy();
        bufferGuard = new ResourceGuard<GLuint>(bufferId, freeBufferFunc);
        return isValid();
    } else {
        std::cout << "Could not create buffer";
        return false;
    }
}

void OpenGLBufferPrivate::destroy() {
    if (bufferGuard) {
        delete bufferGuard;
        bufferGuard = nullptr;
    }
}

bool OpenGLBufferPrivate::bind() {
    if (isValid()) {
        glBindBuffer(type, bufferGuard->get());
        return true;
    } else {
        std::cout << "Buffer not created";
        return false;
    }
}

void OpenGLBufferPrivate::release() {
    if (!isValid()) {
        std::cout << "Buffer not created";
    }
    glBindBuffer(type, 0);
}

bool OpenGLBufferPrivate::isValid() const {
    return (bufferGuard) && (bufferGuard->get() != 0);
}

std::vector<size_t> OpenGLBufferPrivate::bindingPoints;
size_t OpenGLBufferPrivate::getBindingPoint() const {
    for (size_t i = 0, size = bindingPoints.size(); i < size; i++) {
        if (bindingPoints.at(i) != i) {
            bindingPoints.insert(bindingPoints.begin() + i, i);
            return i;
        }
    }

    bindingPoints.push_back(bindingPoints.size());
    return bindingPoints.size();
}

OpenGLBuffer::OpenGLBuffer()
    : dPtr(new OpenGLBufferPrivate(OpenGLBuffer::VertexBuffer)) {}

OpenGLBuffer::OpenGLBuffer(OpenGLBuffer::Type type)
    : dPtr(new OpenGLBufferPrivate(type)) {}

OpenGLBuffer::~OpenGLBuffer() {
    destroy();
}

void OpenGLBuffer::setType(OpenGLBuffer::Type value) {
    D(OpenGLBuffer);
    d->type = value;
}

OpenGLBuffer::Type OpenGLBuffer::getType() const {
    D(const OpenGLBuffer);
    return d->type;
}

void OpenGLBuffer::setUsagePattern(OpenGLBuffer::UsagePattern value) {
    D(OpenGLBuffer);
    d->usagePattern = value;
}

OpenGLBuffer::UsagePattern OpenGLBuffer::getUsagePattern() const {
    D(const OpenGLBuffer);
    return d->usagePattern;
}

bool OpenGLBuffer::create() {
    D(OpenGLBuffer);
    return d->create();
}

bool OpenGLBuffer::isCreated() const {
    D(const OpenGLBuffer);
    return d->isValid();
}

void OpenGLBuffer::destroy() {
    if (dPtr) {
        delete dPtr;
        dPtr = nullptr;
    }
}

bool OpenGLBuffer::bind() {
    D(OpenGLBuffer);
    return d->bind();
}

bool OpenGLBuffer::bindUniform(GLuint program, const char* uniformName) {
    if (isCreated() && uniformName) {
        GLuint uniformBlockIndex = glGetUniformBlockIndex(program, uniformName);
        if (uniformBlockIndex == GL_INVALID_INDEX) {
            std::cout << "Could not find block index" << "'" << uniformName << "'";
            return false;
        }
        D(const OpenGLBuffer);
        GLuint bindingPoint = d->getBindingPoint();
        glUniformBlockBinding(program, uniformBlockIndex, bindingPoint);
        glBindBufferBase(d->type, bindingPoint, d->bufferGuard->get());
        return true;
    } else {
        std::cout << "Buffer not created";
        return false;
    }
}

void OpenGLBuffer::release() {
    D(OpenGLBuffer);
    d->release();
}

void OpenGLBuffer::release(OpenGLBuffer::Type type) {
    glBindBuffer(type, 0);
}

GLuint OpenGLBuffer::getId() const {
    D(const OpenGLBuffer);
    return (d->bufferGuard) ? (d->bufferGuard->get()) : (0);
}

int OpenGLBuffer::getSize() const {
    if (isCreated()) {
        D(const OpenGLBuffer);
        GLint value = -1;
        glGetBufferParameteriv(d->type, GL_BUFFER_SIZE, &value);
        return value;
    } else {
        std::cout << "Buffer not created";
        return -1;
    }
}

bool OpenGLBuffer::read(int offset, void* data, int count) const {
    if (isCreated()) {
        D(const OpenGLBuffer);
        while (glGetError() != GL_NO_ERROR);
        glGetBufferSubData(d->type, offset, count, data);
        return glGetError() == GL_NO_ERROR;
    } else {
        std::cout << "Buffer not created";
        return false;
    }
}

void OpenGLBuffer::write(int offset, const void* data, int count) {
    if (isCreated()) {
        D(const OpenGLBuffer);
        glBufferSubData(d->type, offset, count, data);
    } else {
        std::cout << "Buffer not created";
        return;
    }
}

void OpenGLBuffer::allocate(const void* data, int count) {
    if (isCreated()) {
        D(const OpenGLBuffer);
        glBufferData(d->type, count, data, d->usagePattern);
    } else {
        std::cout << "Buffer not created";
        return;
    }
}

void* OpenGLBuffer::map(OpenGLBuffer::Access access) const {
    if (isCreated()) {
        D(const OpenGLBuffer);
        return glMapBuffer(d->type, access);
    } else {
        std::cout << "Buffer not created";
        return nullptr;
    }
}

void* OpenGLBuffer::mapRange(int offset, int count, OpenGLBuffer::RangeAccessFlag access) const {
    if (isCreated()) {
        D(const OpenGLBuffer);
        return glMapBufferRange(d->type, offset, count, access);
    } else {
        std::cout << "Buffer not created";
        return nullptr;
    }
}

bool OpenGLBuffer::unmap() {
    if (isCreated()) {
        D(const OpenGLBuffer);
        return glUnmapBuffer(d->type) == GL_TRUE;
    } else {
        std::cout << "Buffer not created";
        return false;
    }
}