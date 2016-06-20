#ifndef VECTOR_HPP
#define VECTOR_HPP

#include "../CUDA/Allocator.hpp"
#include "../CUDA/OpenGLInterop.hpp"

#include "../OpenGL/OpenGLBuffer.hpp"
#include "../OpenGL/OpenGLShaderProgram.hpp"

namespace Utils {

template <class T>
class Vector {
public:
    inline Vector(size_t count)
        : size(count)
        , elementsHost(nullptr)
        , elementsDevice(nullptr)
        , graphicsResource(nullptr) {

        elementsHost = (T *) CUDA::mallocHost(size * sizeof(T));
        buffer.create();
        buffer.setUsagePattern(OpenGL::OpenGLBuffer::DynamicDraw);
        buffer.bind();
        buffer.bind();
        buffer.allocate(elementsHost, size * sizeof(T));
        buffer.release();
        CUDA::registerGraphicsResource(&graphicsResource, buffer.getId());
    }

    inline Vector(const T* data, size_t count)
        : size(count)
        , elementsHost(nullptr)
        , elementsDevice(nullptr)
        , graphicsResource(nullptr) {

        elementsHost = (T *) CUDA::mallocHost(size * sizeof(T));
        CUDA::moveHostToHost(elementsHost, data, size * sizeof(T));
        buffer.create();
        buffer.setUsagePattern(OpenGL::OpenGLBuffer::DynamicDraw);
        buffer.bind();
        buffer.allocate(elementsHost, size * sizeof(T));
        buffer.release();
        CUDA::registerGraphicsResource(&graphicsResource, buffer.getId());
    }

    inline ~Vector() {
        if (graphicsResource) {
            CUDA::unregisterGraphicsResource(&graphicsResource);
        }
        CUDA::freeHost(elementsHost);
        CUDA::freeDevice(elementsDevice);
    }

public:
    inline void sendData();
    inline void recvData();
    inline void setChanged(bool gpu);

    inline OpenGL::OpenGLBuffer* getBuffer() {
        return &buffer;
    }

    inline T* bindGraphicsResource() {
        if (!graphicsResource) {
            return nullptr;
        }
        return (T *) CUDA::mapGraphicsResource(&graphicsResource);
    }

    inline void unbindGraphicsResource() {
        CUDA::unmapGraphicsResource(graphicsResource);
    }

public:
    inline size_t getSize() const {
        return size;
    }
    inline T* getHost() {
        return elementsHost;
    }
    inline const T* getHost() const {
        return elementsHost;
    }
    inline T* getDevice() {
        return elementsDevice;
    }
    inline const T* getDevice() const {
        return elementsDevice;
    }

private:
    int location;
    size_t size;
    T* elementsHost;
    T* elementsDevice;

    OpenGL::OpenGLBuffer buffer;
    cudaGraphicsResource* graphicsResource;
};

} // namespace Utils

#endif