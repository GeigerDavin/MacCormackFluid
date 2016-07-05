#include "../StdAfx.hpp"
#include "CudaGLGraphicsResource.hpp"

#include "Modules/OpenGLInteroperability.hpp"
#include "Modules/GraphicsInteroperability.hpp"

#include "../Utils/ResourceGuard.hpp"

namespace CUDA {

class CudaGLGraphicsResourcePrivate {
public:
    CudaGLGraphicsResourcePrivate(CudaGLGraphicsResource::Type t)
        : type(t)
        , stream(nullptr)
        , resource(nullptr)
        , imageTarget(CudaGLGraphicsResource::ImageTarget::Texture2D)
        , registerFlag(CudaGLGraphicsResource::RegisterFlag::None) {}

    ~CudaGLGraphicsResourcePrivate() {
        destroy();
    }

    bool create(GLuint index);
    void destroy();
    bool isValid() const;

    cudaStream_t stream;
    cudaGraphicsResource_t resource;
    CudaGLGraphicsResource::Type type;
    CudaGLGraphicsResource::ImageTarget imageTarget;
    CudaGLGraphicsResource::RegisterFlag registerFlag;
};

bool CudaGLGraphicsResourcePrivate::create(GLuint index) {
    if (isValid()) {
        return true;
    }

    if (index) {
        destroy();
        cudaGraphicsResource_t res = nullptr;
        switch (type) {
        case CudaGLGraphicsResource::Type::Image:
            OpenGLInteroperability::registerGraphicsImage(&res, index,
                static_cast<GLenum>(imageTarget), static_cast<uint>(registerFlag));
            break;
        case CudaGLGraphicsResource::Type::Buffer:
            OpenGLInteroperability::registerGraphicsBuffer(&res, index,
                static_cast<uint>(registerFlag));
            break;
        }
        if (res) {
            resource = res;
            return isValid();
        } else {
            std::cerr << "Could not create graphics resource" << std::endl;
            return false;
        }
    } else {
        std::cerr << "Invalid OpenGL index" << std::endl;
        return false;
    }
}

void CudaGLGraphicsResourcePrivate::destroy() {
    if (resource) {
        std::cout << "Free Resource Func" << std::endl;
        GraphicsInteroperability::unregisterGraphicsResource(&resource);
    }
}

bool CudaGLGraphicsResourcePrivate::isValid() const {
    return (resource != 0);
}

CudaGLGraphicsResource::CudaGLGraphicsResource()
    : dPtr(new CudaGLGraphicsResourcePrivate(Type::Buffer)) {}

CudaGLGraphicsResource::CudaGLGraphicsResource(Type type)
    : dPtr(new CudaGLGraphicsResourcePrivate(type)) {}

CudaGLGraphicsResource::~CudaGLGraphicsResource() {
    destroy();
}

void CudaGLGraphicsResource::setType(Type type) {
    D(CudaGLGraphicsResource);
    d->type = type;
}

CudaGLGraphicsResource::Type CudaGLGraphicsResource::getType() const {
    D(const CudaGLGraphicsResource);
    return d->type;
}

void CudaGLGraphicsResource::setRegisterFlag(RegisterFlag flag) {
    D(CudaGLGraphicsResource);
    d->registerFlag = flag;
}

CudaGLGraphicsResource::RegisterFlag CudaGLGraphicsResource::getRegisterFlag() const {
    D(const CudaGLGraphicsResource);
    return d->registerFlag;
}

void CudaGLGraphicsResource::setStream(cudaStream_t stream) {
    D(CudaGLGraphicsResource);
    d->stream = stream;
}

cudaStream_t CudaGLGraphicsResource::getStream() const {
    D(const CudaGLGraphicsResource);
    return d->stream;
}

bool CudaGLGraphicsResource::create(GLuint id) {
    D(CudaGLGraphicsResource);
    return d->create(id);
}

bool CudaGLGraphicsResource::isCreated() const {
    D(const CudaGLGraphicsResource);
    return d->isValid();
}

void CudaGLGraphicsResource::destroy() {
    _delete(dPtr);
}

void* CudaGLGraphicsResource::map(MapFlag flag) {
    if (isCreated()) {
        D(CudaGLGraphicsResource);
        return GraphicsInteroperability::mapGraphicsResourcePointer(&d->resource, d->stream);
    } else {
        std::cerr << "Graphics resource not created" << std::endl;
        return nullptr;
    }
}

cudaArray_t CudaGLGraphicsResource::mapArray(MapFlag flag) const {
    return nullptr;
}

cudaMipmappedArray_t CudaGLGraphicsResource::mapMipmappedArray(MapFlag flag) const {
    return nullptr;
}

void CudaGLGraphicsResource::unmap() const {
    if (isCreated()) {
        D(const CudaGLGraphicsResource);
        GraphicsInteroperability::unmapGraphicsResource(d->resource, d->stream);
    } else {
        std::cerr << "Graphics resource not created" << std::endl;
    }
}

} // namespace CUDA