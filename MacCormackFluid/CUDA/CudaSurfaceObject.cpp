#include "../StdAfx.hpp"
#include "CudaSurfaceObject.hpp"

#include "Modules/SurfaceObjectManagement.hpp"

#include "../Utils/ResourceGuard.hpp"

#include <cstring>

namespace CUDA {

class CudaSurfaceObjectPrivate {
public:
    CudaSurfaceObjectPrivate(CudaSurfaceObject::ResourceType type)
        : resourceType(type)
        , surfaceGuard(nullptr) {

        memset(&channelDesc, 0, sizeof(cudaChannelFormatDesc));
    }

    ~CudaSurfaceObjectPrivate() {
        destroy();
    }

    bool create(void* data);
    void destroy();
    bool isValid() const;

    cudaChannelFormatDesc channelDesc;
    CudaSurfaceObject::ResourceType resourceType;
    Utils::ResourceGuard<cudaSurfaceObject_t>* surfaceGuard;
};

namespace {
    void freeSurfaceFunc(cudaSurfaceObject_t surf) {
        std::cout << "Destroy surface " << surf << std::endl;
        SurfaceObjectManagement::destroySurfaceObject(surf);
    }
}

bool CudaSurfaceObjectPrivate::create(void* data) {
    if (isValid()) {
        return true;
    }

    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(cudaResourceDesc));

    resDesc.resType = static_cast<cudaResourceType>(resourceType);
    switch (resourceType) {
    case CudaSurfaceObject::ResourceType::Array:
        resDesc.res.array.array = static_cast<cudaArray_t>(data);
        break;
    case CudaSurfaceObject::ResourceType::Mipmap:
        resDesc.res.mipmap.mipmap = static_cast<cudaMipmappedArray_t>(data);
        break;
    case CudaSurfaceObject::ResourceType::Linear:
        resDesc.res.linear.devPtr = data;
        resDesc.res.linear.desc = channelDesc;
        break;
    case CudaSurfaceObject::ResourceType::Pitch2D:
        break;
    }

    cudaSurfaceObject_t surf = 0;
    SurfaceObjectManagement::createSurfaceObject(&surf, &resDesc);
    if (surf) {
        destroy();
        surfaceGuard = new Utils::ResourceGuard<cudaSurfaceObject_t>(surf, freeSurfaceFunc);
        return isValid();
    } else {
        std::cerr << "Could not create surface object" << std::endl;
        return false;
    }
}

void CudaSurfaceObjectPrivate::destroy() {
    _delete(surfaceGuard);
}

bool CudaSurfaceObjectPrivate::isValid() const {
    return (surfaceGuard) && (surfaceGuard->get() != 0);
}

CudaSurfaceObject::CudaSurfaceObject()
    : dPtr(new CudaSurfaceObjectPrivate(ResourceType::Array)) {}

CudaSurfaceObject::CudaSurfaceObject(ResourceType type)
    : dPtr(new CudaSurfaceObjectPrivate(type)) {}

CudaSurfaceObject::CudaSurfaceObject(CudaSurfaceObject&& other) {
    dPtr = std::move(other.dPtr);
    other.dPtr = nullptr;
}

CudaSurfaceObject::~CudaSurfaceObject() {
    destroy();
}

CudaSurfaceObject& CudaSurfaceObject::operator = (CudaSurfaceObject&& other) {
    dPtr = std::move(other.dPtr);
    other.dPtr = nullptr;
    return *this;
}

void CudaSurfaceObject::setResourceType(ResourceType type) {
    D(CudaSurfaceObject);
    d->resourceType = type;
}

CudaSurfaceObject::ResourceType CudaSurfaceObject::getResourceType() const {
    D(const CudaSurfaceObject);
    return d->resourceType;
}

bool CudaSurfaceObject::create(void* data) {
    D(CudaSurfaceObject);
    return d->create(data);
}

bool CudaSurfaceObject::isCreated() const {
    D(const CudaSurfaceObject);
    return d->isValid();
}

void CudaSurfaceObject::destroy() {
    _delete(dPtr);
}

cudaSurfaceObject_t CudaSurfaceObject::getId() const {
    D(const CudaSurfaceObject);
    return (d->surfaceGuard) ? (d->surfaceGuard->get()) : (0);
}

} // namespace CUDA
