#include "../StdAfx.hpp"
#include "CudaTextureObject.hpp"

#include "Modules/TextureObjectManagement.hpp"

#include "../Utils/ResourceGuard.hpp"

namespace CUDA {

class CudaTextureObjectPrivate {
public:
    CudaTextureObjectPrivate(CudaTextureObject::ResourceType type)
        : resourceType(type)
        , normalized(false)
        , readMode(CudaTextureObject::ReadMode::ElementType)
        , filterMode(CudaTextureObject::FilterMode::PointFilter)
        , textureGuard(nullptr) {

        addressMode[0] = CudaTextureObject::AddressMode::Wrap;
        addressMode[1] = CudaTextureObject::AddressMode::Wrap;
        addressMode[2] = CudaTextureObject::AddressMode::Wrap;

        memset(&channelDesc, 0, sizeof(cudaChannelFormatDesc));
    }

    ~CudaTextureObjectPrivate() {
        destroy();
    }

    bool create(void* data);
    void destroy();
    bool isValid() const;

    bool normalized;
    cudaChannelFormatDesc channelDesc;
    CudaTextureObject::ReadMode readMode;
    CudaTextureObject::FilterMode filterMode;
    CudaTextureObject::FilterMode mipmapFilterMode;
    CudaTextureObject::ResourceType resourceType;
    CudaTextureObject::AddressMode addressMode[3];
    Utils::ResourceGuard<cudaTextureObject_t>* textureGuard;
};

namespace {
    void freeTextureFunc(cudaTextureObject_t tex) {
        std::cout << "Destroy texture " << tex << std::endl;
        TextureObjectManagement::destroyTextureObject(tex);
    }
}

bool CudaTextureObjectPrivate::create(void* data) {
    if (isValid()) {
        return true;
    }

    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(cudaResourceDesc));

    resDesc.resType = static_cast<cudaResourceType>(resourceType);
    switch (resourceType) {
    case CudaTextureObject::ResourceType::Array:
        resDesc.res.array.array = static_cast<cudaArray_t>(data);
        break;
    case CudaTextureObject::ResourceType::Mipmap:
        normalized = true;
        resDesc.res.mipmap.mipmap = static_cast<cudaMipmappedArray_t>(data);
        break;
    case CudaTextureObject::ResourceType::Linear:
        resDesc.res.linear.devPtr = data;
        resDesc.res.linear.desc = channelDesc;
        break;
    case CudaTextureObject::ResourceType::Pitch2D:
        break;
    }

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(cudaTextureDesc));

    texDesc.normalizedCoords = normalized;
    texDesc.readMode = static_cast<cudaTextureReadMode>(readMode);
    texDesc.filterMode = static_cast<cudaTextureFilterMode>(filterMode);
    //texDesc.mipmapFilterMode = static_cast<cudaTextureFilterMode>(mipmapFilterMode);
    for (int i = 0; i < 3; i++) {
        texDesc.addressMode[i] = static_cast<cudaTextureAddressMode>(addressMode[i]);
    }

    cudaTextureObject_t tex = 0;
    TextureObjectManagement::createTextureObject(&tex, &resDesc, &texDesc, NULL);
    if (tex) {
        destroy();
        textureGuard = new Utils::ResourceGuard<cudaTextureObject_t>(tex, freeTextureFunc);
        return isValid();
    } else {
        std::cerr << "Could not create texture object" << std::endl;
        return false;
    }
}

void CudaTextureObjectPrivate::destroy() {
    _delete(textureGuard);
}

bool CudaTextureObjectPrivate::isValid() const {
    return (textureGuard) && (textureGuard->get() != 0);
}

CudaTextureObject::CudaTextureObject()
    : dPtr(new CudaTextureObjectPrivate(ResourceType::Array)) {}

CudaTextureObject::CudaTextureObject(ResourceType type)
    : dPtr(new CudaTextureObjectPrivate(type)) {}

CudaTextureObject::CudaTextureObject(CudaTextureObject&& other) {
    dPtr = std::move(other.dPtr);
    other.dPtr = nullptr;
}

CudaTextureObject::~CudaTextureObject() {
    destroy();
}

CudaTextureObject& CudaTextureObject::operator = (CudaTextureObject&& other) {
    dPtr = std::move(other.dPtr);
    other.dPtr = nullptr;
    return *this;
}

void CudaTextureObject::setResourceType(ResourceType type) {
    D(CudaTextureObject);
    d->resourceType = type;
}

CudaTextureObject::ResourceType CudaTextureObject::getResourceType() const {
    D(const CudaTextureObject);
    return d->resourceType;
}

void CudaTextureObject::setReadMode(ReadMode mode) {
    D(CudaTextureObject);
    d->readMode = mode;
}

CudaTextureObject::ReadMode CudaTextureObject::getReadMode() const {
    D(const CudaTextureObject);
    return d->readMode;
}

void CudaTextureObject::setFilterMode(FilterMode mode) {
    D(CudaTextureObject);
    d->filterMode = mode;
}

CudaTextureObject::FilterMode CudaTextureObject::getFilterMode() const {
    D(const CudaTextureObject);
    return d->filterMode;
}

void CudaTextureObject::setAddressMode0(AddressMode mode) {
    D(CudaTextureObject);
    d->addressMode[0] = mode;
}

CudaTextureObject::AddressMode CudaTextureObject::getAddressMode0() const {
    D(const CudaTextureObject);
    return d->addressMode[0];
}

void CudaTextureObject::setAddressMode1(AddressMode mode) {
    D(CudaTextureObject);
    d->addressMode[1] = mode;
}

CudaTextureObject::AddressMode CudaTextureObject::getAddressMode1() const {
    D(const CudaTextureObject);
    return d->addressMode[1];
}

void CudaTextureObject::setAddressMode2(AddressMode mode) {
    D(CudaTextureObject);
    d->addressMode[2] = mode;
}

CudaTextureObject::AddressMode CudaTextureObject::getAddressMode2() const {
    D(const CudaTextureObject);
    return d->addressMode[2];
}

void CudaTextureObject::setNormalized(bool normalized) {
    D(CudaTextureObject);
    d->normalized = normalized;
}

bool CudaTextureObject::isNormalized() const {
    D(const CudaTextureObject);
    return d->normalized;
}

bool CudaTextureObject::create(void* data) {
    D(CudaTextureObject);
    return d->create(data);
}

bool CudaTextureObject::isCreated() const {
    D(const CudaTextureObject);
    return d->isValid();
}

void CudaTextureObject::destroy() {
    _delete(dPtr);
}

cudaTextureObject_t CudaTextureObject::getId() const {
    D(const CudaTextureObject);
    return (d->textureGuard) ? (d->textureGuard->get()) : (0);
}

} // namespace CUDA
