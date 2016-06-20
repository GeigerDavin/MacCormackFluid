#ifndef RESOURCE_GUARD_HPP
#define RESOURCE_GUARD_HPP

namespace Utils {

template <class T>
class ResourceGuard {
    DISABLE_COPY(ResourceGuard)

public:
    typedef void (*FreeResourceFunc)(T res);
    inline ResourceGuard(const T& res, FreeResourceFunc f)
        : resource(res)
        , func(f) {}

    inline ~ResourceGuard() {
        freeResource();
    }

public:
    inline const T& get() const {
        return resource;
    }

private:
    inline void freeResource() {
        if (resource && func) {
            func(resource);
            invalidateResource();
        }
    }

    inline void invalidateResource() {
        resource = 0;
    }

private:
    T resource;
    FreeResourceFunc func;
};

} // namespace Utils

#endif