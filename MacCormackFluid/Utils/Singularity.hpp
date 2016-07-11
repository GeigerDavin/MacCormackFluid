#ifndef SINGULARITY_HPP
#define SINGULARITY_HPP

namespace Utils {

template <class T>
class Singularity {
    DISABLE_COPY(Singularity)

public:
    Singularity() {
        _assert(!singularity);
        singularity = static_cast<T *>(this);
    }

    virtual ~Singularity() {
        _assert(singularity);
        singularity = nullptr;
    }

public:
    static T& getSingularity() {
        _assert(singularity);
        return *singularity;
    }

    static T* getSingularityPtr() {
        return singularity;
    }

protected:
    static T* singularity;
};

template <class T>
T* Singularity<T>::singularity = nullptr;

} // namespace Utils

#endif