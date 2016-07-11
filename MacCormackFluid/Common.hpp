#ifndef COMMON_HPP
#define COMMON_HPP

#ifndef NOEXCEPT
# define NOEXCEPT throw()
#endif

#define DISABLE_COPY(Class)                                                         \
    private:                                                                        \
        Class(const Class &) = delete;                                              \
        Class& operator = (const Class &) = delete;

template <class T> static inline T* getPtrHelper(T* ptr) { return ptr; }
template <class Wrapper> static inline typename Wrapper::pointer getPtrHelper
    (const Wrapper &p) { return p.get(); }

#define DECLARE_PRIVATE(Class)                                                      \
    inline Class##Private* dFunc()                                                  \
        { return reinterpret_cast<Class##Private *>(getPtrHelper(dPtr)); }          \
    inline const Class##Private* dFunc() const                                      \
        { return reinterpret_cast<const Class##Private *>(getPtrHelper(dPtr)); }    \
    friend class Class##Private;

#define DECLARE_PRIVATE_D(dPtr, Class)                                              \
    inline Class##Private* dFunc()                                                  \
        { return reinterpret_cast<Class##Private *>(dPtr); }                        \
    inline const Class##Private* dFunc() const                                      \
        { return reinterpret_cast<const Class##Private *>(dPtr); }                  \
    friend class Class##Private;

#define DECLARE_PUBLIC(Class)                                                       \
    inline Class* qFunc()                                                           \
        { return static_cast<Class *>(qPtr); }                                      \
    inline const Class* qFunc() const                                               \
        { return static_cast<const Class *>(qPtr); }                                \
    friend class Class;

#define D(Class) Class##Private* const d = dFunc()
#define Q(Class) Class* const q = qFunc()

#ifndef PI
# define PI 3.14159265f
#endif

#define TID                                                                         \
    uint3 tid = make_uint3(blockIdx.x * blockDim.x + threadIdx.x,                   \
                           blockIdx.y * blockDim.y + threadIdx.y,                   \
                           blockIdx.z * blockDim.z + threadIdx.z)

#define TID_CONST                                                                   \
    const uint3 tid = make_uint3(blockIdx.x * blockDim.x + threadIdx.x,             \
                                 blockIdx.y * blockDim.y + threadIdx.y,             \
                                 blockIdx.z * blockDim.z + threadIdx.z)

typedef unsigned int uint;
typedef unsigned char uchar;
typedef unsigned long ulong;
typedef unsigned long long ulonglong;
typedef std::string String;

#define HINLINE __host__ inline
#define DINLINE __forceinline__ __device__
#define HDINLINE __forceinline__ __device__ __host__

#define _delete(Var) if((Var)) { delete (Var); Var = NULL; }
#define _assert(Expression) if (!Expression) std::cout << "Assertion failed for" << #Expression

#define CUDA_BOUNDARY_MODE cudaBoundaryModeZero

#endif
