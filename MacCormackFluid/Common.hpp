#ifndef COMMON_HPP
#define COMMON_HPP

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
#ifndef THREADS
# define THREADS 32
#endif

#define xyz float4
#define make_xyz(x, y, z) make_float4(x, y, z, 0)
#define make_xyzw(x, y, z, w) make_float4(x, y, z, w)

#define TIDX(count) \
    const uint tidX = blockIdx.x * blockDim.x + threadIdx.x; \
    if (tidX >= count) return

typedef unsigned int uint;
typedef unsigned char uchar;
typedef unsigned long ulong;
typedef unsigned long long ulonglong;
typedef std::string String;

#define HINLINE __host__ inline
#define DINLINE __device__ __forceinline__
#define HDINLINE __device__ __host__ __forceinline__

#define _delete(Var) if((Var)) { delete (Var); Var = NULL; }
#define _assert(Expression) if (!Expression) std::cout << "Assertion failed for" << #Expression

#endif