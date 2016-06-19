//#ifndef VECTOR_TYPES_HPP
//#define VECTOR_TYPES_HPP
//
//
//
//// Addition
//HDINLINE xyz operator + (const xyz& a, float b) {
//    return make_xyzw(a.x + b, a.y + b, a.z + b, a.w + b);
//}
//
//HDINLINE xyz operator + (const xyz& a, const xyz& b) {
//    return make_xyzw(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
//}
//
//HDINLINE xyz& operator += (xyz& a, float b) {
//    a = a + b;
//    return a;
//}
//
//HDINLINE xyz& operator += (xyz& a, const xyz& b) {
//    a = a + b;
//    return a;
//}
//
//// Subtraction
//HDINLINE xyz operator - (const xyz& a, float b) {
//    return make_xyzw(a.x - b, a.y - b, a.z - b, a.w - b);
//}
//
//HDINLINE xyz operator - (const xyz& a, const xyz& b) {
//    return make_xyzw(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
//}
//
//HDINLINE xyz& operator -= (xyz& a, float b) {
//    a = a - b;
//    return a;
//}
//
//HDINLINE xyz& operator -= (xyz& a, const xyz& b) {
//    a = a - b;
//    return a;
//}
//
//HDINLINE xyz operator - (const xyz& v) {
//    return make_xyzw(-v.x, -v.y, -v.z, v.w);
//}
//
//// Multiplication
//HDINLINE xyz operator * (const xyz& a, float b) {
//    return make_xyzw(a.x * b, a.y * b, a.z * b, a.w * b);
//}
//
//HDINLINE xyz operator * (float b, const xyz& a) {
//    return a * b;
//}
//
//HDINLINE xyz operator * (const xyz& a, const xyz& b) {
//    return make_xyzw(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
//}
//
//HDINLINE xyz& operator *= (xyz& a, float b) {
//    a = a * b;
//    return a;
//}
//
//HDINLINE xyz& operator *= (xyz& a, const xyz& b) {
//    a = a * b;
//    return a;
//}
//
//// Division
//HDINLINE xyz operator / (const xyz& a, float b) {
//    return make_xyzw(a.x / b, a.y / b, a.z / b, a.w / b);
//}
//
//HDINLINE xyz operator / (float b, const xyz& a) {
//    return make_xyzw(b / a.x, b / a.y, b / a.z, b / a.w);
//}
//
//HDINLINE xyz operator / (const xyz& a, const xyz& b) {
//    return make_xyzw(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
//}
//
//HDINLINE xyz& operator /= (xyz& a, float b) {
//    a = a / b;
//    return a;
//}
//
//HDINLINE xyz& operator /= (xyz& a, const xyz& b) {
//    a = a / b;
//    return a;
//}
//
//// Helper
//HDINLINE float dot(const xyz& a, const xyz& b) {
//    return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
//}
//
//HDINLINE float dot3D(const xyz& a, const xyz& b) {
//    return a.x*b.x + a.y*b.y + a.z*b.z;
//}
//
//HDINLINE float dot2D(const xyz& a, const xyz& b) {
//    return a.x*b.x + a.y*b.y;
//}
//
//HDINLINE xyz cross(const xyz& a, const xyz& b) {
//    return make_xyz(a.y*b.z - a.z*b.y,
//                    a.z*b.x - a.x*b.z,
//                    a.x*b.y - a.y*b.x);
//}
//
//HDINLINE float length(const xyz& v) {
//    return sqrtf(dot(v, v));
//}
//
//HDINLINE float length3D(const xyz& v) {
//    return sqrtf(dot3D(v, v));
//}
//
//HDINLINE float length2D(const xyz& v) {
//    return sqrtf(dot2D(v, v));
//}
//
//HDINLINE float lensq(const xyz& v) {
//    return dot(v, v);
//}
//
//HDINLINE float lensq3D(const xyz& v) {
//    return dot3D(v, v);
//}
//
//HDINLINE float lensq2D(const xyz& v) {
//    return dot2D(v, v);
//}
//
//HDINLINE xyz normalize(const xyz& v) {
//    float len = length(v);
//    if (len < 0.00001f) {
//        return v;
//    }
//    return v / len;
//}
//
//HDINLINE xyz xyz_min(const xyz& a, const xyz& b) {
//    return make_xyz(minf(a.x, b.x), minf(a.y, b.y), minf(a.z, b.z));
//}
//
//HDINLINE xyz xyz_max(const xyz& a, const xyz& b) {
//    return make_xyz(maxf(a.x, b.x), maxf(a.y, b.y), maxf(a.z, b.z));
//}
//
//#endif