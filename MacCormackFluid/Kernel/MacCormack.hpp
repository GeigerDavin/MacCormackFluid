#ifndef MAC_CORMACK_HPP
#define MAC_CORMACK_HPP

namespace Kernel {

void computeCube(Utils::Vector<GLfloat>& vertices, Utils::Vector<GLfloat>& colors);

void advect(float4* speedOut, size_t size);
void advectBackward(float4* speedOut, size_t size);

void advectMacCormack(float4* speedOut, size_t size);

void drawSphere(float4* speedOut, size_t size);

void divergence(float4* divergenceOut, size_t size);

void jacobi(float4* pressureOut, size_t size);

void project(float4* speedOut, float4* speedSizeOut, size_t size);

} // namespace Kernel

#endif