#include "../StdAfx.hpp"
#include "SimpleRotationCamera.hpp"

namespace Camera {

#ifndef min
#define min(x, y) ((x < y) ? x : y)
#endif
#ifndef max
#define max(x, y) ((x > y) ? x : y)
#endif

SimpleRotationCamera::SimpleRotationCamera() {
    orientation.identity();
    rotate.identity();
}

void SimpleRotationCamera::pointOnCamera(float& x, float& y, float& z, int mouseX, int mouseY, int mouseWidth, int mouseHeight) {
    x = (float)(mouseX - mouseWidth / 2) / (float)min(mouseWidth, mouseHeight) * 2;
    y = (float)(mouseY - mouseHeight / 2) / (float)min(mouseWidth, mouseHeight) * 2;

    float rr = x*x+y*y;

    if (rr < 1) {
        z = sqrt(1 - rr);
    } else {
        x = x / sqrt(rr);
        y = y / sqrt(rr);
        z = 0;
    }
}

void SimpleRotationCamera::dragStart(int mouseX, int mouseY, int mouseWidth, int mouseHeight) {
    pointOnCamera(x0, y0, z0, mouseX, mouseY, mouseWidth, mouseHeight);
}
void SimpleRotationCamera::dragMove(int mouseX, int mouseY, int mouseWidth, int mouseHeight) {
    pointOnCamera(x1, y1, z1, mouseX, mouseY, mouseWidth, mouseHeight);

    float xa =  y0 * z1 - y1*z0;
    float ya = z0*x1 - z1*x0;
    float za = x0 * y1 - x1 * y0;

    float angle = 2 * acos(min(max(x0*x1+y0*y1+z0*z1, -1), 1));

    rotate.rotateAroundAxis(xa, ya, za, angle);
}
void SimpleRotationCamera::dragEnd() {
    orientation = orientation * rotate;
    rotate.identity();
}

Math::Matrix3x3 SimpleRotationCamera::getRotationMatrix() {
    return orientation * rotate;
}

} // namespace Camera