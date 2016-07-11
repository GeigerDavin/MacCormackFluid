#ifndef SIMPLE_ROTATION_CAMERA_HPP
#define SIMPLE_ROTATION_CAMERA_HPP

#include "../Math/Matrix3x3.hpp"

namespace Camera {

class SimpleRotationCamera {
public:
    SimpleRotationCamera();

    void pointOnCamera(float& x, float& y, float& z, int mouseX, int mouseY, int mouseWidth, int mouseHeight);

    void dragStart(int mouseX, int mouseY, int mouseWidth, int mouseHeight);
    void dragMove(int mouseX, int mouseY, int mouseWidth, int mouseHeight);
    void dragEnd();

    Math::Matrix3x3 getRotationMatrix();

private:
    float x0, y0, z0;
    float x1, y1, z1;

    Math::Matrix3x3 rotate;
    Math::Matrix3x3 orientation;
};

} // namespace Camera

#endif