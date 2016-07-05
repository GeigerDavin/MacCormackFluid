#ifndef TRACK_SPHERE_HPP
#define TRACK_SPHERE_HPP

#include "Algebra.hpp"

namespace Utils {

class TrackSphere {
public:
    TrackSphere();

    void pointOnTrackSphere(float& x, float& y, float& z, int mouseX, int mouseY, int mouseWidth, int mouseHeight);

    void dragStart(int mouseX, int mouseY, int mouseWidth, int mouseHeight);
    void dragMove(int mouseX, int mouseY, int mouseWidth, int mouseHeight);
    void dragEnd();

    Mat3 getRotationMatrix();


private:
    float x0, y0, z0;
    float x1, y1, z1;

    Mat3 rotate;
    Mat3 orientation;
};

} // namespace Utils

#endif