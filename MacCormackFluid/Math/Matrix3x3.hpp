#ifndef MATRIX3X3_HPP
#define MATRIX3X3_HPP

namespace Math {

class Matrix3x3 {
public:
    void identity();
    Matrix3x3 operator * (Matrix3x3& m2) const;
    void rotateAroundAxis(float xa, float ya, float za, float angle);
    float& operator () (int row, int col);
    operator float* ();

private:
    float m[3][3];
};

} // namespace Math

#endif