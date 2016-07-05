#ifndef ALGEBRA_HPP
#define ALGEBRA_HPP

namespace Utils {

class Mat3 {
public:
    void identity();
    Mat3 operator * (Mat3& m2) const;

    void rotateAroundAxis(float xa, float ya, float za, float angle);

    float& operator () (int row, int col);

    operator float* ();

private:
    float m[3][3];
};

} // namespace Utils


#endif