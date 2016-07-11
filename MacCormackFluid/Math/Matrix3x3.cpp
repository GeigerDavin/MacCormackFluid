#include "../StdAfx.hpp"
#include "Matrix3x3.hpp"

namespace Math {

void Matrix3x3::identity() {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            m[j][i] = 0;
        }
    }
    m[0][0] = 1;
    m[1][1] = 1;
    m[2][2] = 1;
}

Matrix3x3 Matrix3x3::operator * (Matrix3x3& m2) const {
    Matrix3x3 res;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            res(j, i) = 0;
            for (int k = 0; k < 3; k++) {
                res(j, i) += m[j][k] * m2(k, i);
            }
        }
    }
    return res;
}

float& Matrix3x3::operator () (int row, int col) {
    return m[row][col];
}

Matrix3x3::operator float * () {
    return &m[0][0];
}

void Matrix3x3::rotateAroundAxis(float xa, float ya, float za, float angle) {
    float t = sqrt(xa * xa + ya * ya + za * za);
    if (t != 0) {
        xa/=t;
        ya/=t;
        za/=t;
    }

    m[0][0] = 1 + (1 - cos(angle)) * (xa * xa - 1);
    m[1][1] = 1 + (1 - cos(angle)) * (ya * ya - 1);
    m[2][2] = 1 + (1 - cos(angle)) * (za * za - 1);

    m[0][1] = +za*sin(angle) + (1 - cos(angle))*xa*ya;
    m[1][0] = -za*sin(angle) + (1 - cos(angle))*xa*ya;

    m[0][2] = -ya*sin(angle) + (1 - cos(angle))*xa*za;
    m[2][0] = +ya*sin(angle) + (1 - cos(angle))*xa*za;

    m[1][2] = +xa*sin(angle) + (1 - cos(angle))*ya*za;
    m[2][1] = -xa*sin(angle) + (1 - cos(angle))*ya*za;
}

} // namespace Math