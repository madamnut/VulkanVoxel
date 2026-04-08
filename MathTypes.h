#pragma once

#include <cmath>

struct Vec2 {
    float x = 0.0f;
    float y = 0.0f;
};

struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

struct Mat4 {
    float m[16] = {};
};

inline Vec3 operator+(const Vec3& a, const Vec3& b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

inline Vec3 operator-(const Vec3& a, const Vec3& b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

inline Vec3 operator*(const Vec3& v, float s) {
    return {v.x * s, v.y * s, v.z * s};
}

inline float Dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Vec3 Cross(const Vec3& a, const Vec3& b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    };
}

inline float Length(const Vec3& v) {
    return std::sqrt(Dot(v, v));
}

inline Vec3 Normalize(const Vec3& v) {
    const float len = Length(v);
    if (len <= 0.00001f) {
        return {};
    }

    return {v.x / len, v.y / len, v.z / len};
}

inline Mat4 Identity() {
    Mat4 mat{};
    mat.m[0] = 1.0f;
    mat.m[5] = 1.0f;
    mat.m[10] = 1.0f;
    mat.m[15] = 1.0f;
    return mat;
}

inline Mat4 Multiply(const Mat4& a, const Mat4& b) {
    Mat4 out{};

    for (int column = 0; column < 4; ++column) {
        for (int row = 0; row < 4; ++row) {
            out.m[column * 4 + row] =
                a.m[0 * 4 + row] * b.m[column * 4 + 0] +
                a.m[1 * 4 + row] * b.m[column * 4 + 1] +
                a.m[2 * 4 + row] * b.m[column * 4 + 2] +
                a.m[3 * 4 + row] * b.m[column * 4 + 3];
        }
    }

    return out;
}

inline Mat4 Perspective(float fovYRadians, float aspect, float nearPlane, float farPlane) {
    Mat4 mat{};

    const float tanHalfFov = std::tan(fovYRadians * 0.5f);
    mat.m[0] = 1.0f / (aspect * tanHalfFov);
    mat.m[5] = -1.0f / tanHalfFov;
    mat.m[10] = farPlane / (nearPlane - farPlane);
    mat.m[11] = -1.0f;
    mat.m[14] = (nearPlane * farPlane) / (nearPlane - farPlane);

    return mat;
}

inline Mat4 LookAt(const Vec3& eye, const Vec3& target, const Vec3& up) {
    const Vec3 forward = Normalize(target - eye);
    const Vec3 right = Normalize(Cross(forward, up));
    const Vec3 cameraUp = Cross(right, forward);

    Mat4 mat = Identity();
    mat.m[0] = right.x;
    mat.m[1] = cameraUp.x;
    mat.m[2] = -forward.x;
    mat.m[4] = right.y;
    mat.m[5] = cameraUp.y;
    mat.m[6] = -forward.y;
    mat.m[8] = right.z;
    mat.m[9] = cameraUp.z;
    mat.m[10] = -forward.z;
    mat.m[12] = -Dot(right, eye);
    mat.m[13] = -Dot(cameraUp, eye);
    mat.m[14] = Dot(forward, eye);

    return mat;
}
