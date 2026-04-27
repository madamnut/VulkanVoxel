#pragma once

#include <utility>

constexpr float kPi = 3.14159265358979323846f;

struct Vec3
{
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

struct Mat4
{
    float m[16]{};
};

Vec3 operator+(Vec3 lhs, Vec3 rhs);
Vec3 operator-(Vec3 lhs, Vec3 rhs);
Vec3 operator*(Vec3 value, float scalar);
Vec3 operator-(Vec3 value);

float dot(Vec3 lhs, Vec3 rhs);
Vec3 cross(Vec3 lhs, Vec3 rhs);
Vec3 normalize(Vec3 value);
float length(Vec3 value);

Vec3 cameraForward(float yaw, float pitch);
Vec3 cameraRight(float yaw);

Mat4 multiply(Mat4 lhs, Mat4 rhs);
Mat4 makeViewMatrix(Vec3 position, float yaw, float pitch);
Mat4 makeViewMatrixFromForward(Vec3 position, Vec3 forward);
std::pair<float, float> yawPitchFromForward(Vec3 forward);
Mat4 makePerspectiveMatrix(float fovYRadians, float aspect, float nearPlane, float farPlane);
