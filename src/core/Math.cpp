#include "core/Math.h"

#include <algorithm>
#include <cmath>

Vec3 operator+(Vec3 lhs, Vec3 rhs)
{
    return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};
}

Vec3 operator-(Vec3 lhs, Vec3 rhs)
{
    return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
}

Vec3 operator*(Vec3 value, float scalar)
{
    return {value.x * scalar, value.y * scalar, value.z * scalar};
}

Vec3 operator-(Vec3 value)
{
    return {-value.x, -value.y, -value.z};
}

float dot(Vec3 lhs, Vec3 rhs)
{
    return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

Vec3 cross(Vec3 lhs, Vec3 rhs)
{
    return {
        lhs.y * rhs.z - lhs.z * rhs.y,
        lhs.z * rhs.x - lhs.x * rhs.z,
        lhs.x * rhs.y - lhs.y * rhs.x,
    };
}

Vec3 normalize(Vec3 value)
{
    const float valueLength = std::sqrt(dot(value, value));
    if (valueLength <= 0.00001f)
    {
        return {};
    }

    return value * (1.0f / valueLength);
}

float length(Vec3 value)
{
    return std::sqrt(dot(value, value));
}

Vec3 cameraForward(float yaw, float pitch)
{
    return normalize({
        std::cos(pitch) * std::cos(yaw),
        std::sin(pitch),
        std::cos(pitch) * std::sin(yaw),
    });
}

Vec3 cameraRight(float yaw)
{
    return normalize({-std::sin(yaw), 0.0f, std::cos(yaw)});
}

Mat4 multiply(Mat4 lhs, Mat4 rhs)
{
    Mat4 result{};
    for (int column = 0; column < 4; ++column)
    {
        for (int row = 0; row < 4; ++row)
        {
            for (int i = 0; i < 4; ++i)
            {
                result.m[column * 4 + row] += lhs.m[i * 4 + row] * rhs.m[column * 4 + i];
            }
        }
    }

    return result;
}

Mat4 makeViewMatrix(Vec3 position, float yaw, float pitch)
{
    const Vec3 forward = cameraForward(yaw, pitch);
    const Vec3 right = normalize(cross(forward, {0.0f, 1.0f, 0.0f}));
    const Vec3 up = normalize(cross(right, forward));

    Mat4 view{};
    view.m[0] = right.x;
    view.m[1] = up.x;
    view.m[2] = forward.x;
    view.m[4] = right.y;
    view.m[5] = up.y;
    view.m[6] = forward.y;
    view.m[8] = right.z;
    view.m[9] = up.z;
    view.m[10] = forward.z;
    view.m[12] = -dot(right, position);
    view.m[13] = -dot(up, position);
    view.m[14] = -dot(forward, position);
    view.m[15] = 1.0f;
    return view;
}

Mat4 makeViewMatrixFromForward(Vec3 position, Vec3 forward)
{
    forward = normalize(forward);
    if (length(forward) <= 0.00001f)
    {
        forward = {1.0f, 0.0f, 0.0f};
    }

    Vec3 right = normalize(cross(forward, {0.0f, 1.0f, 0.0f}));
    if (length(right) <= 0.00001f)
    {
        right = {1.0f, 0.0f, 0.0f};
    }
    const Vec3 up = normalize(cross(right, forward));

    Mat4 view{};
    view.m[0] = right.x;
    view.m[1] = up.x;
    view.m[2] = forward.x;
    view.m[4] = right.y;
    view.m[5] = up.y;
    view.m[6] = forward.y;
    view.m[8] = right.z;
    view.m[9] = up.z;
    view.m[10] = forward.z;
    view.m[12] = -dot(right, position);
    view.m[13] = -dot(up, position);
    view.m[14] = -dot(forward, position);
    view.m[15] = 1.0f;
    return view;
}

std::pair<float, float> yawPitchFromForward(Vec3 forward)
{
    forward = normalize(forward);
    if (length(forward) <= 0.00001f)
    {
        return {0.0f, 0.0f};
    }

    return {
        std::atan2(forward.z, forward.x),
        std::asin(std::clamp(forward.y, -1.0f, 1.0f)),
    };
}

Mat4 makePerspectiveMatrix(float fovYRadians, float aspect, float nearPlane, float farPlane)
{
    const float tanHalfFov = std::tan(fovYRadians * 0.5f);
    const float depthScale = farPlane / (farPlane - nearPlane);

    Mat4 projection{};
    projection.m[0] = 1.0f / (aspect * tanHalfFov);
    projection.m[5] = 1.0f / tanHalfFov;
    projection.m[10] = depthScale;
    projection.m[11] = 1.0f;
    projection.m[14] = -nearPlane * depthScale;
    return projection;
}
