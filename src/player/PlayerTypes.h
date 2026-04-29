#pragma once

#include "core/Math.h"

constexpr float kPlayerHalfWidth = 0.3f;
constexpr float kPlayerHeight = 1.75f;
constexpr float kPlayerEyeHeight = 1.5625f;
constexpr float kFlySpeed = 64.0f;
constexpr float kWalkSpeed = 6.0f;
constexpr float kJumpSpeed = 8.5f;
constexpr float kGravity = 24.0f;
constexpr float kPhysicsDeltaSeconds = 1.0f / 60.0f;
constexpr int kMaxPhysicsStepsPerFrame = 4;
constexpr float kThirdPersonDistance = 4.5f;

enum class MovementMode
{
    Fly,
    Ground,
};

enum class CameraViewMode
{
    FirstPerson,
    ThirdPersonRear,
    ThirdPersonFront,
};

struct CameraState
{
    float yaw = 0.0f;   // +X is east, so the initial view looks east.
    float pitch = 0.0f;
    Vec3 position{80.0f, 310.0f - kPlayerEyeHeight, 80.0f}; // Player feet center.
    bool firstMouse = true;
    double lastMouseX = 0.0;
    double lastMouseY = 0.0;
};
