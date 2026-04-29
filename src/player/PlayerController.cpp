#include "player/PlayerController.h"

#include "core/Math.h"

#include <algorithm>
#include <cmath>

void PlayerController::update(
    CameraState& camera,
    const PlayerInputState& input,
    float deltaSeconds,
    const SolidBlockQuery& isSolidBlock)
{
    deltaSeconds = std::min(deltaSeconds, 0.25f);

    if (movementMode_ == MovementMode::Fly)
    {
        Vec3 movement{};
        const Vec3 forward = cameraForward(camera.yaw, 0.0f);
        const Vec3 right = cameraRight(camera.yaw);

        if (input.moveForward)
        {
            movement = movement + forward;
        }
        if (input.moveBackward)
        {
            movement = movement - forward;
        }
        if (input.moveRight)
        {
            movement = movement + right;
        }
        if (input.moveLeft)
        {
            movement = movement - right;
        }
        if (input.moveUp)
        {
            movement = movement + Vec3{0.0f, 1.0f, 0.0f};
        }
        if (input.moveDown)
        {
            movement = movement - Vec3{0.0f, 1.0f, 0.0f};
        }

        if (dot(movement, movement) > 0.0f)
        {
            camera.position = camera.position + normalize(movement) * (kFlySpeed * deltaSeconds);
        }
        return;
    }

    physicsAccumulatorSeconds_ += deltaSeconds;
    int steps = 0;
    while (physicsAccumulatorSeconds_ >= kPhysicsDeltaSeconds && steps < kMaxPhysicsStepsPerFrame)
    {
        stepGroundPhysics(camera, input, kPhysicsDeltaSeconds, isSolidBlock);
        physicsAccumulatorSeconds_ -= kPhysicsDeltaSeconds;
        ++steps;
    }
    if (steps == kMaxPhysicsStepsPerFrame)
    {
        physicsAccumulatorSeconds_ = std::min(physicsAccumulatorSeconds_, kPhysicsDeltaSeconds);
    }
}

void PlayerController::toggleMovementMode()
{
    if (movementMode_ == MovementMode::Fly)
    {
        movementMode_ = MovementMode::Ground;
    }
    else
    {
        movementMode_ = MovementMode::Fly;
    }
    verticalVelocity_ = 0.0f;
    playerGrounded_ = false;
    physicsAccumulatorSeconds_ = 0.0f;
}

void PlayerController::cycleCameraViewMode()
{
    if (cameraViewMode_ == CameraViewMode::FirstPerson)
    {
        cameraViewMode_ = CameraViewMode::ThirdPersonRear;
    }
    else if (cameraViewMode_ == CameraViewMode::ThirdPersonRear)
    {
        cameraViewMode_ = CameraViewMode::ThirdPersonFront;
    }
    else
    {
        cameraViewMode_ = CameraViewMode::FirstPerson;
    }
}

void PlayerController::setMovementMode(MovementMode movementMode)
{
    movementMode_ = movementMode;
    verticalVelocity_ = 0.0f;
    playerGrounded_ = false;
    physicsAccumulatorSeconds_ = 0.0f;
}

void PlayerController::setCameraViewMode(CameraViewMode cameraViewMode)
{
    cameraViewMode_ = cameraViewMode;
}

MovementMode PlayerController::movementMode() const
{
    return movementMode_;
}

CameraViewMode PlayerController::cameraViewMode() const
{
    return cameraViewMode_;
}

Vec3 PlayerController::playerFeetPosition(const CameraState& camera) const
{
    return {camera.position.x, camera.position.y - kPlayerEyeHeight, camera.position.z};
}

bool PlayerController::isThirdPersonView() const
{
    return cameraViewMode_ != CameraViewMode::FirstPerson;
}

Vec3 PlayerController::renderCameraForward(const CameraState& camera) const
{
    const Vec3 forward = cameraForward(camera.yaw, camera.pitch);
    return cameraViewMode_ == CameraViewMode::ThirdPersonFront ? -forward : forward;
}

Vec3 PlayerController::renderCameraPosition(const CameraState& camera) const
{
    const Vec3 forward = cameraForward(camera.yaw, camera.pitch);
    if (cameraViewMode_ == CameraViewMode::ThirdPersonRear)
    {
        return camera.position - forward * kThirdPersonDistance;
    }
    if (cameraViewMode_ == CameraViewMode::ThirdPersonFront)
    {
        return camera.position + forward * kThirdPersonDistance;
    }
    return camera.position;
}

void PlayerController::stepGroundPhysics(
    CameraState& camera,
    const PlayerInputState& input,
    float deltaSeconds,
    const SolidBlockQuery& isSolidBlock)
{
    Vec3 movement{};
    const Vec3 forward = cameraForward(camera.yaw, 0.0f);
    const Vec3 right = cameraRight(camera.yaw);

    if (input.moveForward)
    {
        movement = movement + forward;
    }
    if (input.moveBackward)
    {
        movement = movement - forward;
    }
    if (input.moveRight)
    {
        movement = movement + right;
    }
    if (input.moveLeft)
    {
        movement = movement - right;
    }
    if (dot(movement, movement) > 0.0f)
    {
        movement = normalize(movement);
    }

    if (input.moveUp && playerGrounded_)
    {
        verticalVelocity_ = kJumpSpeed;
        playerGrounded_ = false;
    }

    verticalVelocity_ -= kGravity * deltaSeconds;

    bool hitGround = false;
    Vec3 nextPosition = camera.position;
    const Vec3 horizontalDelta = movement * (kWalkSpeed * deltaSeconds);
    nextPosition = movePlayerAxis(nextPosition, {horizontalDelta.x, 0.0f, 0.0f}, hitGround, isSolidBlock);
    nextPosition = movePlayerAxis(nextPosition, {0.0f, 0.0f, horizontalDelta.z}, hitGround, isSolidBlock);
    nextPosition = movePlayerAxis(nextPosition, {0.0f, verticalVelocity_ * deltaSeconds, 0.0f}, hitGround, isSolidBlock);

    if (hitGround && verticalVelocity_ < 0.0f)
    {
        verticalVelocity_ = 0.0f;
        playerGrounded_ = true;
    }
    else
    {
        playerGrounded_ = false;
    }

    camera.position = nextPosition;
}

Vec3 PlayerController::movePlayerAxis(
    Vec3 startPosition,
    Vec3 axisDelta,
    bool& hitGround,
    const SolidBlockQuery& isSolidBlock) const
{
    const float distance = length(axisDelta);
    if (distance <= 0.00001f)
    {
        return startPosition;
    }

    Vec3 position = startPosition;
    const int stepCount = std::max(1, static_cast<int>(std::ceil(distance / 0.2f)));
    const Vec3 stepDelta = axisDelta * (1.0f / static_cast<float>(stepCount));
    for (int step = 0; step < stepCount; ++step)
    {
        const Vec3 candidate = position + stepDelta;
        if (!isPlayerCollidingAt(candidate, isSolidBlock))
        {
            position = candidate;
            continue;
        }

        if (stepDelta.y < 0.0f)
        {
            hitGround = true;
        }

        Vec3 safePosition = position;
        Vec3 blockedPosition = candidate;
        for (int i = 0; i < 8; ++i)
        {
            const Vec3 midpoint = (safePosition + blockedPosition) * 0.5f;
            if (isPlayerCollidingAt(midpoint, isSolidBlock))
            {
                blockedPosition = midpoint;
            }
            else
            {
                safePosition = midpoint;
            }
        }
        return safePosition;
    }

    return position;
}

bool PlayerController::isPlayerCollidingAt(Vec3 eyePosition, const SolidBlockQuery& isSolidBlock) const
{
    const float minX = eyePosition.x - kPlayerHalfWidth;
    const float maxX = eyePosition.x + kPlayerHalfWidth;
    const float minY = eyePosition.y - kPlayerEyeHeight;
    const float maxY = minY + kPlayerHeight;
    const float minZ = eyePosition.z - kPlayerHalfWidth;
    const float maxZ = eyePosition.z + kPlayerHalfWidth;

    const int blockMinX = static_cast<int>(std::floor(minX));
    const int blockMaxX = static_cast<int>(std::floor(maxX - 0.0001f));
    const int blockMinY = static_cast<int>(std::floor(minY));
    const int blockMaxY = static_cast<int>(std::floor(maxY - 0.0001f));
    const int blockMinZ = static_cast<int>(std::floor(minZ));
    const int blockMaxZ = static_cast<int>(std::floor(maxZ - 0.0001f));

    for (int y = blockMinY; y <= blockMaxY; ++y)
    {
        for (int z = blockMinZ; z <= blockMaxZ; ++z)
        {
            for (int x = blockMinX; x <= blockMaxX; ++x)
            {
                if (isSolidBlock(x, y, z))
                {
                    return true;
                }
            }
        }
    }
    return false;
}
