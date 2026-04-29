#pragma once

#include "player/PlayerTypes.h"

#include <functional>

struct PlayerInputState
{
    bool moveForward = false;
    bool moveBackward = false;
    bool moveRight = false;
    bool moveLeft = false;
    bool moveUp = false;
    bool moveDown = false;
};

class PlayerController
{
public:
    using SolidBlockQuery = std::function<bool(int, int, int)>;

    void update(CameraState& camera, const PlayerInputState& input, float deltaSeconds, const SolidBlockQuery& isSolidBlock);
    void toggleMovementMode();
    void cycleCameraViewMode();
    void setMovementMode(MovementMode movementMode);
    void setCameraViewMode(CameraViewMode cameraViewMode);
    MovementMode movementMode() const;
    CameraViewMode cameraViewMode() const;

    Vec3 playerFeetPosition(const CameraState& camera) const;
    bool isThirdPersonView() const;
    Vec3 renderCameraForward(const CameraState& camera) const;
    Vec3 renderCameraPosition(const CameraState& camera) const;

private:
    void stepGroundPhysics(
        CameraState& camera,
        const PlayerInputState& input,
        float deltaSeconds,
        const SolidBlockQuery& isSolidBlock);
    Vec3 movePlayerAxis(
        Vec3 startPosition,
        Vec3 axisDelta,
        bool& hitGround,
        const SolidBlockQuery& isSolidBlock) const;
    bool isPlayerCollidingAt(Vec3 eyePosition, const SolidBlockQuery& isSolidBlock) const;

    MovementMode movementMode_ = MovementMode::Fly;
    CameraViewMode cameraViewMode_ = CameraViewMode::FirstPerson;
    float verticalVelocity_ = 0.0f;
    float physicsAccumulatorSeconds_ = 0.0f;
    bool playerGrounded_ = false;
};
