#include "VulkanVoxel.h"

#define NOMINMAX
#include <Windows.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kWindowWidth = 1280;
constexpr int kWindowHeight = 720;
constexpr int kMaxFramesInFlight = 2;
constexpr std::size_t kMaxOverlayVertexCount = 4194304;
constexpr double kSelectedBlockTraceIntervalSeconds = 1.0 / 60.0;
constexpr float kMoveSpeed = 120.0f;
constexpr float kWalkSpeed = 6.0f;
constexpr float kJumpSpeed = 8.5f;
constexpr float kGravity = 24.0f;
constexpr float kPlayerHalfWidth = 0.3f;
constexpr float kPlayerHeight = 1.75f;
constexpr float kAxisStepSize = 0.2f;
constexpr double kMovementToggleTapWindowSeconds = 0.3;
constexpr int kPhysicsTicksPerSecond = 20;
constexpr float kPhysicsDeltaTime = 1.0f / static_cast<float>(kPhysicsTicksPerSecond);
constexpr int kMaxPhysicsStepsPerFrame = 8;
constexpr std::uint64_t kDayLengthTicks = 24ull * 60ull * kPhysicsTicksPerSecond;
constexpr std::uint64_t kHourLengthTicks = 60ull * kPhysicsTicksPerSecond;
constexpr std::uint64_t kMinuteLengthTicks = kPhysicsTicksPerSecond;
constexpr float kMouseSensitivity = 0.08f;
constexpr float kPi = 3.14159265358979323846f;
constexpr const char* kWindowTitle = "VulkanVoxel";
constexpr float kOverlayGlyphScale = 1.0f;
constexpr float kOverlayMargin = 16.0f;
constexpr float kOverlayLineGap = 8.0f;
constexpr float kOverlayReferenceWidth = 1920.0f;
constexpr float kOverlayReferenceHeight = 1080.0f;
constexpr float kInteractionRange = 10.0f;
constexpr float kSelectionExpand = 0.002f;
constexpr float kPlayerEyeHeight = 1.5625f;
constexpr float kThirdPersonDistance = 4.5f;

int GetChunkCoordinate(float worldCoordinate, int chunkSize) {
    return static_cast<int>(std::floor(worldCoordinate / static_cast<float>(chunkSize)));
}

bool HasOverlayGlyph(const std::array<FontGlyphBitmap, 128>& glyphs, char c) {
    const unsigned char code = static_cast<unsigned char>(c);
    return code < glyphs.size() && glyphs[code].valid;
}

std::string SanitizeOverlayText(const std::string& text, const std::array<FontGlyphBitmap, 128>& glyphs) {
    std::string result;
    result.reserve(text.size());

    for (unsigned char c : text) {
        const char upper = static_cast<char>(std::toupper(c));
        if (HasOverlayGlyph(glyphs, upper)) {
            result.push_back(upper);
        } else if (HasOverlayGlyph(glyphs, static_cast<char>(c))) {
            result.push_back(static_cast<char>(c));
        } else {
            result.push_back(' ');
        }
    }

    return result;
}

float GetOverlayTextWidth(const std::string& text, const std::array<FontGlyphBitmap, 128>& glyphs, float scale) {
    float width = 0.0f;
    for (char c : text) {
        const unsigned char code = static_cast<unsigned char>(c);
        if (code >= glyphs.size() || !glyphs[code].valid) {
            continue;
        }
        width += static_cast<float>(glyphs[code].advance) * scale;
    }
    return width;
}

float GetOverlayUiScale(const VkExtent2D& extent) {
    if (extent.width == 0 || extent.height == 0) {
        return 1.0f;
    }

    const float widthScale = static_cast<float>(extent.width) / kOverlayReferenceWidth;
    const float heightScale = static_cast<float>(extent.height) / kOverlayReferenceHeight;
    return std::clamp(std::min(widthScale, heightScale), 0.75f, 3.0f);
}

void AppendOverlayQuad(
    std::vector<OverlayVertex>& vertices,
    float leftPixels,
    float topPixels,
    float rightPixels,
    float bottomPixels,
    float uvLeft,
    float uvTop,
    float uvRight,
    float uvBottom,
    float red,
    float green,
    float blue,
    const VkExtent2D& extent
) {
    const float left = (leftPixels / static_cast<float>(extent.width)) * 2.0f - 1.0f;
    const float right = (rightPixels / static_cast<float>(extent.width)) * 2.0f - 1.0f;
    const float top = (topPixels / static_cast<float>(extent.height)) * 2.0f - 1.0f;
    const float bottom = (bottomPixels / static_cast<float>(extent.height)) * 2.0f - 1.0f;

    const OverlayVertex topLeft{{left, top}, {uvLeft, uvTop}, {red, green, blue}};
    const OverlayVertex topRight{{right, top}, {uvRight, uvTop}, {red, green, blue}};
    const OverlayVertex bottomRight{{right, bottom}, {uvRight, uvBottom}, {red, green, blue}};
    const OverlayVertex bottomLeft{{left, bottom}, {uvLeft, uvBottom}, {red, green, blue}};

    vertices.push_back(topLeft);
    vertices.push_back(topRight);
    vertices.push_back(bottomRight);
    vertices.push_back(topLeft);
    vertices.push_back(bottomRight);
    vertices.push_back(bottomLeft);
}

bool ProjectDirectionToScreen(
    const Vec3& direction,
    const Vec3& cameraForward,
    const Vec3& cameraRight,
    const Vec3& cameraUp,
    float tanHalfFovX,
    float tanHalfFovY,
    const VkExtent2D& extent,
    float& outCenterX,
    float& outCenterY
) {
    const float viewX = Dot(direction, cameraRight);
    const float viewY = Dot(direction, cameraUp);
    const float viewZ = Dot(direction, cameraForward);
    if (viewZ <= 0.0001f) {
        return false;
    }

    const float ndcX = viewX / (viewZ * tanHalfFovX);
    const float ndcY = viewY / (viewZ * tanHalfFovY);
    if (ndcX < -1.2f || ndcX > 1.2f || ndcY < -1.2f || ndcY > 1.2f) {
        return false;
    }

    outCenterX = (ndcX * 0.5f + 0.5f) * static_cast<float>(extent.width);
    outCenterY = (0.5f - ndcY * 0.5f) * static_cast<float>(extent.height);
    return true;
}

void AppendGlyph(
    std::vector<OverlayVertex>& vertices,
    const FontGlyphBitmap& glyph,
    float startX,
    float startY,
    float scale,
    float red,
    float green,
    float blue,
    const VkExtent2D& extent
) {
    if (!glyph.valid || glyph.width <= 0 || glyph.height <= 0) {
        return;
    }

    const float left = startX + static_cast<float>(glyph.offsetX) * scale;
    const float top = startY + static_cast<float>(glyph.offsetY) * scale;
    const float right = left + static_cast<float>(glyph.width) * scale;
    const float bottom = top + static_cast<float>(glyph.height) * scale;

    AppendOverlayQuad(
        vertices,
        left,
        top,
        right,
        bottom,
        glyph.u0,
        glyph.v0,
        glyph.u1,
        glyph.v1,
        red,
        green,
        blue,
        extent
    );
}

void AppendOutlinedGlyph(
    std::vector<OverlayVertex>& vertices,
    const FontGlyphBitmap& glyph,
    float startX,
    float startY,
    float scale,
    const VkExtent2D& extent
) {
    constexpr std::array<Vec2, 8> outlineOffsets = {{
        {-1.0f, -1.0f},
        {0.0f, -1.0f},
        {1.0f, -1.0f},
        {-1.0f, 0.0f},
        {1.0f, 0.0f},
        {-1.0f, 1.0f},
        {0.0f, 1.0f},
        {1.0f, 1.0f},
    }};

    for (const Vec2& offset : outlineOffsets) {
        AppendGlyph(
            vertices,
            glyph,
            startX + offset.x * scale,
            startY + offset.y * scale,
            scale,
            0.0f,
            0.0f,
            0.0f,
            extent
        );
    }

    AppendGlyph(vertices, glyph, startX, startY, scale, 1.0f, 1.0f, 1.0f, extent);
}

void AppendOutlinedText(
    std::vector<OverlayVertex>& vertices,
    const std::array<FontGlyphBitmap, 128>& glyphs,
    const std::string& text,
    float startX,
    float startY,
    float scale,
    const VkExtent2D& extent
) {
    float cursorX = startX;

    for (char c : text) {
        const unsigned char code = static_cast<unsigned char>(c);
        if (code >= glyphs.size() || !glyphs[code].valid) {
            continue;
        }

        AppendOutlinedGlyph(vertices, glyphs[code], cursorX, startY, scale, extent);
        cursorX += static_cast<float>(glyphs[code].advance) * scale;
    }
}

std::string FormatFloat(double value, int precision) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(precision) << value;
    return stream.str();
}

std::string FormatVersion(std::uint32_t version) {
    std::ostringstream stream;
    stream << VK_VERSION_MAJOR(version) << '.'
           << VK_VERSION_MINOR(version) << '.'
           << VK_VERSION_PATCH(version);
    return stream.str();
}

std::string FormatGameTime(std::uint64_t globalTick) {
    const std::uint64_t day = globalTick / kDayLengthTicks + 1;
    const std::uint64_t dayTick = globalTick % kDayLengthTicks;
    const std::uint64_t hour = dayTick / kHourLengthTicks;
    const std::uint64_t minute = (dayTick % kHourLengthTicks) / kMinuteLengthTicks;

    std::ostringstream stream;
    stream << "TIME: DAY " << day << ' '
           << std::setw(2) << std::setfill('0') << hour << ':'
           << std::setw(2) << std::setfill('0') << minute;
    return stream.str();
}

std::string FormatDriverVersion(std::uint32_t vendorId, std::uint32_t version) {
    std::ostringstream stream;

    if (vendorId == 0x10DE) {
        stream << ((version >> 22) & 0x3FF) << '.'
               << ((version >> 14) & 0x0FF) << '.'
               << ((version >> 6) & 0x0FF) << '.'
               << (version & 0x03F);
        return stream.str();
    }

    return FormatVersion(version);
}

std::string FormatGigabytes(std::uint64_t bytes) {
    constexpr double bytesPerGb = 1024.0 * 1024.0 * 1024.0;
    return FormatFloat(static_cast<double>(bytes) / bytesPerGb, 2);
}

std::string FormatUsageLine(
    const std::string& label,
    std::uint64_t usedBytes,
    std::uint64_t totalBytes
) {
    return label + " USED: " + FormatGigabytes(usedBytes) + " / " + FormatGigabytes(totalBytes) + " GB";
}

std::string FormatRuntimeProfileLine(const std::string& label, const RuntimeProfileStage& stage) {
    std::ostringstream stream;
    stream << label << ": ";
    if (stage.samples == 0) {
        stream << "0.000 / 0.000 MS";
    } else {
        stream << std::fixed << std::setprecision(3)
               << stage.averageMs << " / " << stage.maxMs << " MS";
    }
    return stream.str();
}

std::string FormatRuntimeCountLine(const std::string& label, const RuntimeProfileStage& stage) {
    std::ostringstream stream;
    stream << label << ": ";
    if (stage.samples == 0) {
        stream << "0.000 / 0.000";
    } else {
        stream << std::fixed << std::setprecision(3)
               << stage.averageMs << " / " << stage.maxMs;
    }
    return stream.str();
}

Vec3 LerpVec3(const Vec3& a, const Vec3& b, float t) {
    return {
        a.x + (b.x - a.x) * t,
        a.y + (b.y - a.y) * t,
        a.z + (b.z - a.z) * t,
    };
}

float WrapAngle180(float angleDegrees) {
    while (angleDegrees > 180.0f) {
        angleDegrees -= 360.0f;
    }
    while (angleDegrees <= -180.0f) {
        angleDegrees += 360.0f;
    }
    return angleDegrees;
}

const char* GetCardinalDirectionLabel(float yawDegrees) {
    const float wrappedYaw = WrapAngle180(yawDegrees);

    if (wrappedYaw >= -45.0f && wrappedYaw < 45.0f) {
        return "EAST";
    }
    if (wrappedYaw >= 45.0f && wrappedYaw < 135.0f) {
        return "SOUTH";
    }
    if (wrappedYaw >= -135.0f && wrappedYaw < -45.0f) {
        return "NORTH";
    }
    return "WEST";
}

}  // namespace

bool QueueFamilyIndices::IsComplete() const {
    return graphicsFamily.has_value() && presentFamily.has_value();
}

int VulkanVoxelApp::Run() {
    try {
        InitWindow();
        InitVulkan();
        MainLoop();
        Cleanup();
        return 0;
    } catch (...) {
        Cleanup();
        throw;
    }
}

bool VulkanVoxelApp::IsSolidBlockAt(int worldX, int worldY, int worldZ) {
    if (worldY < 0 || worldY >= kWorldSizeY) {
        return true;
    }

    return world_.GetBlock(worldX, worldY, worldZ) != 0;
}

bool VulkanVoxelApp::IsPlayerCollidingAt(const Vec3& eyePosition) {
    const float minX = eyePosition.x - kPlayerHalfWidth;
    const float maxX = eyePosition.x + kPlayerHalfWidth;
    const float minY = eyePosition.y - kPlayerEyeHeight;
    const float maxY = minY + kPlayerHeight;
    const float minZ = eyePosition.z - kPlayerHalfWidth;
    const float maxZ = eyePosition.z + kPlayerHalfWidth;

    if (minY < 0.0f || maxY > static_cast<float>(kWorldSizeY)) {
        return true;
    }

    const int blockMinX = static_cast<int>(std::floor(minX));
    const int blockMaxX = static_cast<int>(std::floor(maxX - 0.0001f));
    const int blockMinY = static_cast<int>(std::floor(minY));
    const int blockMaxY = static_cast<int>(std::floor(maxY - 0.0001f));
    const int blockMinZ = static_cast<int>(std::floor(minZ));
    const int blockMaxZ = static_cast<int>(std::floor(maxZ - 0.0001f));

    for (int worldY = blockMinY; worldY <= blockMaxY; ++worldY) {
        for (int worldZ = blockMinZ; worldZ <= blockMaxZ; ++worldZ) {
            for (int worldX = blockMinX; worldX <= blockMaxX; ++worldX) {
                if (IsSolidBlockAt(worldX, worldY, worldZ)) {
                    return true;
                }
            }
        }
    }

    return false;
}

Vec3 VulkanVoxelApp::MovePlayerAxis(const Vec3& startPosition, const Vec3& axisDelta, bool& hitNegativeY) {
    Vec3 position = startPosition;
    const float distance = Length(axisDelta);
    if (distance <= 0.00001f) {
        return position;
    }

    const int stepCount = std::max(1, static_cast<int>(std::ceil(distance / kAxisStepSize)));
    const Vec3 stepDelta = axisDelta * (1.0f / static_cast<float>(stepCount));

    for (int step = 0; step < stepCount; ++step) {
        Vec3 candidate = position + stepDelta;
        if (IsPlayerCollidingAt(candidate)) {
            if (stepDelta.y < 0.0f) {
                hitNegativeY = true;
            }
            Vec3 safePosition = position;
            Vec3 blockedPosition = candidate;
            for (int iteration = 0; iteration < 8; ++iteration) {
                const Vec3 midPoint = (safePosition + blockedPosition) * 0.5f;
                if (IsPlayerCollidingAt(midPoint)) {
                    blockedPosition = midPoint;
                } else {
                    safePosition = midPoint;
                }
            }
            position = safePosition;
            break;
        }
        position = candidate;
    }

    return position;
}

BlockRaycastHit VulkanVoxelApp::TraceSelectedBlock() {
    const auto profileStartTime = std::chrono::steady_clock::now();
    BlockRaycastHit hit{};

    Vec3 origin = GetRenderCameraPosition();
    Vec3 direction = GetRenderCameraForward();
    if (cameraViewMode_ == CameraViewMode::ThirdPersonFront) {
        origin = cameraPosition_;
        direction = GetForwardVector();
    }
    auto recordRaycastWalkSample = [&](double waitMs) {
        const double totalMs =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - profileStartTime).count();
        const double walkMs = std::max(0.0, totalMs - waitMs);
        AccumulateRuntimeProfileSample(
            walkMs,
            raycastProfileTotalMs_,
            raycastProfileMaxMs_,
            raycastProfileSamples_
        );
    };
    if (Length(direction) <= 0.00001f) {
        recordRaycastWalkSample(0.0);
        return hit;
    }

    int x = static_cast<int>(std::floor(origin.x));
    int y = static_cast<int>(std::floor(origin.y));
    int z = static_cast<int>(std::floor(origin.z));

    const int stepX = direction.x > 0.0f ? 1 : (direction.x < 0.0f ? -1 : 0);
    const int stepY = direction.y > 0.0f ? 1 : (direction.y < 0.0f ? -1 : 0);
    const int stepZ = direction.z > 0.0f ? 1 : (direction.z < 0.0f ? -1 : 0);

    const float nextBoundaryX = stepX > 0 ? static_cast<float>(x + 1) : static_cast<float>(x);
    const float nextBoundaryY = stepY > 0 ? static_cast<float>(y + 1) : static_cast<float>(y);
    const float nextBoundaryZ = stepZ > 0 ? static_cast<float>(z + 1) : static_cast<float>(z);

    float tMaxX = stepX != 0 ? (nextBoundaryX - origin.x) / direction.x : std::numeric_limits<float>::infinity();
    float tMaxY = stepY != 0 ? (nextBoundaryY - origin.y) / direction.y : std::numeric_limits<float>::infinity();
    float tMaxZ = stepZ != 0 ? (nextBoundaryZ - origin.z) / direction.z : std::numeric_limits<float>::infinity();
    float tDeltaX = stepX != 0 ? 1.0f / std::abs(direction.x) : std::numeric_limits<float>::infinity();
    float tDeltaY = stepY != 0 ? 1.0f / std::abs(direction.y) : std::numeric_limits<float>::infinity();
    float tDeltaZ = stepZ != 0 ? 1.0f / std::abs(direction.z) : std::numeric_limits<float>::infinity();

    int previousX = x;
    int previousY = y;
    int previousZ = z;

    const auto waitStartTime = std::chrono::steady_clock::now();
    std::shared_lock lock(worldMutex_);
    const double waitMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - waitStartTime).count();
    AccumulateRuntimeProfileSample(
        waitMs,
        raycastWaitProfileTotalMs_,
        raycastWaitProfileMaxMs_,
        raycastWaitProfileSamples_
    );

    while (true) {
        if (VoxelWorld::IsInsideWorld(x, y, z) && world_.GetBlock(x, y, z) != 0) {
            hit.hit = true;
            hit.blockX = x;
            hit.blockY = y;
            hit.blockZ = z;
            hit.placeX = previousX;
            hit.placeY = previousY;
            hit.placeZ = previousZ;
            recordRaycastWalkSample(waitMs);
            return hit;
        }

        previousX = x;
        previousY = y;
        previousZ = z;

        float distance = tMaxX;
        if (tMaxY < distance) {
            distance = tMaxY;
        }
        if (tMaxZ < distance) {
            distance = tMaxZ;
        }

        if (distance > kInteractionRange) {
            break;
        }

        if (tMaxX <= tMaxY && tMaxX <= tMaxZ) {
            x += stepX;
            tMaxX += tDeltaX;
        } else if (tMaxY <= tMaxZ) {
            y += stepY;
            tMaxY += tDeltaY;
        } else {
            z += stepZ;
            tMaxZ += tDeltaZ;
        }
    }

    recordRaycastWalkSample(waitMs);
    return hit;
}

const char* gFatalStage = "startup";

void VulkanVoxelApp::InitWindow() {
    if (glfwInit() != GLFW_TRUE) {
        throw std::runtime_error("Failed to initialize GLFW.");
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    window_ = glfwCreateWindow(kWindowWidth, kWindowHeight, kWindowTitle, nullptr, nullptr);
    if (window_ == nullptr) {
        throw std::runtime_error("Failed to create GLFW window.");
    }

    glfwSetInputMode(window_, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
}

void VulkanVoxelApp::InitVulkan() {
    gFatalStage = "LoadWorldSettings";
    worldSettings_ = LoadWorldSettings(std::string(ASSET_DIR) + "/assets/config/world.json");
    {
        gFatalStage = "UpdateStreamingTargets";
        const int centerChunkX = static_cast<int>(std::floor(cameraPosition_.x / static_cast<float>(kChunkSizeX)));
        const int centerChunkZ = static_cast<int>(std::floor(cameraPosition_.z / static_cast<float>(kChunkSizeZ)));
        std::unique_lock lock(worldMutex_);
        world_.UpdateStreamingTargets(centerChunkX, centerChunkZ, worldSettings_.chunkRadius);
    }
    gFatalStage = "CreateInstance";
    CreateInstance();
    gFatalStage = "CreateSurface";
    CreateSurface();
    gFatalStage = "PickPhysicalDevice";
    PickPhysicalDevice();
    gFatalStage = "CreateLogicalDevice";
    CreateLogicalDevice();
    gFatalStage = "CreateSwapChain";
    CreateSwapChain();
    gFatalStage = "LoadStaticDebugInfo";
    LoadStaticDebugInfo();
    gFatalStage = "RefreshSystemUsageStats";
    RefreshSystemUsageStats();
    gFatalStage = "CreateImageViews";
    CreateImageViews();
    gFatalStage = "CreateCommandPool";
    CreateCommandPool();
    gFatalStage = "CreateDescriptorSetLayout";
    CreateDescriptorSetLayout();
    gFatalStage = "CreateRenderPass";
    CreateRenderPass();
    gFatalStage = "CreateDepthResources";
    CreateDepthResources();
    gFatalStage = "CreateTextureImage";
    CreateTextureImage();
    gFatalStage = "CreateTextureImageView";
    CreateTextureImageView();
    gFatalStage = "CreateTextureSampler";
    CreateTextureSampler();
    gFatalStage = "LoadPlayerMesh";
    LoadPlayerMesh();
    gFatalStage = "CreatePlayerTextureImage";
    CreatePlayerTextureImage();
    gFatalStage = "CreatePlayerTextureImageView";
    CreatePlayerTextureImageView();
    gFatalStage = "CreatePlayerTextureSampler";
    CreatePlayerTextureSampler();
    gFatalStage = "LoadOverlayFont";
    LoadOverlayFont();
    gFatalStage = "StartWorldMeshWorker";
    StartWorldMeshWorker();
    gFatalStage = "RequestWorldMeshBuild";
    RequestWorldMeshBuild();
    gFatalStage = "CreatePlayerBuffers";
    CreatePlayerBuffers();
    gFatalStage = "CreateOverlayBuffer";
    CreateOverlayBuffer();
    gFatalStage = "CreateSelectionBuffer";
    CreateSelectionBuffer();
    gFatalStage = "CreateEntityColliderBuffer";
    CreateEntityColliderBuffer();
    gFatalStage = "CreateUniformBuffers";
    CreateUniformBuffers();
    gFatalStage = "CreateDescriptorPool";
    CreateDescriptorPool();
    gFatalStage = "CreateDescriptorSets";
    CreateDescriptorSets();
    gFatalStage = "CreatePipelines";
    CreatePipelines();
    gFatalStage = "CreateFramebuffers";
    CreateFramebuffers();
    gFatalStage = "CreateCommandBuffers";
    CreateCommandBuffers();
    gFatalStage = "CreateSyncObjects";
    CreateSyncObjects();
    gFatalStage = "InitVulkanDone";
}

void VulkanVoxelApp::MainLoop() {
    auto lastFrameTime = std::chrono::steady_clock::now();

    while (glfwWindowShouldClose(window_) == GLFW_FALSE) {
        const auto now = std::chrono::steady_clock::now();
        const float deltaTime = std::chrono::duration<float>(now - lastFrameTime).count();
        lastFrameTime = now;

        glfwPollEvents();
        ProcessInput(deltaTime);
        ConsumeCompletedWorldMesh();
        UpdateOverlayText(deltaTime);
        DrawFrame();
    }

    if (device_ != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device_);
    }
}

void VulkanVoxelApp::Cleanup() {
    StopWorldMeshWorker();

    std::vector<RegionSaveTask> remainingSaveTasks = world_.DrainPendingSaveTasks(std::numeric_limits<std::size_t>::max());
    for (const RegionSaveTask& saveTask : remainingSaveTasks) {
        world_.ExecuteSaveTask(saveTask);
        world_.CompletePendingSaveTask(saveTask);
    }

    try {
        std::unique_lock lock(worldMutex_);
        world_.Save();
    } catch (const std::exception& e) {
        OutputDebugStringA(e.what());
        OutputDebugStringA("\n");
    }

    if (device_ != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device_);

        for (VkFence fence : inFlightFences_) {
            vkDestroyFence(device_, fence, nullptr);
        }
        for (VkSemaphore semaphore : renderFinishedSemaphores_) {
            vkDestroySemaphore(device_, semaphore, nullptr);
        }
        for (VkSemaphore semaphore : imageAvailableSemaphores_) {
            vkDestroySemaphore(device_, semaphore, nullptr);
        }

        CleanupSwapChain();

        if (commandPool_ != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device_, commandPool_, nullptr);
            commandPool_ = VK_NULL_HANDLE;
        }

        if (descriptorPool_ != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(device_, descriptorPool_, nullptr);
        }
        if (descriptorSetLayout_ != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(device_, descriptorSetLayout_, nullptr);
        }

        for (std::size_t i = 0; i < uniformBuffers_.size(); ++i) {
            if (uniformBuffers_[i] != VK_NULL_HANDLE) {
                vkDestroyBuffer(device_, uniformBuffers_[i], nullptr);
            }
            if (uniformBuffersMemory_[i] != VK_NULL_HANDLE) {
                vkFreeMemory(device_, uniformBuffersMemory_[i], nullptr);
            }
        }

        if (overlayVertexBuffer_ != VK_NULL_HANDLE) {
            vkDestroyBuffer(device_, overlayVertexBuffer_, nullptr);
        }
        if (overlayVertexBufferMemory_ != VK_NULL_HANDLE) {
            vkFreeMemory(device_, overlayVertexBufferMemory_, nullptr);
        }
        for (std::size_t i = 0; i < celestialVertexBuffers_.size(); ++i) {
            if (celestialVertexBuffers_[i] != VK_NULL_HANDLE) {
                vkDestroyBuffer(device_, celestialVertexBuffers_[i], nullptr);
            }
            if (celestialVertexBuffersMemory_[i] != VK_NULL_HANDLE) {
                vkFreeMemory(device_, celestialVertexBuffersMemory_[i], nullptr);
            }
        }
        if (selectionVertexBuffer_ != VK_NULL_HANDLE) {
            vkDestroyBuffer(device_, selectionVertexBuffer_, nullptr);
        }
        if (selectionVertexBufferMemory_ != VK_NULL_HANDLE) {
            vkFreeMemory(device_, selectionVertexBufferMemory_, nullptr);
        }
        if (entityColliderVertexBuffer_ != VK_NULL_HANDLE) {
            vkDestroyBuffer(device_, entityColliderVertexBuffer_, nullptr);
        }
        if (entityColliderVertexBufferMemory_ != VK_NULL_HANDLE) {
            vkFreeMemory(device_, entityColliderVertexBufferMemory_, nullptr);
        }
        DestroyWorldMeshBuffers();
        DestroyPlayerBuffers();

        if (textureSampler_ != VK_NULL_HANDLE) {
            vkDestroySampler(device_, textureSampler_, nullptr);
        }
        if (playerTextureSampler_ != VK_NULL_HANDLE) {
            vkDestroySampler(device_, playerTextureSampler_, nullptr);
        }
        if (overlayFontSampler_ != VK_NULL_HANDLE) {
            vkDestroySampler(device_, overlayFontSampler_, nullptr);
        }
        if (sunSampler_ != VK_NULL_HANDLE) {
            vkDestroySampler(device_, sunSampler_, nullptr);
        }
        if (moonSampler_ != VK_NULL_HANDLE) {
            vkDestroySampler(device_, moonSampler_, nullptr);
        }
        if (overlayFontImageView_ != VK_NULL_HANDLE) {
            vkDestroyImageView(device_, overlayFontImageView_, nullptr);
        }
        if (sunImageView_ != VK_NULL_HANDLE) {
            vkDestroyImageView(device_, sunImageView_, nullptr);
        }
        if (moonImageView_ != VK_NULL_HANDLE) {
            vkDestroyImageView(device_, moonImageView_, nullptr);
        }
        if (overlayFontImage_ != VK_NULL_HANDLE) {
            vkDestroyImage(device_, overlayFontImage_, nullptr);
        }
        if (sunImage_ != VK_NULL_HANDLE) {
            vkDestroyImage(device_, sunImage_, nullptr);
        }
        if (moonImage_ != VK_NULL_HANDLE) {
            vkDestroyImage(device_, moonImage_, nullptr);
        }
        if (overlayFontImageMemory_ != VK_NULL_HANDLE) {
            vkFreeMemory(device_, overlayFontImageMemory_, nullptr);
        }
        if (sunImageMemory_ != VK_NULL_HANDLE) {
            vkFreeMemory(device_, sunImageMemory_, nullptr);
        }
        if (moonImageMemory_ != VK_NULL_HANDLE) {
            vkFreeMemory(device_, moonImageMemory_, nullptr);
        }
        if (screenshotStagingBuffer_ != VK_NULL_HANDLE) {
            vkDestroyBuffer(device_, screenshotStagingBuffer_, nullptr);
        }
        if (screenshotStagingBufferMemory_ != VK_NULL_HANDLE) {
            vkFreeMemory(device_, screenshotStagingBufferMemory_, nullptr);
        }
        if (textureImageView_ != VK_NULL_HANDLE) {
            vkDestroyImageView(device_, textureImageView_, nullptr);
        }
        if (textureImage_ != VK_NULL_HANDLE) {
            vkDestroyImage(device_, textureImage_, nullptr);
        }
        if (textureImageMemory_ != VK_NULL_HANDLE) {
            vkFreeMemory(device_, textureImageMemory_, nullptr);
        }
        if (playerTextureImageView_ != VK_NULL_HANDLE) {
            vkDestroyImageView(device_, playerTextureImageView_, nullptr);
        }
        if (playerTextureImage_ != VK_NULL_HANDLE) {
            vkDestroyImage(device_, playerTextureImage_, nullptr);
        }
        if (playerTextureImageMemory_ != VK_NULL_HANDLE) {
            vkFreeMemory(device_, playerTextureImageMemory_, nullptr);
        }

        vkDestroyDevice(device_, nullptr);
        device_ = VK_NULL_HANDLE;
    }

    if (surface_ != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(instance_, surface_, nullptr);
    }
    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
    }
    if (window_ != nullptr) {
        glfwDestroyWindow(window_);
        window_ = nullptr;
    }

    glfwTerminate();
}

void VulkanVoxelApp::CleanupSwapChain() {
    if (commandPool_ != VK_NULL_HANDLE && !commandBuffers_.empty()) {
        vkFreeCommandBuffers(
            device_,
            commandPool_,
            static_cast<std::uint32_t>(commandBuffers_.size()),
            commandBuffers_.data()
        );
        commandBuffers_.clear();
    }

    for (VkFramebuffer framebuffer : swapChainFramebuffers_) {
        vkDestroyFramebuffer(device_, framebuffer, nullptr);
    }
    swapChainFramebuffers_.clear();

    if (skyPipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_, skyPipeline_, nullptr);
        skyPipeline_ = VK_NULL_HANDLE;
    }
    if (skyPipelineLayout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device_, skyPipelineLayout_, nullptr);
        skyPipelineLayout_ = VK_NULL_HANDLE;
    }
    if (overlayPipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_, overlayPipeline_, nullptr);
        overlayPipeline_ = VK_NULL_HANDLE;
    }
    if (overlayPipelineLayout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device_, overlayPipelineLayout_, nullptr);
        overlayPipelineLayout_ = VK_NULL_HANDLE;
    }
    if (selectionPipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_, selectionPipeline_, nullptr);
        selectionPipeline_ = VK_NULL_HANDLE;
    }
    if (selectionPipelineLayout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device_, selectionPipelineLayout_, nullptr);
        selectionPipelineLayout_ = VK_NULL_HANDLE;
    }
    if (terrainPipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_, terrainPipeline_, nullptr);
        terrainPipeline_ = VK_NULL_HANDLE;
    }
    if (terrainPipelineLayout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device_, terrainPipelineLayout_, nullptr);
        terrainPipelineLayout_ = VK_NULL_HANDLE;
    }
    if (worldPipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_, worldPipeline_, nullptr);
        worldPipeline_ = VK_NULL_HANDLE;
    }
    if (worldPipelineLayout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device_, worldPipelineLayout_, nullptr);
        worldPipelineLayout_ = VK_NULL_HANDLE;
    }
    if (renderPass_ != VK_NULL_HANDLE) {
        vkDestroyRenderPass(device_, renderPass_, nullptr);
        renderPass_ = VK_NULL_HANDLE;
    }

    if (depthImageView_ != VK_NULL_HANDLE) {
        vkDestroyImageView(device_, depthImageView_, nullptr);
        depthImageView_ = VK_NULL_HANDLE;
    }
    if (depthImage_ != VK_NULL_HANDLE) {
        vkDestroyImage(device_, depthImage_, nullptr);
        depthImage_ = VK_NULL_HANDLE;
    }
    if (depthImageMemory_ != VK_NULL_HANDLE) {
        vkFreeMemory(device_, depthImageMemory_, nullptr);
        depthImageMemory_ = VK_NULL_HANDLE;
    }

    for (VkImageView imageView : swapChainImageViews_) {
        vkDestroyImageView(device_, imageView, nullptr);
    }
    swapChainImageViews_.clear();

    if (swapChain_ != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(device_, swapChain_, nullptr);
        swapChain_ = VK_NULL_HANDLE;
    }

    swapChainImages_.clear();
    imagesInFlight_.clear();
}

void VulkanVoxelApp::RecreateSwapChain() {
    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window_, &width, &height);
    while (width == 0 || height == 0) {
        glfwGetFramebufferSize(window_, &width, &height);
        glfwWaitEvents();
    }

    vkDeviceWaitIdle(device_);

    CleanupSwapChain();
    CreateSwapChain();
    CreateImageViews();
    CreateRenderPass();
    CreateDepthResources();
    CreatePipelines();
    CreateFramebuffers();
    CreateCommandBuffers();

    imagesInFlight_.assign(swapChainImages_.size(), VK_NULL_HANDLE);

    RebuildOverlayVertices();
    overlayDirty_ = false;
    UploadOverlayVertices();
    UpdatePlayerRenderMesh();

    lastMeshChunkX_ = -1;
    lastMeshChunkZ_ = -1;
    lastMeshYawBucket_ = -1;
    lastMeshPitchBucket_ = -1;
    RequestWorldMeshBuild();
}

void VulkanVoxelApp::ToggleFullscreen() {
    GLFWmonitor* primaryMonitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* videoMode = primaryMonitor != nullptr ? glfwGetVideoMode(primaryMonitor) : nullptr;
    if (primaryMonitor == nullptr || videoMode == nullptr) {
        return;
    }

    if (!fullscreenEnabled_) {
        glfwGetWindowPos(window_, &windowedPosX_, &windowedPosY_);
        glfwGetWindowSize(window_, &windowedWidth_, &windowedHeight_);
        glfwSetWindowMonitor(
            window_,
            primaryMonitor,
            0,
            0,
            videoMode->width,
            videoMode->height,
            videoMode->refreshRate
        );
        fullscreenEnabled_ = true;
    } else {
        glfwSetWindowMonitor(
            window_,
            nullptr,
            windowedPosX_,
            windowedPosY_,
            windowedWidth_,
            windowedHeight_,
            0
        );
        fullscreenEnabled_ = false;
    }

    firstMouseSample_ = true;
    RecreateSwapChain();
}

void VulkanVoxelApp::ProcessInput(float deltaTime) {
    if (glfwGetKey(window_, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window_, GLFW_TRUE);
    }

    const bool isF3Pressed = glfwGetKey(window_, GLFW_KEY_F3) == GLFW_PRESS;
    const bool isCPressed = glfwGetKey(window_, GLFW_KEY_C) == GLFW_PRESS;
    const bool isF2Pressed = glfwGetKey(window_, GLFW_KEY_F2) == GLFW_PRESS;
    if (!isF3Pressed && previousF3Pressed_ && !f3CommandExecutedWhileHeld_) {
        debugOverlayEnabled_ = !debugOverlayEnabled_;
        RebuildOverlayVertices();
        overlayDirty_ = true;
    }

    if (isF3Pressed && isCPressed && !previousCPressed_) {
        entityColliderVisible_ = !entityColliderVisible_;
        f3CommandExecutedWhileHeld_ = true;
        UpdateEntityColliderBuffer();
    }

    if (!isF3Pressed) {
        f3CommandExecutedWhileHeld_ = false;
    }

    previousF3Pressed_ = isF3Pressed;
    previousCPressed_ = isCPressed;

    if (isF2Pressed && !previousF2Pressed_) {
        screenshotRequested_ = true;
    }
    previousF2Pressed_ = isF2Pressed;

    const bool isF5Pressed = glfwGetKey(window_, GLFW_KEY_F5) == GLFW_PRESS;
    if (isF5Pressed && !previousF5Pressed_) {
        if (cameraViewMode_ == CameraViewMode::FirstPerson) {
            cameraViewMode_ = CameraViewMode::ThirdPersonFront;
        } else if (cameraViewMode_ == CameraViewMode::ThirdPersonFront) {
            cameraViewMode_ = CameraViewMode::ThirdPersonRear;
        } else {
            cameraViewMode_ = CameraViewMode::FirstPerson;
        }
        RebuildOverlayVertices();
        overlayDirty_ = true;
    }
    previousF5Pressed_ = isF5Pressed;

    const bool isF11Pressed = glfwGetKey(window_, GLFW_KEY_F11) == GLFW_PRESS;
    if (isF11Pressed && !previousF11Pressed_) {
        ToggleFullscreen();
    }
    previousF11Pressed_ = isF11Pressed;

    double cursorX = 0.0;
    double cursorY = 0.0;
    glfwGetCursorPos(window_, &cursorX, &cursorY);

    if (firstMouseSample_) {
        lastMouseX_ = cursorX;
        lastMouseY_ = cursorY;
        firstMouseSample_ = false;
    }

    const double deltaX = cursorX - lastMouseX_;
    const double deltaY = cursorY - lastMouseY_;
    lastMouseX_ = cursorX;
    lastMouseY_ = cursorY;

    cameraYaw_ += static_cast<float>(deltaX) * kMouseSensitivity;
    cameraPitch_ -= static_cast<float>(deltaY) * kMouseSensitivity;
    cameraYaw_ = WrapAngle180(cameraYaw_);
    cameraPitch_ = std::clamp(cameraPitch_, -89.0f, 89.0f);

    const bool isSpacePressed = glfwGetKey(window_, GLFW_KEY_SPACE) == GLFW_PRESS;
    const bool isSpaceJustPressed = isSpacePressed && !previousSpacePressed_;
    const bool isCtrlPressed = glfwGetKey(window_, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS ||
                               glfwGetKey(window_, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS;
    const bool isWPressed = glfwGetKey(window_, GLFW_KEY_W) == GLFW_PRESS;
    const bool isSPressed = glfwGetKey(window_, GLFW_KEY_S) == GLFW_PRESS;
    const bool isDPressed = glfwGetKey(window_, GLFW_KEY_D) == GLFW_PRESS;
    const bool isAPressed = glfwGetKey(window_, GLFW_KEY_A) == GLFW_PRESS;

    moveForwardHeld_ = isWPressed;
    moveBackwardHeld_ = isSPressed;
    moveRightHeld_ = isDPressed;
    moveLeftHeld_ = isAPressed;
    moveUpHeld_ = isSpacePressed;
    moveDownHeld_ = isCtrlPressed;

    if (isSpaceJustPressed) {
        const double nowSeconds = glfwGetTime();
        if (nowSeconds - lastSpaceTapTime_ <= kMovementToggleTapWindowSeconds) {
            movementMode_ = movementMode_ == MovementMode::Fly ? MovementMode::Walk : MovementMode::Fly;
            lastSpaceTapTime_ = -1000.0;
            verticalVelocity_ = 0.0f;
            playerGrounded_ = false;
            jumpSuppressedUntilSpaceRelease_ = true;
            previousPhysicsCameraPosition_ = cameraPosition_;
            physicsAccumulatorSeconds_ = 0.0;
            RebuildOverlayVertices();
            overlayDirty_ = true;
        } else {
            lastSpaceTapTime_ = nowSeconds;
        }
    }

    if (movementMode_ == MovementMode::Fly) {
        Vec3 movement{};
        if (moveForwardHeld_) {
            movement = movement + GetHorizontalForwardVector();
        }
        if (moveBackwardHeld_) {
            movement = movement - GetHorizontalForwardVector();
        }
        if (moveRightHeld_) {
            movement = movement + GetRightVector();
        }
        if (moveLeftHeld_) {
            movement = movement - GetRightVector();
        }
        if (isSpacePressed) {
            movement.y += 1.0f;
        }
        if (isCtrlPressed) {
            movement.y -= 1.0f;
        }

        if (Length(movement) > 0.0f) {
            movement = Normalize(movement);
            previousPhysicsCameraPosition_ = cameraPosition_;
            cameraPosition_ = cameraPosition_ + movement * (kMoveSpeed * deltaTime);
        }

        cameraPosition_.y = std::clamp(cameraPosition_.y, 1.0f, static_cast<float>(kWorldSizeY - 1));
        previousPhysicsCameraPosition_ = cameraPosition_;
    }

    physicsAccumulatorSeconds_ += std::min(deltaTime, 0.25f);

    int physicsSteps = 0;
    while (physicsAccumulatorSeconds_ >= kPhysicsDeltaTime && physicsSteps < kMaxPhysicsStepsPerFrame) {
        StepFixedPhysics();
        physicsAccumulatorSeconds_ -= kPhysicsDeltaTime;
        ++physicsSteps;
    }

    if (physicsSteps == kMaxPhysicsStepsPerFrame && physicsAccumulatorSeconds_ >= kPhysicsDeltaTime) {
        physicsAccumulatorSeconds_ = 0.0;
    }

    if (!isSpacePressed) {
        jumpSuppressedUntilSpaceRelease_ = false;
    }

    previousSpacePressed_ = isSpacePressed;

    selectedBlockTraceAccumulatorSeconds_ += deltaTime;
    if (selectedBlockTraceAccumulatorSeconds_ >= kSelectedBlockTraceIntervalSeconds) {
        selectedBlockTraceAccumulatorSeconds_ =
            std::fmod(selectedBlockTraceAccumulatorSeconds_, kSelectedBlockTraceIntervalSeconds);

        const BlockRaycastHit raycastHit = TraceSelectedBlock();
        if (raycastHit.hit) {
            selectedBlockHit_ = raycastHit;
        } else {
            selectedBlockHit_.reset();
        }
        UpdateSelectionBuffer();
    }
    UpdateEntityColliderBuffer();

    const bool isLeftMousePressed = glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
    const bool isRightMousePressed = glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;

    if (isLeftMousePressed && !previousLeftMousePressed_ && selectedBlockHit_.has_value()) {
        bool changed = false;
        {
            std::unique_lock lock(worldMutex_);
            changed = world_.SetBlock(
                selectedBlockHit_->blockX,
                selectedBlockHit_->blockY,
                selectedBlockHit_->blockZ,
                0
            );
        }
        if (changed) {
            RequestWorldMeshBuild();
        }
    }

    if (isRightMousePressed && !previousRightMousePressed_ && selectedBlockHit_.has_value()) {
        const BlockRaycastHit& hit = *selectedBlockHit_;
        bool changed = false;
        {
            std::unique_lock lock(worldMutex_);
            if (VoxelWorld::IsInsideWorld(hit.placeX, hit.placeY, hit.placeZ) &&
                world_.GetBlock(hit.placeX, hit.placeY, hit.placeZ) == 0) {
                changed = world_.SetBlock(hit.placeX, hit.placeY, hit.placeZ, 1);
            }
        }
        if (changed) {
            RequestWorldMeshBuild();
        }
    }

    previousLeftMousePressed_ = isLeftMousePressed;
    previousRightMousePressed_ = isRightMousePressed;

    UpdateWorldMeshIfNeeded();
}

void VulkanVoxelApp::StepFixedPhysics() {
    ++globalTick_;

    if (movementMode_ != MovementMode::Walk) {
        return;
    }

    previousPhysicsCameraPosition_ = cameraPosition_;

    Vec3 movement{};
    if (moveForwardHeld_) {
        movement = movement + GetHorizontalForwardVector();
    }
    if (moveBackwardHeld_) {
        movement = movement - GetHorizontalForwardVector();
    }
    if (moveRightHeld_) {
        movement = movement + GetRightVector();
    }
    if (moveLeftHeld_) {
        movement = movement - GetRightVector();
    }

    if (Length(movement) > 0.0f) {
        movement = Normalize(movement);
    }

    if (moveUpHeld_ && playerGrounded_ && !jumpSuppressedUntilSpaceRelease_) {
        verticalVelocity_ = kJumpSpeed;
        playerGrounded_ = false;
    }

    verticalVelocity_ -= kGravity * kPhysicsDeltaTime;

    Vec3 nextPosition = cameraPosition_;
    const Vec3 horizontalDelta = movement * (kWalkSpeed * kPhysicsDeltaTime);
    bool hitGround = false;

    {
        const auto collisionWaitStartTime = std::chrono::steady_clock::now();
        std::shared_lock lock(worldMutex_);
        const auto collisionWorkStartTime = std::chrono::steady_clock::now();
        AccumulateRuntimeProfileSample(
            std::chrono::duration<double, std::milli>(collisionWorkStartTime - collisionWaitStartTime).count(),
            collisionWaitProfileTotalMs_,
            collisionWaitProfileMaxMs_,
            collisionWaitProfileSamples_
        );
        nextPosition = MovePlayerAxis(nextPosition, {horizontalDelta.x, 0.0f, 0.0f}, hitGround);
        nextPosition = MovePlayerAxis(nextPosition, {0.0f, 0.0f, horizontalDelta.z}, hitGround);
        nextPosition = MovePlayerAxis(nextPosition, {0.0f, verticalVelocity_ * kPhysicsDeltaTime, 0.0f}, hitGround);
        AccumulateRuntimeProfileSample(
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - collisionWorkStartTime).count(),
            collisionProfileTotalMs_,
            collisionProfileMaxMs_,
            collisionProfileSamples_
        );
    }

    if (hitGround && verticalVelocity_ < 0.0f) {
        verticalVelocity_ = 0.0f;
        playerGrounded_ = true;
    } else {
        playerGrounded_ = false;
    }

    cameraPosition_ = nextPosition;
}

float VulkanVoxelApp::GetPhysicsInterpolationAlpha() const {
    if (movementMode_ != MovementMode::Walk) {
        return 1.0f;
    }

    const float alpha = static_cast<float>(physicsAccumulatorSeconds_ / kPhysicsDeltaTime);
    return std::clamp(alpha, 0.0f, 1.0f);
}

Vec3 VulkanVoxelApp::GetInterpolatedCameraPosition() const {
    if (movementMode_ != MovementMode::Walk) {
        return cameraPosition_;
    }

    return LerpVec3(previousPhysicsCameraPosition_, cameraPosition_, GetPhysicsInterpolationAlpha());
}

Vec3 VulkanVoxelApp::GetInterpolatedPlayerFeetPosition() const {
    const Vec3 eyePosition = GetInterpolatedCameraPosition();
    return {
        eyePosition.x,
        eyePosition.y - kPlayerEyeHeight,
        eyePosition.z,
    };
}

Vec3 VulkanVoxelApp::GetForwardVector() const {
    const float yawRadians = cameraYaw_ * kPi / 180.0f;
    const float pitchRadians = cameraPitch_ * kPi / 180.0f;

    return Normalize({
        std::cos(pitchRadians) * std::cos(yawRadians),
        std::sin(pitchRadians),
        std::cos(pitchRadians) * std::sin(yawRadians),
    });
}

Vec3 VulkanVoxelApp::GetHorizontalForwardVector() const {
    Vec3 forward = GetForwardVector();
    forward.y = 0.0f;

    if (Length(forward) <= 0.00001f) {
        return {0.0f, 0.0f, -1.0f};
    }

    return Normalize(forward);
}

Vec3 VulkanVoxelApp::GetRightVector() const {
    return Normalize(Cross(GetHorizontalForwardVector(), {0.0f, 1.0f, 0.0f}));
}

Vec3 VulkanVoxelApp::GetPlayerFeetPosition() const {
    return {
        cameraPosition_.x,
        cameraPosition_.y - kPlayerEyeHeight,
        cameraPosition_.z,
    };
}

bool VulkanVoxelApp::IsThirdPersonView() const {
    return cameraViewMode_ != CameraViewMode::FirstPerson;
}

Vec3 VulkanVoxelApp::GetRenderCameraPosition() {
    const Vec3 baseCameraPosition = GetInterpolatedCameraPosition();

    if (cameraViewMode_ == CameraViewMode::FirstPerson) {
        return baseCameraPosition;
    }

    const Vec3 forward = GetForwardVector();
    const Vec3 target = baseCameraPosition;
    const float direction = cameraViewMode_ == CameraViewMode::ThirdPersonRear ? -1.0f : 1.0f;
    const Vec3 desiredCamera = target + forward * (kThirdPersonDistance * direction);
    const Vec3 cameraDelta = desiredCamera - baseCameraPosition;
    const float cameraDistance = Length(cameraDelta);
    if (cameraDistance <= 0.00001f) {
        return baseCameraPosition;
    }

    const int stepCount = std::max(1, static_cast<int>(std::ceil(cameraDistance / 0.1f)));
    Vec3 lastFreePosition = baseCameraPosition;

    std::shared_lock lock(worldMutex_);
    for (int step = 1; step <= stepCount; ++step) {
        const float t = static_cast<float>(step) / static_cast<float>(stepCount);
        const Vec3 sample = baseCameraPosition + cameraDelta * t;
        const int blockX = static_cast<int>(std::floor(sample.x));
        const int blockY = static_cast<int>(std::floor(sample.y));
        const int blockZ = static_cast<int>(std::floor(sample.z));

        if (IsSolidBlockAt(blockX, blockY, blockZ)) {
            return lastFreePosition;
        }

        lastFreePosition = sample;
    }

    return lastFreePosition;
}

Vec3 VulkanVoxelApp::GetRenderCameraForward() {
    const Vec3 forward = GetForwardVector();

    if (cameraViewMode_ == CameraViewMode::ThirdPersonFront) {
        return forward * -1.0f;
    }

    return forward;
}

void VulkanVoxelApp::UpdateWorldMeshIfNeeded() {
    const int chunkX = GetChunkCoordinate(cameraPosition_.x, kChunkSizeX);
    const int chunkZ = GetChunkCoordinate(cameraPosition_.z, kChunkSizeZ);

    if (chunkX == lastMeshChunkX_ &&
        chunkZ == lastMeshChunkZ_) {
        return;
    }

    lastMeshChunkX_ = chunkX;
    lastMeshChunkZ_ = chunkZ;
    RequestWorldMeshBuild();
}

void VulkanVoxelApp::UpdateOverlayText(float deltaTime) {
    runtimeProfileElapsedSeconds_ += deltaTime;
    if (!runtimeProfileCollectionEnabled_.load(std::memory_order_relaxed) && runtimeProfileElapsedSeconds_ >= 5.0) {
        ConsumeVoxelWorldRuntimeProfileSnapshot();
        runtimeProfileCollectionEnabled_.store(true, std::memory_order_relaxed);
    }

    ++frameCounter_;
    fpsAccumulatorSeconds_ += deltaTime;
    overlayRefreshAccumulatorSeconds_ += deltaTime;
    bool statsUpdated = false;

    if (fpsAccumulatorSeconds_ >= 0.1) {
        currentFps_ = static_cast<std::uint32_t>(
            static_cast<double>(frameCounter_) / fpsAccumulatorSeconds_ + 0.5
        );
        currentFrameTimeMs_ = currentFps_ > 0 ? 1000.0 / static_cast<double>(currentFps_) : 0.0;
        frameCounter_ = 0;
        fpsAccumulatorSeconds_ = 0.0;
        RefreshSystemUsageStats();
        RefreshRuntimeProfileStats();
        statsUpdated = true;
    }

    if (!statsUpdated && overlayRefreshAccumulatorSeconds_ < 0.1) {
        return;
    }

    overlayRefreshAccumulatorSeconds_ = 0.0;
    RebuildOverlayVertices();
    overlayDirty_ = true;
}

void VulkanVoxelApp::RefreshSystemUsageStats() {
    MEMORYSTATUSEX memoryStatus{};
    memoryStatus.dwLength = sizeof(memoryStatus);
    if (GlobalMemoryStatusEx(&memoryStatus) != FALSE) {
        totalRamBytes_ = memoryStatus.ullTotalPhys;
        usedRamBytes_ = memoryStatus.ullTotalPhys - memoryStatus.ullAvailPhys;
    }

    usedVramBytes_ = QueryVideoMemoryUsageBytes();
}

void VulkanVoxelApp::AccumulateRuntimeProfileSample(
    double elapsedMs,
    double& totalMs,
    double& maxMs,
    std::uint32_t& count
) {
    if (!runtimeProfileCollectionEnabled_.load(std::memory_order_relaxed)) {
        return;
    }
    std::lock_guard lock(runtimeProfileMutex_);
    if (!runtimeProfileCollectionEnabled_.load(std::memory_order_relaxed)) {
        return;
    }
    totalMs += elapsedMs;
    maxMs = std::max(maxMs, elapsedMs);
    ++count;
}

void VulkanVoxelApp::RefreshRuntimeProfileStats() {
    const VoxelWorldRuntimeProfileSnapshot worldProfile = ConsumeVoxelWorldRuntimeProfileSnapshot();
    if (!runtimeProfileCollectionEnabled_.load(std::memory_order_relaxed)) {
        return;
    }
    std::lock_guard lock(runtimeProfileMutex_);
    if (worldProfile.chunkLoad.samples > 0) {
        chunkLoadProfileCumulativeMs_ += worldProfile.chunkLoad.averageMs * static_cast<double>(worldProfile.chunkLoad.samples);
        chunkLoadProfileCumulativeSamples_ += worldProfile.chunkLoad.samples;
        chunkLoadProfile_.samples = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(chunkLoadProfileCumulativeSamples_, static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()))
        );
        chunkLoadProfile_.averageMs = chunkLoadProfileCumulativeMs_ / static_cast<double>(chunkLoadProfileCumulativeSamples_);
        chunkLoadProfile_.maxMs = std::max(chunkLoadProfile_.maxMs, worldProfile.chunkLoad.maxMs);
    }
    if (worldProfile.diskLoad.samples > 0) {
        diskLoadProfileCumulativeMs_ += worldProfile.diskLoad.averageMs * static_cast<double>(worldProfile.diskLoad.samples);
        diskLoadProfileCumulativeSamples_ += worldProfile.diskLoad.samples;
        diskLoadProfile_.samples = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(diskLoadProfileCumulativeSamples_, static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()))
        );
        diskLoadProfile_.averageMs = diskLoadProfileCumulativeMs_ / static_cast<double>(diskLoadProfileCumulativeSamples_);
        diskLoadProfile_.maxMs = std::max(diskLoadProfile_.maxMs, worldProfile.diskLoad.maxMs);
    }
    if (worldProfile.generate.samples > 0) {
        generateProfileCumulativeMs_ += worldProfile.generate.averageMs * static_cast<double>(worldProfile.generate.samples);
        generateProfileCumulativeSamples_ += worldProfile.generate.samples;
        generateProfile_.samples = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(generateProfileCumulativeSamples_, static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()))
        );
        generateProfile_.averageMs = generateProfileCumulativeMs_ / static_cast<double>(generateProfileCumulativeSamples_);
        generateProfile_.maxMs = std::max(generateProfile_.maxMs, worldProfile.generate.maxMs);
    }
    if (worldProfile.meshBuild.samples > 0) {
        meshBuildProfileCumulativeMs_ += worldProfile.meshBuild.averageMs * static_cast<double>(worldProfile.meshBuild.samples);
        meshBuildProfileCumulativeSamples_ += worldProfile.meshBuild.samples;
        meshBuildProfile_.samples = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(meshBuildProfileCumulativeSamples_, static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()))
        );
        meshBuildProfile_.averageMs = meshBuildProfileCumulativeMs_ / static_cast<double>(meshBuildProfileCumulativeSamples_);
        meshBuildProfile_.maxMs = std::max(meshBuildProfile_.maxMs, worldProfile.meshBuild.maxMs);
    }
    if (worldProfile.save.samples > 0) {
        saveProfileCumulativeMs_ += worldProfile.save.averageMs * static_cast<double>(worldProfile.save.samples);
        saveProfileCumulativeSamples_ += worldProfile.save.samples;
        saveProfile_.samples = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(saveProfileCumulativeSamples_, static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()))
        );
        saveProfile_.averageMs = saveProfileCumulativeMs_ / static_cast<double>(saveProfileCumulativeSamples_);
        saveProfile_.maxMs = std::max(saveProfile_.maxMs, worldProfile.save.maxMs);
    }
    if (worldProfile.unload.samples > 0) {
        unloadProfileCumulativeMs_ += worldProfile.unload.averageMs * static_cast<double>(worldProfile.unload.samples);
        unloadProfileCumulativeSamples_ += worldProfile.unload.samples;
        unloadProfile_.samples = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(unloadProfileCumulativeSamples_, static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()))
        );
        unloadProfile_.averageMs = unloadProfileCumulativeMs_ / static_cast<double>(unloadProfileCumulativeSamples_);
        unloadProfile_.maxMs = std::max(unloadProfile_.maxMs, worldProfile.unload.maxMs);
    }
    if (worldProfile.unloadCount.samples > 0) {
        unloadCountProfileCumulativeValue_ += worldProfile.unloadCount.averageMs * static_cast<double>(worldProfile.unloadCount.samples);
        unloadCountProfileCumulativeSamples_ += worldProfile.unloadCount.samples;
        unloadCountProfile_.samples = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(unloadCountProfileCumulativeSamples_, static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()))
        );
        unloadCountProfile_.averageMs =
            unloadCountProfileCumulativeValue_ / static_cast<double>(unloadCountProfileCumulativeSamples_);
        unloadCountProfile_.maxMs = std::max(unloadCountProfile_.maxMs, worldProfile.unloadCount.maxMs);
    }
    if (worldProfile.saveFile.samples > 0) {
        saveFileProfileCumulativeMs_ += worldProfile.saveFile.averageMs * static_cast<double>(worldProfile.saveFile.samples);
        saveFileProfileCumulativeSamples_ += worldProfile.saveFile.samples;
        saveFileProfile_.samples = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(saveFileProfileCumulativeSamples_, static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()))
        );
        saveFileProfile_.averageMs = saveFileProfileCumulativeMs_ / static_cast<double>(saveFileProfileCumulativeSamples_);
        saveFileProfile_.maxMs = std::max(saveFileProfile_.maxMs, worldProfile.saveFile.maxMs);
    }
    if (worldProfile.saveCount.samples > 0) {
        saveCountProfileCumulativeValue_ += worldProfile.saveCount.averageMs * static_cast<double>(worldProfile.saveCount.samples);
        saveCountProfileCumulativeSamples_ += worldProfile.saveCount.samples;
        saveCountProfile_.samples = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(saveCountProfileCumulativeSamples_, static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()))
        );
        saveCountProfile_.averageMs =
            saveCountProfileCumulativeValue_ / static_cast<double>(saveCountProfileCumulativeSamples_);
        saveCountProfile_.maxMs = std::max(saveCountProfile_.maxMs, worldProfile.saveCount.maxMs);
    }
    if (worldProfile.getBlock.samples > 0) {
        getBlockProfileCumulativeMs_ += worldProfile.getBlock.averageMs * static_cast<double>(worldProfile.getBlock.samples);
        getBlockProfileCumulativeSamples_ += worldProfile.getBlock.samples;
        getBlockProfile_.samples = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(getBlockProfileCumulativeSamples_, static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()))
        );
        getBlockProfile_.averageMs = getBlockProfileCumulativeMs_ / static_cast<double>(getBlockProfileCumulativeSamples_);
        getBlockProfile_.maxMs = std::max(getBlockProfile_.maxMs, worldProfile.getBlock.maxMs);
    }
    if (worldProfile.generatedBlock.samples > 0) {
        generatedBlockProfileCumulativeMs_ +=
            worldProfile.generatedBlock.averageMs * static_cast<double>(worldProfile.generatedBlock.samples);
        generatedBlockProfileCumulativeSamples_ += worldProfile.generatedBlock.samples;
        generatedBlockProfile_.samples = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(
                generatedBlockProfileCumulativeSamples_,
                static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())
            )
        );
        generatedBlockProfile_.averageMs =
            generatedBlockProfileCumulativeMs_ / static_cast<double>(generatedBlockProfileCumulativeSamples_);
        generatedBlockProfile_.maxMs = std::max(generatedBlockProfile_.maxMs, worldProfile.generatedBlock.maxMs);
    }

    const auto finalizeStage = [](
        RuntimeProfileStage& stage,
        double& cumulativeMs,
        std::uint64_t& cumulativeSamples,
        double& totalMs,
        double& maxMs,
        std::uint32_t& count
    ) {
        if (count > 0) {
            cumulativeMs += totalMs;
            cumulativeSamples += count;
            stage.samples = static_cast<std::uint32_t>(
                std::min<std::uint64_t>(cumulativeSamples, static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()))
            );
            stage.averageMs = cumulativeMs / static_cast<double>(cumulativeSamples);
            stage.maxMs = std::max(stage.maxMs, maxMs);
        }
        totalMs = 0.0;
        maxMs = 0.0;
        count = 0;
    };

    finalizeStage(
        uploadProfile_,
        uploadProfileCumulativeMs_,
        uploadProfileCumulativeSamples_,
        uploadProfileTotalMs_,
        uploadProfileMaxMs_,
        uploadProfileSamples_
    );
    finalizeStage(
        uploadChunkProfile_,
        uploadChunkProfileCumulativeMs_,
        uploadChunkProfileCumulativeSamples_,
        uploadChunkProfileTotalMs_,
        uploadChunkProfileMaxMs_,
        uploadChunkProfileSamples_
    );
    finalizeStage(
        uploadCountProfile_,
        uploadCountProfileCumulativeValue_,
        uploadCountProfileCumulativeSamples_,
        uploadCountProfileTotalValue_,
        uploadCountProfileMaxValue_,
        uploadCountProfileSamples_
    );
    finalizeStage(
        uploadRemoveProfile_,
        uploadRemoveProfileCumulativeMs_,
        uploadRemoveProfileCumulativeSamples_,
        uploadRemoveProfileTotalMs_,
        uploadRemoveProfileMaxMs_,
        uploadRemoveProfileSamples_
    );
    finalizeStage(
        uploadInsertProfile_,
        uploadInsertProfileCumulativeMs_,
        uploadInsertProfileCumulativeSamples_,
        uploadInsertProfileTotalMs_,
        uploadInsertProfileMaxMs_,
        uploadInsertProfileSamples_
    );
    finalizeStage(
        uploadAllocProfile_,
        uploadAllocProfileCumulativeMs_,
        uploadAllocProfileCumulativeSamples_,
        uploadAllocProfileTotalMs_,
        uploadAllocProfileMaxMs_,
        uploadAllocProfileSamples_
    );
    finalizeStage(
        uploadCopyProfile_,
        uploadCopyProfileCumulativeMs_,
        uploadCopyProfileCumulativeSamples_,
        uploadCopyProfileTotalMs_,
        uploadCopyProfileMaxMs_,
        uploadCopyProfileSamples_
    );
    finalizeStage(
        drawCpuProfile_,
        drawCpuProfileCumulativeMs_,
        drawCpuProfileCumulativeSamples_,
        drawCpuProfileTotalMs_,
        drawCpuProfileMaxMs_,
        drawCpuProfileSamples_
    );
    finalizeStage(
        waitProfile_,
        waitProfileCumulativeMs_,
        waitProfileCumulativeSamples_,
        waitProfileTotalMs_,
        waitProfileMaxMs_,
        waitProfileSamples_
    );
    finalizeStage(
        acquireProfile_,
        acquireProfileCumulativeMs_,
        acquireProfileCumulativeSamples_,
        acquireProfileTotalMs_,
        acquireProfileMaxMs_,
        acquireProfileSamples_
    );
    finalizeStage(
        submitProfile_,
        submitProfileCumulativeMs_,
        submitProfileCumulativeSamples_,
        submitProfileTotalMs_,
        submitProfileMaxMs_,
        submitProfileSamples_
    );
    finalizeStage(
        presentProfile_,
        presentProfileCumulativeMs_,
        presentProfileCumulativeSamples_,
        presentProfileTotalMs_,
        presentProfileMaxMs_,
        presentProfileSamples_
    );
    finalizeStage(
        frameProfile_,
        frameProfileCumulativeMs_,
        frameProfileCumulativeSamples_,
        frameProfileTotalMs_,
        frameProfileMaxMs_,
        frameProfileSamples_
    );
    finalizeStage(
        worldMeshLockProfile_,
        worldMeshLockProfileCumulativeMs_,
        worldMeshLockProfileCumulativeSamples_,
        worldMeshLockProfileTotalMs_,
        worldMeshLockProfileMaxMs_,
        worldMeshLockProfileSamples_
    );
    finalizeStage(
        worldMeshSaveQueueProfile_,
        worldMeshSaveQueueProfileCumulativeMs_,
        worldMeshSaveQueueProfileCumulativeSamples_,
        worldMeshSaveQueueProfileTotalMs_,
        worldMeshSaveQueueProfileMaxMs_,
        worldMeshSaveQueueProfileSamples_
    );
    finalizeStage(
        worldMeshFinalizeProfile_,
        worldMeshFinalizeProfileCumulativeMs_,
        worldMeshFinalizeProfileCumulativeSamples_,
        worldMeshFinalizeProfileTotalMs_,
        worldMeshFinalizeProfileMaxMs_,
        worldMeshFinalizeProfileSamples_
    );
    finalizeStage(
        worldMeshRenderDrainProfile_,
        worldMeshRenderDrainProfileCumulativeMs_,
        worldMeshRenderDrainProfileCumulativeSamples_,
        worldMeshRenderDrainProfileTotalMs_,
        worldMeshRenderDrainProfileMaxMs_,
        worldMeshRenderDrainProfileSamples_
    );
    finalizeStage(
        unloadProfile_,
        unloadProfileCumulativeMs_,
        unloadProfileCumulativeSamples_,
        unloadProfileTotalMs_,
        unloadProfileMaxMs_,
        unloadProfileSamples_
    );
    finalizeStage(
        unloadCountProfile_,
        unloadCountProfileCumulativeValue_,
        unloadCountProfileCumulativeSamples_,
        unloadCountProfileTotalValue_,
        unloadCountProfileMaxValue_,
        unloadCountProfileSamples_
    );
    finalizeStage(
        saveFileProfile_,
        saveFileProfileCumulativeMs_,
        saveFileProfileCumulativeSamples_,
        saveFileProfileTotalMs_,
        saveFileProfileMaxMs_,
        saveFileProfileSamples_
    );
    finalizeStage(
        saveCountProfile_,
        saveCountProfileCumulativeValue_,
        saveCountProfileCumulativeSamples_,
        saveCountProfileTotalValue_,
        saveCountProfileMaxValue_,
        saveCountProfileSamples_
    );
    finalizeStage(
        raycastWaitProfile_,
        raycastWaitProfileCumulativeMs_,
        raycastWaitProfileCumulativeSamples_,
        raycastWaitProfileTotalMs_,
        raycastWaitProfileMaxMs_,
        raycastWaitProfileSamples_
    );
    finalizeStage(
        raycastProfile_,
        raycastProfileCumulativeMs_,
        raycastProfileCumulativeSamples_,
        raycastProfileTotalMs_,
        raycastProfileMaxMs_,
        raycastProfileSamples_
    );
    finalizeStage(
        collisionWaitProfile_,
        collisionWaitProfileCumulativeMs_,
        collisionWaitProfileCumulativeSamples_,
        collisionWaitProfileTotalMs_,
        collisionWaitProfileMaxMs_,
        collisionWaitProfileSamples_
    );
    finalizeStage(
        collisionProfile_,
        collisionProfileCumulativeMs_,
        collisionProfileCumulativeSamples_,
        collisionProfileTotalMs_,
        collisionProfileMaxMs_,
        collisionProfileSamples_
    );
    finalizeStage(
        getBlockProfile_,
        getBlockProfileCumulativeMs_,
        getBlockProfileCumulativeSamples_,
        getBlockProfileTotalMs_,
        getBlockProfileMaxMs_,
        getBlockProfileSamples_
    );
    finalizeStage(
        generatedBlockProfile_,
        generatedBlockProfileCumulativeMs_,
        generatedBlockProfileCumulativeSamples_,
        generatedBlockProfileTotalMs_,
        generatedBlockProfileMaxMs_,
        generatedBlockProfileSamples_
    );
}

void VulkanVoxelApp::LoadStaticDebugInfo() {
    gpuName_ = physicalDeviceProperties_.deviceName;
    apiVersionString_ = FormatVersion(physicalDeviceProperties_.apiVersion);
    driverVersionString_ = FormatDriverVersion(
        physicalDeviceProperties_.vendorID,
        physicalDeviceProperties_.driverVersion
    );

    VkPhysicalDeviceMemoryProperties memoryProperties{};
    vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &memoryProperties);
    totalVramBytes_ = 0;
    for (std::uint32_t i = 0; i < memoryProperties.memoryHeapCount; ++i) {
        if ((memoryProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) {
            totalVramBytes_ += memoryProperties.memoryHeaps[i].size;
        }
    }

    HKEY key = nullptr;
    if (RegOpenKeyExA(
            HKEY_LOCAL_MACHINE,
            "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
            0,
            KEY_READ,
            &key) == ERROR_SUCCESS) {
        char buffer[256] = {};
        DWORD bufferSize = sizeof(buffer);
        if (RegQueryValueExA(
                key,
                "ProcessorNameString",
                nullptr,
                nullptr,
                reinterpret_cast<LPBYTE>(buffer),
                &bufferSize) == ERROR_SUCCESS) {
            cpuName_ = buffer;
        }
        RegCloseKey(key);
    }

    if (cpuName_.empty()) {
        cpuName_ = "UNKNOWN CPU";
    }
}

void VulkanVoxelApp::RebuildOverlayVertices() {
    overlayVertices_.clear();
    celestialVertexCount_ = 0;
    const float overlayUiScale = GetOverlayUiScale(swapChainExtent_);
    const float overlayGlyphScale = kOverlayGlyphScale * overlayUiScale;
    const float overlayMargin = kOverlayMargin * overlayUiScale;
    const float overlayLineGap = kOverlayLineGap * overlayUiScale;

    if (crosshairLoaded_) {
        const float width = crosshairWidth_;
        const float height = crosshairHeight_;
        const float left = static_cast<float>(swapChainExtent_.width) * 0.5f - width * 0.5f;
        const float top = static_cast<float>(swapChainExtent_.height) * 0.5f - height * 0.5f;
        AppendOverlayQuad(
            overlayVertices_,
            left,
            top,
            left + width,
            top + height,
            crosshairU0_,
            crosshairV0_,
            crosshairU1_,
            crosshairV1_,
            1.0f,
            1.0f,
            1.0f,
            swapChainExtent_
        );
    }

    if (!debugOverlayEnabled_) {
        overlayVertexCount_ = static_cast<std::uint32_t>(overlayVertices_.size());
        return;
    }

    const int chunkX = GetChunkCoordinate(cameraPosition_.x, kChunkSizeX);
    const int chunkZ = GetChunkCoordinate(cameraPosition_.z, kChunkSizeZ);
    const int subChunk = std::clamp(static_cast<int>(cameraPosition_.y) / kSubChunkSize, 0, kSubChunkCountY - 1);
    const std::uint32_t faceCount = worldIndexCount_ / 6;
    const std::uint32_t triangleCount = worldIndexCount_ / 3;
    std::vector<std::string> leftLines;
    std::vector<std::string> lowerLeftLines;
    std::vector<std::string> rightLines;
    std::vector<std::string> lowerRightLines;

    std::uint64_t totalPoolCapacityQuads = 0;
    std::uint64_t totalCommittedQuads = 0;
    std::uint64_t totalFreeCommittedQuads = 0;
    std::uint32_t usedPoolCount = 0;
    for (const WorldQuadPool& pool : worldQuadPools_) {
        totalPoolCapacityQuads += pool.capacityQuads;
        totalCommittedQuads += pool.committedQuads;

        std::uint64_t freeCommittedQuads = 0;
        for (const WorldQuadRange& range : pool.freeRanges) {
            freeCommittedQuads += range.count;
        }
        totalFreeCommittedQuads += freeCommittedQuads;

        const std::uint64_t liveQuads =
            static_cast<std::uint64_t>(pool.committedQuads) > freeCommittedQuads
                ? static_cast<std::uint64_t>(pool.committedQuads) - freeCommittedQuads
                : 0;
        if (liveQuads > 0) {
            ++usedPoolCount;
        }
    }
    const std::uint64_t totalLiveQuads =
        totalCommittedQuads > totalFreeCommittedQuads ? totalCommittedQuads - totalFreeCommittedQuads : 0;

    {
        std::ostringstream stream;
        stream << "FPS: " << std::setw(5) << currentFps_ << " [" << std::fixed << std::setprecision(4)
               << currentFrameTimeMs_ << "MS]";
        leftLines.push_back(stream.str());
    }
    leftLines.push_back(
        "POS: X " + FormatFloat(cameraPosition_.x, 2) +
        " Y " + FormatFloat(cameraPosition_.y, 2) +
        " Z " + FormatFloat(cameraPosition_.z, 2)
    );
    leftLines.push_back(
        "ROT: YAW " + FormatFloat(cameraYaw_, 2) +
        " PITCH " + FormatFloat(cameraPitch_, 2) +
        " [" + std::string(GetCardinalDirectionLabel(cameraYaw_)) + "]"
    );
    leftLines.push_back(
        "CHUNK: X " + std::to_string(chunkX) +
        " Z " + std::to_string(chunkZ) +
        " SUB: " + std::to_string(subChunk)
    );
    leftLines.push_back("LOADED CHUNKS: " + std::to_string(loadedChunkCount_));
    leftLines.push_back("FACE COUNT: " + std::to_string(faceCount));
    leftLines.push_back(FormatGameTime(globalTick_));

    lowerLeftLines.push_back(FormatRuntimeProfileLine("RAYWAIT", raycastWaitProfile_));
    lowerLeftLines.push_back(FormatRuntimeProfileLine("RAYWALK", raycastProfile_));
    lowerLeftLines.push_back(FormatRuntimeProfileLine("COLWAIT", collisionWaitProfile_));
    lowerLeftLines.push_back(FormatRuntimeProfileLine("COLWALK", collisionProfile_));
    lowerLeftLines.push_back(FormatRuntimeProfileLine("GETBLOCK", getBlockProfile_));
    lowerLeftLines.push_back(FormatRuntimeProfileLine("GENBLOCK", generatedBlockProfile_));

    rightLines.push_back("GPU: " + gpuName_);
    rightLines.push_back("CPU: " + cpuName_);
    rightLines.push_back(FormatUsageLine("RAM", usedRamBytes_, totalRamBytes_));
    rightLines.push_back(
        usedVramBytes_.has_value()
            ? FormatUsageLine("VRAM", usedVramBytes_.value(), totalVramBytes_)
            : std::string("VRAM USED: N/A / ") + FormatGigabytes(totalVramBytes_) + " GB"
    );
    rightLines.push_back(
        "RES: " + std::to_string(swapChainExtent_.width) + " X " + std::to_string(swapChainExtent_.height)
    );
    rightLines.push_back("API: VULKAN " + apiVersionString_);
    rightLines.push_back("DRIVER: " + driverVersionString_);
    rightLines.push_back("DRAW CALLS: " + std::to_string(drawCallCount_));
    rightLines.push_back("VERTEX COUNT: " + std::to_string(worldVertexCount_));
    rightLines.push_back("TRIANGLE COUNT: " + std::to_string(triangleCount));

    float leftY = overlayMargin;
    for (const std::string& rawLine : leftLines) {
        const std::string line = SanitizeOverlayText(rawLine, overlayFontGlyphs_);
        AppendOutlinedText(
            overlayVertices_,
            overlayFontGlyphs_,
            line,
            overlayMargin,
            leftY,
            overlayGlyphScale,
            swapChainExtent_
        );
        leftY += static_cast<float>(overlayFontLineHeight_) * overlayGlyphScale + overlayLineGap;
    }

    float rightY = overlayMargin;
    for (const std::string& rawLine : rightLines) {
        const std::string line = SanitizeOverlayText(rawLine, overlayFontGlyphs_);
        const float startX = static_cast<float>(swapChainExtent_.width) -
                             overlayMargin -
                             GetOverlayTextWidth(line, overlayFontGlyphs_, overlayGlyphScale);
        AppendOutlinedText(
            overlayVertices_,
            overlayFontGlyphs_,
            line,
            startX,
            rightY,
            overlayGlyphScale,
            swapChainExtent_
        );
        rightY += static_cast<float>(overlayFontLineHeight_) * overlayGlyphScale + overlayLineGap;
    }

    const float lineAdvance = static_cast<float>(overlayFontLineHeight_) * overlayGlyphScale + overlayLineGap;
    float lowerLeftY = static_cast<float>(swapChainExtent_.height) - overlayMargin -
                       lineAdvance * static_cast<float>(lowerLeftLines.size());
    for (const std::string& rawLine : lowerLeftLines) {
        const std::string line = SanitizeOverlayText(rawLine, overlayFontGlyphs_);
        AppendOutlinedText(
            overlayVertices_,
            overlayFontGlyphs_,
            line,
            overlayMargin,
            lowerLeftY,
            overlayGlyphScale,
            swapChainExtent_
        );
        lowerLeftY += lineAdvance;
    }

    float lowerRightY = static_cast<float>(swapChainExtent_.height) - overlayMargin -
                        lineAdvance * static_cast<float>(lowerRightLines.size());
    for (const std::string& rawLine : lowerRightLines) {
        const std::string line = SanitizeOverlayText(rawLine, overlayFontGlyphs_);
        const float startX = static_cast<float>(swapChainExtent_.width) -
                             overlayMargin -
                             GetOverlayTextWidth(line, overlayFontGlyphs_, overlayGlyphScale);
        AppendOutlinedText(
            overlayVertices_,
            overlayFontGlyphs_,
            line,
            startX,
            lowerRightY,
            overlayGlyphScale,
            swapChainExtent_
        );
        lowerRightY += lineAdvance;
    }

    if (overlayVertices_.size() > kMaxOverlayVertexCount) {
        throw std::runtime_error("Overlay vertex buffer is too small.");
    }

    overlayVertexCount_ = static_cast<std::uint32_t>(overlayVertices_.size());
}

void VulkanVoxelApp::UpdateCelestialVertices(std::uint32_t frameIndex) {
    sunVertexCount_ = 0;
    moonVertexCount_ = 0;
    celestialVertexCount_ = 0;

    if (frameIndex >= celestialVertexBuffersMemory_.size() || celestialVertexBuffersMemory_[frameIndex] == VK_NULL_HANDLE) {
        return;
    }

    std::vector<OverlayVertex> celestialVertices;
    celestialVertices.reserve(12);

    const Vec3 renderCameraForward = GetRenderCameraForward();
    Vec3 renderCameraRight = Normalize(Cross(renderCameraForward, {0.0f, 1.0f, 0.0f}));
    if (Length(renderCameraRight) <= 0.0001f) {
        renderCameraRight = {1.0f, 0.0f, 0.0f};
    }
    const Vec3 renderCameraUp = Normalize(Cross(renderCameraRight, renderCameraForward));
    const float aspectRatio = swapChainExtent_.height > 0
        ? static_cast<float>(swapChainExtent_.width) / static_cast<float>(swapChainExtent_.height)
        : 16.0f / 9.0f;
    const float tanHalfFovY = std::tan(70.0f * 0.5f * kPi / 180.0f);
    const float tanHalfFovX = tanHalfFovY * aspectRatio;

    const double fractionalTick = static_cast<double>(globalTick_) +
        physicsAccumulatorSeconds_ * static_cast<double>(kPhysicsTicksPerSecond);
    const double dayFraction = std::fmod(fractionalTick, static_cast<double>(kDayLengthTicks)) /
        static_cast<double>(kDayLengthTicks);
    const float sunAngleRadians = static_cast<float>(dayFraction * 2.0 * kPi - kPi * 0.5);
    const Vec3 sunDirection = Normalize({std::cos(sunAngleRadians), std::sin(sunAngleRadians), 0.0f});
    const Vec3 moonDirection = sunDirection * -1.0f;

    const auto appendCelestialBody = [&](bool loaded, const Vec3& direction, float u0, float v0, float u1, float v1, float width, float height, std::uint32_t& outVertexCount) {
        if (!loaded) {
            outVertexCount = 0;
            return;
        }

        float centerX = 0.0f;
        float centerY = 0.0f;
        if (!ProjectDirectionToScreen(
                direction,
                renderCameraForward,
                renderCameraRight,
                renderCameraUp,
                tanHalfFovX,
                tanHalfFovY,
                swapChainExtent_,
                centerX,
                centerY)) {
            outVertexCount = 0;
            return;
        }

        const std::size_t beforeCount = celestialVertices.size();
        const float scaledWidth = width * 0.5f;
        const float scaledHeight = height * 0.5f;
        const float left = centerX - scaledWidth * 0.5f;
        const float top = centerY - scaledHeight * 0.5f;
        AppendOverlayQuad(
            celestialVertices,
            left,
            top,
            left + scaledWidth,
            top + scaledHeight,
            u0,
            v0,
            u1,
            v1,
            1.0f,
            1.0f,
            1.0f,
            swapChainExtent_
        );
        outVertexCount = static_cast<std::uint32_t>(celestialVertices.size() - beforeCount);
    };

    appendCelestialBody(sunLoaded_, sunDirection, sunU0_, sunV0_, sunU1_, sunV1_, sunWidth_, sunHeight_, sunVertexCount_);
    appendCelestialBody(moonLoaded_, moonDirection, moonU0_, moonV0_, moonU1_, moonV1_, moonWidth_, moonHeight_, moonVertexCount_);
    celestialVertexCount_ = static_cast<std::uint32_t>(celestialVertices.size());

    if (celestialVertices.empty()) {
        return;
    }

    void* mappedData = nullptr;
    const VkDeviceSize dataSize = sizeof(OverlayVertex) * static_cast<VkDeviceSize>(celestialVertices.size());
    if (vkMapMemory(device_, celestialVertexBuffersMemory_[frameIndex], 0, dataSize, 0, &mappedData) != VK_SUCCESS) {
        throw std::runtime_error("Failed to map celestial vertex buffer.");
    }

    std::memcpy(mappedData, celestialVertices.data(), static_cast<std::size_t>(dataSize));
    vkUnmapMemory(device_, celestialVertexBuffersMemory_[frameIndex]);
}

void VulkanVoxelApp::UploadOverlayVertices() {
    if (overlayVertexCount_ == 0) {
        return;
    }

    void* mappedData = nullptr;
    const VkDeviceSize dataSize = sizeof(OverlayVertex) * static_cast<VkDeviceSize>(overlayVertices_.size());
    if (vkMapMemory(device_, overlayVertexBufferMemory_, 0, dataSize, 0, &mappedData) != VK_SUCCESS) {
        throw std::runtime_error("Failed to map overlay vertex buffer.");
    }

    std::memcpy(mappedData, overlayVertices_.data(), static_cast<std::size_t>(dataSize));
    vkUnmapMemory(device_, overlayVertexBufferMemory_);
}

void VulkanVoxelApp::UpdateSelectionBuffer() {
    std::array<SelectionVertex, 24> vertices{};
    selectionVertexCount_ = 0;

    if (selectionVertexBufferMemory_ == VK_NULL_HANDLE) {
        return;
    }

    if (selectedBlockHit_.has_value()) {
        const float minX = static_cast<float>(selectedBlockHit_->blockX) - kSelectionExpand;
        const float minY = static_cast<float>(selectedBlockHit_->blockY) - kSelectionExpand;
        const float minZ = static_cast<float>(selectedBlockHit_->blockZ) - kSelectionExpand;
        const float maxX = static_cast<float>(selectedBlockHit_->blockX + 1) + kSelectionExpand;
        const float maxY = static_cast<float>(selectedBlockHit_->blockY + 1) + kSelectionExpand;
        const float maxZ = static_cast<float>(selectedBlockHit_->blockZ + 1) + kSelectionExpand;

        const Vec3 corners[8] = {
            {minX, minY, minZ},
            {maxX, minY, minZ},
            {maxX, minY, maxZ},
            {minX, minY, maxZ},
            {minX, maxY, minZ},
            {maxX, maxY, minZ},
            {maxX, maxY, maxZ},
            {minX, maxY, maxZ},
        };

        constexpr int edgeIndices[24] = {
            0, 1, 1, 2, 2, 3, 3, 0,
            4, 5, 5, 6, 6, 7, 7, 4,
            0, 4, 1, 5, 2, 6, 3, 7,
        };

        for (int i = 0; i < 24; ++i) {
            const Vec3& point = corners[edgeIndices[i]];
            vertices[static_cast<std::size_t>(i)] = {{point.x, point.y, point.z}};
        }
        selectionVertexCount_ = 24;
    }

    void* mappedData = nullptr;
    const VkDeviceSize dataSize = sizeof(SelectionVertex) * vertices.size();
    if (vkMapMemory(device_, selectionVertexBufferMemory_, 0, dataSize, 0, &mappedData) != VK_SUCCESS) {
        throw std::runtime_error("Failed to map selection vertex buffer.");
    }
    std::memcpy(mappedData, vertices.data(), static_cast<std::size_t>(dataSize));
    vkUnmapMemory(device_, selectionVertexBufferMemory_);
}

void VulkanVoxelApp::UpdateEntityColliderBuffer() {
    std::array<SelectionVertex, 24> vertices{};
    entityColliderVertexCount_ = 0;

    if (entityColliderVertexBufferMemory_ == VK_NULL_HANDLE || !entityColliderVisible_) {
        return;
    }

    const Vec3 feet = GetInterpolatedPlayerFeetPosition();
    const float minX = feet.x - kPlayerHalfWidth;
    const float minY = feet.y;
    const float minZ = feet.z - kPlayerHalfWidth;
    const float maxX = feet.x + kPlayerHalfWidth;
    const float maxY = feet.y + kPlayerHeight;
    const float maxZ = feet.z + kPlayerHalfWidth;

    const Vec3 corners[8] = {
        {minX, minY, minZ},
        {maxX, minY, minZ},
        {maxX, minY, maxZ},
        {minX, minY, maxZ},
        {minX, maxY, minZ},
        {maxX, maxY, minZ},
        {maxX, maxY, maxZ},
        {minX, maxY, maxZ},
    };

    constexpr int edgeIndices[24] = {
        0, 1, 1, 2, 2, 3, 3, 0,
        4, 5, 5, 6, 6, 7, 7, 4,
        0, 4, 1, 5, 2, 6, 3, 7,
    };

    for (int i = 0; i < 24; ++i) {
        const Vec3& point = corners[edgeIndices[i]];
        vertices[static_cast<std::size_t>(i)] = {{point.x, point.y, point.z}};
    }
    entityColliderVertexCount_ = 24;

    void* mappedData = nullptr;
    const VkDeviceSize dataSize = sizeof(SelectionVertex) * vertices.size();
    if (vkMapMemory(device_, entityColliderVertexBufferMemory_, 0, dataSize, 0, &mappedData) != VK_SUCCESS) {
        throw std::runtime_error("Failed to map entity collider vertex buffer.");
    }
    std::memcpy(mappedData, vertices.data(), static_cast<std::size_t>(dataSize));
    vkUnmapMemory(device_, entityColliderVertexBufferMemory_);
}

int main() {
    VulkanVoxelApp app;

    try {
        return app.Run();
    } catch (const std::exception& e) {
        OutputDebugStringA("Fatal stage: ");
        OutputDebugStringA(gFatalStage);
        OutputDebugStringA("\n");
        std::cerr << "Fatal stage: " << gFatalStage << '\n';
        try {
            const char* message = e.what();
            if (message != nullptr) {
                OutputDebugStringA(message);
                OutputDebugStringA("\n");
                std::cerr << message << '\n';
            }
        } catch (...) {
        }
        return 1;
    }
}
