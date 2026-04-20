#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "MathTypes.h"
#include "VoxelWorld.h"
#include "WorldSettings.h"

#include <chrono>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <array>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

struct QueueFamilyIndices {
    std::optional<std::uint32_t> graphicsFamily;
    std::optional<std::uint32_t> presentFamily;

    bool IsComplete() const;
};

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities{};
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

struct UniformBufferObject {
    Mat4 viewProj;
    float cameraRight[4];
    float cameraUp[4];
    float cameraForward[4];
    float projectionParams[4];
};

struct TerrainPushConstants {
    std::int32_t chunkMinX = 0;
    std::int32_t chunkMinZ = 0;
};

struct SelectionVertex {
    float position[3];
};

struct WorldRenderBatch {
    PendingChunkId id{};
    std::uint32_t poolIndex = 0;
    std::uint32_t quadOffset = 0;
    std::uint32_t quadCapacity = 0;
    std::uint32_t quadCount = 0;
    std::uint32_t vertexCount = 0;
    std::uint32_t indexCount = 0;
};

struct WorldQuadRange {
    std::uint32_t offset = 0;
    std::uint32_t count = 0;
};

struct WorldQuadPool {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mappedData = nullptr;
    std::uint32_t capacityQuads = 0;
    std::uint32_t committedQuads = 0;
    std::vector<WorldQuadRange> freeRanges;
};

enum class CameraViewMode {
    FirstPerson,
    ThirdPersonRear,
    ThirdPersonFront,
};

enum class MovementMode {
    Fly,
    Walk,
};

extern const char* gFatalStage;

class VulkanVoxelApp {
public:
    int Run();

private:
    void InitWindow();
    void InitVulkan();
    void MainLoop();
    void Cleanup();

    void ProcessInput(float deltaTime);
    void StepFixedPhysics();
    float GetPhysicsInterpolationAlpha() const;
    Vec3 GetInterpolatedCameraPosition() const;
    Vec3 GetInterpolatedPlayerFeetPosition() const;
    Vec3 GetForwardVector() const;
    Vec3 GetHorizontalForwardVector() const;
    Vec3 GetRightVector() const;
    Vec3 GetPlayerFeetPosition() const;
    Vec3 GetRenderCameraPosition();
    Vec3 GetRenderCameraForward();
    bool IsThirdPersonView() const;
    bool IsSolidBlockAt(int worldX, int worldY, int worldZ);
    bool IsPlayerCollidingAt(const Vec3& eyePosition);
    Vec3 MovePlayerAxis(const Vec3& startPosition, const Vec3& axisDelta, bool& hitNegativeY);
    BlockRaycastHit TraceSelectedBlock();
    void UpdateWorldMeshIfNeeded();
    void ToggleFullscreen();
    void CleanupSwapChain();
    void RecreateSwapChain();

    void CreateInstance();
    void CreateSurface();
    void PickPhysicalDevice();
    void CreateLogicalDevice();
    void CreateSwapChain();
    void CreateImageViews();
    void CreateCommandPool();
    void CreateDescriptorSetLayout();
    void CreateRenderPass();
    void CreateDepthResources();
    void CreateTextureImage();
    void CreateTextureImageView();
    void CreateTextureSampler();
    void LoadPlayerMesh();
    void CreatePlayerTextureImage();
    void CreatePlayerTextureImageView();
    void CreatePlayerTextureSampler();
    void CreatePlayerBuffers();
    void UpdatePlayerRenderMesh();
    void DestroyPlayerBuffers();
    void LoadOverlayFont();
    void CreateSelectionBuffer();
    void UpdateSelectionBuffer();
    void CreateEntityColliderBuffer();
    void UpdateEntityColliderBuffer();
    void BuildWorldMesh();
    void RequestWorldMeshBuild();
    void StartWorldMeshWorker();
    void StopWorldMeshWorker();
    void ConsumeCompletedWorldMesh();
    void UploadWorldRenderUpdate(const WorldRenderUpdate& update);
    void DestroyWorldMeshBuffers();
    void DestroyWorldRenderBatch(WorldRenderBatch& batch);
    void DestroyWorldQuadPool(WorldQuadPool& pool);
    void UploadWorldRenderBatch(WorldRenderBatch& batch, const ChunkMeshBatchData& batchData);
    void TryCleanupRetiredWorldRenderBatches();
    std::uint32_t CreateWorldQuadPool(std::uint32_t capacityQuads);
    WorldQuadRange AllocateWorldQuadRange(WorldQuadPool& pool, std::uint32_t quadCount);
    void ReleaseWorldQuadRange(WorldQuadPool& pool, std::uint32_t offset, std::uint32_t count);
    void CreateOverlayBuffer();
    void CreateUniformBuffers();
    void CreateDescriptorPool();
    void CreateDescriptorSets();
    void CreatePipelines();
    void CreateFramebuffers();
    void CreateCommandBuffers();
    void CreateSyncObjects();
    void DrawFrame();
    void PrepareScreenshotCapture();
    void RecordScreenshotCopyCommands(VkCommandBuffer commandBuffer, std::uint32_t imageIndex);
    void FinalizeScreenshotCapture();
    void SaveScreenshotPng(
        const std::vector<std::uint8_t>& rgbaPixels,
        std::uint32_t width,
        std::uint32_t height
    ) const;
    void UpdateUniformBuffer(std::uint32_t frameIndex);
    void UpdateOverlayText(float deltaTime);
    void UpdateCelestialVertices(std::uint32_t frameIndex);
    void RefreshSystemUsageStats();
    void AccumulateRuntimeProfileSample(double elapsedMs, double& totalMs, double& maxMs, std::uint32_t& count);
    void RefreshRuntimeProfileStats();
    void LoadStaticDebugInfo();
    void RebuildOverlayVertices();
    void UploadOverlayVertices();

    QueueFamilyIndices FindQueueFamilies(VkPhysicalDevice device) const;
    bool IsDeviceSuitable(VkPhysicalDevice device) const;
    bool CheckDeviceExtensionSupport(VkPhysicalDevice device) const;
    SwapChainSupportDetails QuerySwapChainSupport(VkPhysicalDevice device) const;
    VkSurfaceFormatKHR ChooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& formats) const;
    VkPresentModeKHR ChooseSwapPresentMode(const std::vector<VkPresentModeKHR>& presentModes) const;
    VkExtent2D ChooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) const;
    VkShaderModule CreateShaderModule(const std::vector<char>& code) const;
    void RecordCommandBuffer(VkCommandBuffer commandBuffer, std::uint32_t imageIndex);
    std::vector<char> ReadFile(const char* path) const;
    std::uint32_t FindMemoryType(std::uint32_t typeFilter, VkMemoryPropertyFlags properties) const;
    void CreateBuffer(
        VkDeviceSize size,
        VkBufferUsageFlags usage,
        VkMemoryPropertyFlags properties,
        VkBuffer& buffer,
        VkDeviceMemory& bufferMemory
    ) const;
    void CopyBuffer(VkBuffer sourceBuffer, VkBuffer destinationBuffer, VkDeviceSize size) const;
    void CreateImage(
        std::uint32_t width,
        std::uint32_t height,
        VkFormat format,
        VkImageTiling tiling,
        VkImageUsageFlags usage,
        VkMemoryPropertyFlags properties,
        VkImage& image,
        VkDeviceMemory& imageMemory,
        std::uint32_t mipLevels = 1,
        std::uint32_t arrayLayers = 1
    ) const;
    VkImageView CreateImageView(
        VkImage image,
        VkFormat format,
        VkImageAspectFlags aspectFlags,
        std::uint32_t mipLevels = 1,
        VkImageViewType viewType = VK_IMAGE_VIEW_TYPE_2D,
        std::uint32_t layerCount = 1
    ) const;
    void TransitionImageLayout(
        VkImage image,
        VkFormat format,
        VkImageLayout oldLayout,
        VkImageLayout newLayout,
        std::uint32_t mipLevels = 1,
        std::uint32_t layerCount = 1
    ) const;
    void CopyBufferToImage(
        VkBuffer buffer,
        VkImage image,
        std::uint32_t width,
        std::uint32_t height,
        std::uint32_t layerCount = 1
    ) const;
    void GenerateMipmaps(VkImage image, VkFormat imageFormat, std::uint32_t width, std::uint32_t height, std::uint32_t mipLevels) const;
    VkCommandBuffer BeginSingleTimeCommands() const;
    void EndSingleTimeCommands(VkCommandBuffer commandBuffer) const;
    VkFormat FindSupportedFormat(
        const std::vector<VkFormat>& candidates,
        VkImageTiling tiling,
        VkFormatFeatureFlags features
    ) const;
    VkFormat FindDepthFormat() const;
    bool HasStencilComponent(VkFormat format) const;
    std::optional<std::uint64_t> QueryVideoMemoryUsageBytes() const;

    GLFWwindow* window_ = nullptr;

    VkInstance instance_ = VK_NULL_HANDLE;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue graphicsQueue_ = VK_NULL_HANDLE;
    VkQueue presentQueue_ = VK_NULL_HANDLE;

    VkSwapchainKHR swapChain_ = VK_NULL_HANDLE;
    std::vector<VkImage> swapChainImages_;
    std::vector<VkImageView> swapChainImageViews_;
    VkFormat swapChainImageFormat_ = VK_FORMAT_UNDEFINED;
    VkExtent2D swapChainExtent_{};
    VkPresentModeKHR presentMode_ = VK_PRESENT_MODE_FIFO_KHR;
    VkPhysicalDeviceProperties physicalDeviceProperties_{};

    VkDescriptorSetLayout descriptorSetLayout_ = VK_NULL_HANDLE;
    VkRenderPass renderPass_ = VK_NULL_HANDLE;

    VkPipelineLayout worldPipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline worldPipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout terrainPipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline terrainPipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout skyPipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline skyPipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout overlayPipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline overlayPipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout selectionPipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline selectionPipeline_ = VK_NULL_HANDLE;

    VkImage depthImage_ = VK_NULL_HANDLE;
    VkDeviceMemory depthImageMemory_ = VK_NULL_HANDLE;
    VkImageView depthImageView_ = VK_NULL_HANDLE;
    VkFormat depthFormat_ = VK_FORMAT_UNDEFINED;

    VkImage textureImage_ = VK_NULL_HANDLE;
    VkDeviceMemory textureImageMemory_ = VK_NULL_HANDLE;
    VkImageView textureImageView_ = VK_NULL_HANDLE;
    VkSampler textureSampler_ = VK_NULL_HANDLE;
    std::uint32_t textureMipLevels_ = 1;
    VkImage playerTextureImage_ = VK_NULL_HANDLE;
    VkDeviceMemory playerTextureImageMemory_ = VK_NULL_HANDLE;
    VkImageView playerTextureImageView_ = VK_NULL_HANDLE;
    VkSampler playerTextureSampler_ = VK_NULL_HANDLE;
    VkImage overlayFontImage_ = VK_NULL_HANDLE;
    VkDeviceMemory overlayFontImageMemory_ = VK_NULL_HANDLE;
    VkImageView overlayFontImageView_ = VK_NULL_HANDLE;
    VkSampler overlayFontSampler_ = VK_NULL_HANDLE;
    VkImage sunImage_ = VK_NULL_HANDLE;
    VkDeviceMemory sunImageMemory_ = VK_NULL_HANDLE;
    VkImageView sunImageView_ = VK_NULL_HANDLE;
    VkSampler sunSampler_ = VK_NULL_HANDLE;
    VkImage moonImage_ = VK_NULL_HANDLE;
    VkDeviceMemory moonImageMemory_ = VK_NULL_HANDLE;
    VkImageView moonImageView_ = VK_NULL_HANDLE;
    VkSampler moonSampler_ = VK_NULL_HANDLE;

    std::uint32_t worldVertexCount_ = 0;
    std::uint32_t worldIndexCount_ = 0;
    std::vector<WorldQuadPool> worldQuadPools_;
    std::unordered_map<PendingChunkId, WorldRenderBatch, PendingChunkIdHash> worldRenderBatches_;
    std::vector<VkBuffer> playerVertexBuffers_;
    std::vector<VkDeviceMemory> playerVertexBuffersMemory_;
    std::vector<VkDeviceSize> playerVertexBufferCapacities_;
    std::uint32_t playerVertexCount_ = 0;
    VkBuffer playerIndexBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory playerIndexBufferMemory_ = VK_NULL_HANDLE;
    std::uint32_t playerIndexCount_ = 0;
    VkDeviceSize playerIndexBufferCapacity_ = 0;

    VkBuffer overlayVertexBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory overlayVertexBufferMemory_ = VK_NULL_HANDLE;
    std::vector<VkBuffer> celestialVertexBuffers_;
    std::vector<VkDeviceMemory> celestialVertexBuffersMemory_;
    std::vector<OverlayVertex> overlayVertices_;
    std::uint32_t sunVertexCount_ = 0;
    std::uint32_t moonVertexCount_ = 0;
    std::uint32_t celestialVertexCount_ = 0;
    std::uint32_t overlayVertexCount_ = 0;
    bool overlayDirty_ = false;
    VkBuffer selectionVertexBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory selectionVertexBufferMemory_ = VK_NULL_HANDLE;
    std::uint32_t selectionVertexCount_ = 0;
    VkBuffer entityColliderVertexBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory entityColliderVertexBufferMemory_ = VK_NULL_HANDLE;
    std::uint32_t entityColliderVertexCount_ = 0;

    std::vector<VkBuffer> uniformBuffers_;
    std::vector<VkDeviceMemory> uniformBuffersMemory_;
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> descriptorSets_;
    std::vector<VkDescriptorSet> playerDescriptorSets_;
    std::vector<VkDescriptorSet> overlayDescriptorSets_;
    std::vector<VkDescriptorSet> sunDescriptorSets_;
    std::vector<VkDescriptorSet> moonDescriptorSets_;

    std::vector<VkFramebuffer> swapChainFramebuffers_;

    VkCommandPool commandPool_ = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> commandBuffers_;

    std::vector<VkSemaphore> imageAvailableSemaphores_;
    std::vector<VkSemaphore> renderFinishedSemaphores_;
    std::vector<VkFence> inFlightFences_;
    std::vector<VkFence> imagesInFlight_;
    std::size_t currentFrame_ = 0;

    VoxelWorld world_;
    WorldSettings worldSettings_{};
    mutable std::shared_mutex worldMutex_;
    Vec3 cameraPosition_{1.0f, 400.0f, 1.0f};
    Vec3 previousPhysicsCameraPosition_{1.0f, 400.0f, 1.0f};
    float cameraYaw_ = -90.0f;
    float cameraPitch_ = -35.0f;
    CameraViewMode cameraViewMode_ = CameraViewMode::FirstPerson;
    MovementMode movementMode_ = MovementMode::Fly;
    float verticalVelocity_ = 0.0f;
    bool playerGrounded_ = false;
    double lastMouseX_ = 0.0;
    double lastMouseY_ = 0.0;
    bool firstMouseSample_ = true;
    bool fullscreenEnabled_ = false;
    bool debugOverlayEnabled_ = true;
    bool entityColliderVisible_ = false;
    bool previousF3Pressed_ = false;
    bool previousCPressed_ = false;
    bool f3CommandExecutedWhileHeld_ = false;
    bool previousF2Pressed_ = false;
    bool previousF5Pressed_ = false;
    bool previousF11Pressed_ = false;
    bool previousSpacePressed_ = false;
    bool previousLeftMousePressed_ = false;
    bool previousRightMousePressed_ = false;
    bool moveForwardHeld_ = false;
    bool moveBackwardHeld_ = false;
    bool moveRightHeld_ = false;
    bool moveLeftHeld_ = false;
    bool moveUpHeld_ = false;
    bool moveDownHeld_ = false;
    bool jumpSuppressedUntilSpaceRelease_ = false;
    double lastSpaceTapTime_ = -1000.0;
    double physicsAccumulatorSeconds_ = 0.0;
    int windowedPosX_ = 100;
    int windowedPosY_ = 100;
    int windowedWidth_ = 1280;
    int windowedHeight_ = 720;
    std::uint64_t globalTick_ = 7200;

    std::uint32_t frameCounter_ = 0;
    std::uint32_t currentFps_ = 0;
    std::uint32_t drawCallCount_ = 0;
    double currentFrameTimeMs_ = 0.0;
    double fpsAccumulatorSeconds_ = 0.0;
    double overlayRefreshAccumulatorSeconds_ = 0.0;
    RuntimeProfileStage chunkLoadProfile_{};
    RuntimeProfileStage diskLoadProfile_{};
    RuntimeProfileStage generateProfile_{};
    RuntimeProfileStage meshBuildProfile_{};
    RuntimeProfileStage saveProfile_{};
    RuntimeProfileStage uploadProfile_{};
    RuntimeProfileStage uploadChunkProfile_{};
    RuntimeProfileStage uploadCountProfile_{};
    RuntimeProfileStage uploadRemoveProfile_{};
    RuntimeProfileStage uploadInsertProfile_{};
    RuntimeProfileStage uploadAllocProfile_{};
    RuntimeProfileStage uploadCopyProfile_{};
    RuntimeProfileStage drawCpuProfile_{};
    RuntimeProfileStage waitProfile_{};
    RuntimeProfileStage acquireProfile_{};
    RuntimeProfileStage submitProfile_{};
    RuntimeProfileStage presentProfile_{};
    RuntimeProfileStage frameProfile_{};
    RuntimeProfileStage worldMeshLockProfile_{};
    RuntimeProfileStage worldMeshSaveQueueProfile_{};
    RuntimeProfileStage worldMeshFinalizeProfile_{};
    RuntimeProfileStage worldMeshRenderDrainProfile_{};
    RuntimeProfileStage unloadProfile_{};
    RuntimeProfileStage unloadCountProfile_{};
    RuntimeProfileStage saveFileProfile_{};
    RuntimeProfileStage saveCountProfile_{};
    RuntimeProfileStage raycastWaitProfile_{};
    RuntimeProfileStage raycastProfile_{};
    RuntimeProfileStage collisionWaitProfile_{};
    RuntimeProfileStage collisionProfile_{};
    RuntimeProfileStage getBlockProfile_{};
    RuntimeProfileStage generatedBlockProfile_{};
    double chunkLoadProfileCumulativeMs_ = 0.0;
    std::uint64_t chunkLoadProfileCumulativeSamples_ = 0;
    double diskLoadProfileCumulativeMs_ = 0.0;
    std::uint64_t diskLoadProfileCumulativeSamples_ = 0;
    double generateProfileCumulativeMs_ = 0.0;
    std::uint64_t generateProfileCumulativeSamples_ = 0;
    double meshBuildProfileCumulativeMs_ = 0.0;
    std::uint64_t meshBuildProfileCumulativeSamples_ = 0;
    double saveProfileCumulativeMs_ = 0.0;
    std::uint64_t saveProfileCumulativeSamples_ = 0;
    double uploadProfileCumulativeMs_ = 0.0;
    std::uint64_t uploadProfileCumulativeSamples_ = 0;
    double uploadChunkProfileCumulativeMs_ = 0.0;
    std::uint64_t uploadChunkProfileCumulativeSamples_ = 0;
    double uploadCountProfileCumulativeValue_ = 0.0;
    std::uint64_t uploadCountProfileCumulativeSamples_ = 0;
    double uploadRemoveProfileCumulativeMs_ = 0.0;
    std::uint64_t uploadRemoveProfileCumulativeSamples_ = 0;
    double uploadInsertProfileCumulativeMs_ = 0.0;
    std::uint64_t uploadInsertProfileCumulativeSamples_ = 0;
    double uploadAllocProfileCumulativeMs_ = 0.0;
    std::uint64_t uploadAllocProfileCumulativeSamples_ = 0;
    double uploadCopyProfileCumulativeMs_ = 0.0;
    std::uint64_t uploadCopyProfileCumulativeSamples_ = 0;
    double drawCpuProfileCumulativeMs_ = 0.0;
    std::uint64_t drawCpuProfileCumulativeSamples_ = 0;
    double waitProfileCumulativeMs_ = 0.0;
    std::uint64_t waitProfileCumulativeSamples_ = 0;
    double acquireProfileCumulativeMs_ = 0.0;
    std::uint64_t acquireProfileCumulativeSamples_ = 0;
    double submitProfileCumulativeMs_ = 0.0;
    std::uint64_t submitProfileCumulativeSamples_ = 0;
    double presentProfileCumulativeMs_ = 0.0;
    std::uint64_t presentProfileCumulativeSamples_ = 0;
    double frameProfileCumulativeMs_ = 0.0;
    std::uint64_t frameProfileCumulativeSamples_ = 0;
    double worldMeshLockProfileCumulativeMs_ = 0.0;
    std::uint64_t worldMeshLockProfileCumulativeSamples_ = 0;
    double worldMeshSaveQueueProfileCumulativeMs_ = 0.0;
    std::uint64_t worldMeshSaveQueueProfileCumulativeSamples_ = 0;
    double worldMeshFinalizeProfileCumulativeMs_ = 0.0;
    std::uint64_t worldMeshFinalizeProfileCumulativeSamples_ = 0;
    double worldMeshRenderDrainProfileCumulativeMs_ = 0.0;
    std::uint64_t worldMeshRenderDrainProfileCumulativeSamples_ = 0;
    double unloadProfileCumulativeMs_ = 0.0;
    std::uint64_t unloadProfileCumulativeSamples_ = 0;
    double unloadCountProfileCumulativeValue_ = 0.0;
    std::uint64_t unloadCountProfileCumulativeSamples_ = 0;
    double saveFileProfileCumulativeMs_ = 0.0;
    std::uint64_t saveFileProfileCumulativeSamples_ = 0;
    double saveCountProfileCumulativeValue_ = 0.0;
    std::uint64_t saveCountProfileCumulativeSamples_ = 0;
    double raycastWaitProfileCumulativeMs_ = 0.0;
    std::uint64_t raycastWaitProfileCumulativeSamples_ = 0;
    double raycastProfileCumulativeMs_ = 0.0;
    std::uint64_t raycastProfileCumulativeSamples_ = 0;
    double collisionWaitProfileCumulativeMs_ = 0.0;
    std::uint64_t collisionWaitProfileCumulativeSamples_ = 0;
    double collisionProfileCumulativeMs_ = 0.0;
    std::uint64_t collisionProfileCumulativeSamples_ = 0;
    double getBlockProfileCumulativeMs_ = 0.0;
    std::uint64_t getBlockProfileCumulativeSamples_ = 0;
    double generatedBlockProfileCumulativeMs_ = 0.0;
    std::uint64_t generatedBlockProfileCumulativeSamples_ = 0;
    double uploadProfileTotalMs_ = 0.0;
    double uploadProfileMaxMs_ = 0.0;
    std::uint32_t uploadProfileSamples_ = 0;
    double uploadChunkProfileTotalMs_ = 0.0;
    double uploadChunkProfileMaxMs_ = 0.0;
    std::uint32_t uploadChunkProfileSamples_ = 0;
    double uploadCountProfileTotalValue_ = 0.0;
    double uploadCountProfileMaxValue_ = 0.0;
    std::uint32_t uploadCountProfileSamples_ = 0;
    double uploadRemoveProfileTotalMs_ = 0.0;
    double uploadRemoveProfileMaxMs_ = 0.0;
    std::uint32_t uploadRemoveProfileSamples_ = 0;
    double uploadInsertProfileTotalMs_ = 0.0;
    double uploadInsertProfileMaxMs_ = 0.0;
    std::uint32_t uploadInsertProfileSamples_ = 0;
    double uploadAllocProfileTotalMs_ = 0.0;
    double uploadAllocProfileMaxMs_ = 0.0;
    std::uint32_t uploadAllocProfileSamples_ = 0;
    double uploadCopyProfileTotalMs_ = 0.0;
    double uploadCopyProfileMaxMs_ = 0.0;
    std::uint32_t uploadCopyProfileSamples_ = 0;
    double drawCpuProfileTotalMs_ = 0.0;
    double drawCpuProfileMaxMs_ = 0.0;
    std::uint32_t drawCpuProfileSamples_ = 0;
    double waitProfileTotalMs_ = 0.0;
    double waitProfileMaxMs_ = 0.0;
    std::uint32_t waitProfileSamples_ = 0;
    double acquireProfileTotalMs_ = 0.0;
    double acquireProfileMaxMs_ = 0.0;
    std::uint32_t acquireProfileSamples_ = 0;
    double submitProfileTotalMs_ = 0.0;
    double submitProfileMaxMs_ = 0.0;
    std::uint32_t submitProfileSamples_ = 0;
    double presentProfileTotalMs_ = 0.0;
    double presentProfileMaxMs_ = 0.0;
    std::uint32_t presentProfileSamples_ = 0;
    double frameProfileTotalMs_ = 0.0;
    double frameProfileMaxMs_ = 0.0;
    std::uint32_t frameProfileSamples_ = 0;
    double worldMeshLockProfileTotalMs_ = 0.0;
    double worldMeshLockProfileMaxMs_ = 0.0;
    std::uint32_t worldMeshLockProfileSamples_ = 0;
    double worldMeshSaveQueueProfileTotalMs_ = 0.0;
    double worldMeshSaveQueueProfileMaxMs_ = 0.0;
    std::uint32_t worldMeshSaveQueueProfileSamples_ = 0;
    double worldMeshFinalizeProfileTotalMs_ = 0.0;
    double worldMeshFinalizeProfileMaxMs_ = 0.0;
    std::uint32_t worldMeshFinalizeProfileSamples_ = 0;
    double worldMeshRenderDrainProfileTotalMs_ = 0.0;
    double worldMeshRenderDrainProfileMaxMs_ = 0.0;
    std::uint32_t worldMeshRenderDrainProfileSamples_ = 0;
    double unloadProfileTotalMs_ = 0.0;
    double unloadProfileMaxMs_ = 0.0;
    std::uint32_t unloadProfileSamples_ = 0;
    double unloadCountProfileTotalValue_ = 0.0;
    double unloadCountProfileMaxValue_ = 0.0;
    std::uint32_t unloadCountProfileSamples_ = 0;
    double saveFileProfileTotalMs_ = 0.0;
    double saveFileProfileMaxMs_ = 0.0;
    std::uint32_t saveFileProfileSamples_ = 0;
    double saveCountProfileTotalValue_ = 0.0;
    double saveCountProfileMaxValue_ = 0.0;
    std::uint32_t saveCountProfileSamples_ = 0;
    double raycastWaitProfileTotalMs_ = 0.0;
    double raycastWaitProfileMaxMs_ = 0.0;
    std::uint32_t raycastWaitProfileSamples_ = 0;
    double raycastProfileTotalMs_ = 0.0;
    double raycastProfileMaxMs_ = 0.0;
    std::uint32_t raycastProfileSamples_ = 0;
    double collisionWaitProfileTotalMs_ = 0.0;
    double collisionWaitProfileMaxMs_ = 0.0;
    std::uint32_t collisionWaitProfileSamples_ = 0;
    double collisionProfileTotalMs_ = 0.0;
    double collisionProfileMaxMs_ = 0.0;
    std::uint32_t collisionProfileSamples_ = 0;
    double getBlockProfileTotalMs_ = 0.0;
    double getBlockProfileMaxMs_ = 0.0;
    std::uint32_t getBlockProfileSamples_ = 0;
    double generatedBlockProfileTotalMs_ = 0.0;
    double generatedBlockProfileMaxMs_ = 0.0;
    std::uint32_t generatedBlockProfileSamples_ = 0;
    double runtimeProfileElapsedSeconds_ = 0.0;
    std::atomic<bool> runtimeProfileCollectionEnabled_{false};
    std::mutex runtimeProfileMutex_;
    std::size_t loadedChunkCount_ = 0;
    int lastMeshChunkX_ = -1;
    int lastMeshChunkZ_ = -1;
    int lastMeshYawBucket_ = -1;
    int lastMeshPitchBucket_ = -1;
    std::uint64_t totalRamBytes_ = 0;
    std::uint64_t usedRamBytes_ = 0;
    std::uint64_t totalVramBytes_ = 0;
    std::optional<std::uint64_t> usedVramBytes_;
    std::string cpuName_;
    std::string gpuName_;
    std::string apiVersionString_;
    std::string driverVersionString_;
    std::string rendererName_ = "VULKAN";
    std::array<FontGlyphBitmap, 128> overlayFontGlyphs_{};
    int overlayFontLineHeight_ = 0;
    float crosshairU0_ = 0.0f;
    float crosshairV0_ = 0.0f;
    float crosshairU1_ = 0.0f;
    float crosshairV1_ = 0.0f;
    float crosshairWidth_ = 0.0f;
    float crosshairHeight_ = 0.0f;
    bool crosshairLoaded_ = false;
    float sunU0_ = 0.0f;
    float sunV0_ = 0.0f;
    float sunU1_ = 0.0f;
    float sunV1_ = 0.0f;
    float sunWidth_ = 0.0f;
    float sunHeight_ = 0.0f;
    bool sunLoaded_ = false;
    float moonU0_ = 0.0f;
    float moonV0_ = 0.0f;
    float moonU1_ = 0.0f;
    float moonV1_ = 0.0f;
    float moonWidth_ = 0.0f;
    float moonHeight_ = 0.0f;
    bool moonLoaded_ = false;
    std::vector<WorldVertex> playerBaseVertices_;
    std::vector<WorldVertex> playerRenderVertices_;
    std::vector<std::uint32_t> playerIndices_;
    Vec3 playerModelBoundsMin_{};
    Vec3 playerModelBoundsMax_{};
    bool playerLoaded_ = false;
    std::optional<BlockRaycastHit> selectedBlockHit_;
    double selectedBlockTraceAccumulatorSeconds_ = 1.0 / 60.0;
    bool screenshotRequested_ = false;
    bool screenshotCapturePending_ = false;
    VkBuffer screenshotStagingBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory screenshotStagingBufferMemory_ = VK_NULL_HANDLE;
    VkDeviceSize screenshotStagingBufferSize_ = 0;
    std::uint32_t screenshotWidth_ = 0;
    std::uint32_t screenshotHeight_ = 0;

    struct WorldMeshBuildRequest {
        int centerChunkX = 0;
        int centerChunkZ = 0;
        std::uint64_t serial = 0;
    };

    std::thread worldMeshWorkerThread_;
    std::thread worldSaveWorkerThread_;
    std::vector<std::thread> chunkLoadWorkerThreads_;
    std::vector<std::thread> meshWorkerThreads_;
    std::mutex worldMeshWorkerMutex_;
    std::condition_variable worldMeshWorkerCv_;
    bool worldMeshWorkerRunning_ = false;
    bool worldMeshTargetAvailable_ = false;
    bool worldMeshRequestPending_ = false;
    std::uint64_t nextWorldMeshRequestSerial_ = 0;
    std::uint64_t nextCompletedWorldMeshSerial_ = 0;
    std::uint64_t completedWorldMeshSerial_ = 0;
    std::uint64_t uploadedWorldMeshSerial_ = 0;
    WorldMeshBuildRequest pendingWorldMeshRequest_{};
    std::optional<WorldRenderUpdate> completedWorldRenderUpdate_;
    std::deque<PendingChunkId> pendingStorageChunkRequests_;
    std::unordered_set<PendingChunkId, PendingChunkIdHash> pendingStorageChunkRequestSet_;
    std::deque<PreparedChunkColumn> completedPreparedChunkColumns_;
    std::deque<RegionSaveTask> pendingWorldSaveTasks_;
    std::vector<ChunkMeshBatchData> pendingWorldRenderUploads_;
    std::vector<PendingChunkId> pendingWorldRenderRemovals_;
    std::unordered_set<PendingChunkId, PendingChunkIdHash> pendingWorldRenderUploadSet_;
    std::unordered_set<PendingChunkId, PendingChunkIdHash> pendingWorldRenderRemovalSet_;
    std::size_t pendingWorldRenderLoadedChunkCount_ = 0;
};
