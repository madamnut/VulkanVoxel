#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "MathTypes.h"
#include "VoxelWorld.h"

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <array>
#include <string>
#include <thread>
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
};

class VulkanVoxelApp {
public:
    int Run();

private:
    void InitWindow();
    void InitVulkan();
    void MainLoop();
    void Cleanup();

    void ProcessInput(float deltaTime);
    Vec3 GetForwardVector() const;
    Vec3 GetHorizontalForwardVector() const;
    Vec3 GetRightVector() const;
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
    void LoadOverlayFont();
    void BuildWorldMesh();
    void RequestWorldMeshBuild();
    void StartWorldMeshWorker();
    void StopWorldMeshWorker();
    void ConsumeCompletedWorldMesh();
    WorldMeshData BuildWorldMeshData(
        int centerChunkX,
        int centerChunkZ,
        const Vec3& cameraPosition,
        const Vec3& cameraForward,
        float aspectRatio
    );
    void UploadWorldMesh(const WorldMeshData& mesh);
    void DestroyWorldMeshBuffers();
    void CreateOverlayBuffer();
    void CreateUniformBuffers();
    void CreateDescriptorPool();
    void CreateDescriptorSets();
    void CreatePipelines();
    void CreateFramebuffers();
    void CreateCommandBuffers();
    void CreateSyncObjects();
    void DrawFrame();
    void UpdateUniformBuffer(std::uint32_t frameIndex);
    void UpdateOverlayText(float deltaTime);
    void RefreshSystemUsageStats();
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
        VkDeviceMemory& imageMemory
    ) const;
    VkImageView CreateImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags) const;
    void TransitionImageLayout(
        VkImage image,
        VkFormat format,
        VkImageLayout oldLayout,
        VkImageLayout newLayout
    ) const;
    void CopyBufferToImage(VkBuffer buffer, VkImage image, std::uint32_t width, std::uint32_t height) const;
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
    VkPipelineLayout overlayPipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline overlayPipeline_ = VK_NULL_HANDLE;

    VkImage depthImage_ = VK_NULL_HANDLE;
    VkDeviceMemory depthImageMemory_ = VK_NULL_HANDLE;
    VkImageView depthImageView_ = VK_NULL_HANDLE;
    VkFormat depthFormat_ = VK_FORMAT_UNDEFINED;

    VkImage textureImage_ = VK_NULL_HANDLE;
    VkDeviceMemory textureImageMemory_ = VK_NULL_HANDLE;
    VkImageView textureImageView_ = VK_NULL_HANDLE;
    VkSampler textureSampler_ = VK_NULL_HANDLE;
    VkImage overlayFontImage_ = VK_NULL_HANDLE;
    VkDeviceMemory overlayFontImageMemory_ = VK_NULL_HANDLE;
    VkImageView overlayFontImageView_ = VK_NULL_HANDLE;
    VkSampler overlayFontSampler_ = VK_NULL_HANDLE;

    VkBuffer worldVertexBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory worldVertexBufferMemory_ = VK_NULL_HANDLE;
    std::uint32_t worldVertexCount_ = 0;
    VkBuffer worldIndexBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory worldIndexBufferMemory_ = VK_NULL_HANDLE;
    std::uint32_t worldIndexCount_ = 0;

    VkBuffer overlayVertexBuffer_ = VK_NULL_HANDLE;
    VkDeviceMemory overlayVertexBufferMemory_ = VK_NULL_HANDLE;
    std::vector<OverlayVertex> overlayVertices_;
    std::uint32_t overlayVertexCount_ = 0;
    bool overlayDirty_ = false;

    std::vector<VkBuffer> uniformBuffers_;
    std::vector<VkDeviceMemory> uniformBuffersMemory_;
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> descriptorSets_;
    std::vector<VkDescriptorSet> overlayDescriptorSets_;

    std::vector<VkFramebuffer> swapChainFramebuffers_;

    VkCommandPool commandPool_ = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> commandBuffers_;

    std::vector<VkSemaphore> imageAvailableSemaphores_;
    std::vector<VkSemaphore> renderFinishedSemaphores_;
    std::vector<VkFence> inFlightFences_;
    std::vector<VkFence> imagesInFlight_;
    std::size_t currentFrame_ = 0;

    VoxelWorld world_;
    Vec3 cameraPosition_{256.0f, 290.0f, 340.0f};
    float cameraYaw_ = -90.0f;
    float cameraPitch_ = -35.0f;
    double lastMouseX_ = 0.0;
    double lastMouseY_ = 0.0;
    bool firstMouseSample_ = true;
    bool fullscreenEnabled_ = false;
    bool debugOverlayEnabled_ = true;
    bool previousF3Pressed_ = false;
    bool previousF11Pressed_ = false;
    int windowedPosX_ = 100;
    int windowedPosY_ = 100;
    int windowedWidth_ = 1280;
    int windowedHeight_ = 720;

    std::uint32_t frameCounter_ = 0;
    std::uint32_t currentFps_ = 0;
    std::uint32_t drawCallCount_ = 0;
    double currentFrameTimeMs_ = 0.0;
    double fpsAccumulatorSeconds_ = 0.0;
    double overlayRefreshAccumulatorSeconds_ = 0.0;
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
    std::string presentModeString_ = "FIFO";
    std::array<FontGlyphBitmap, 128> overlayFontGlyphs_{};
    int overlayFontLineHeight_ = 0;

    struct WorldMeshBuildRequest {
        int centerChunkX = 0;
        int centerChunkZ = 0;
        Vec3 cameraPosition{};
        Vec3 cameraForward{};
        float aspectRatio = 16.0f / 9.0f;
        std::uint64_t serial = 0;
    };

    std::thread worldMeshWorkerThread_;
    std::mutex worldMeshWorkerMutex_;
    std::condition_variable worldMeshWorkerCv_;
    bool worldMeshWorkerRunning_ = false;
    bool worldMeshRequestPending_ = false;
    std::uint64_t nextWorldMeshRequestSerial_ = 0;
    std::uint64_t completedWorldMeshSerial_ = 0;
    std::uint64_t uploadedWorldMeshSerial_ = 0;
    WorldMeshBuildRequest pendingWorldMeshRequest_{};
    std::optional<WorldMeshData> completedWorldMesh_;
};
