#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define NOMINMAX
#include <Windows.h>

#include "core/FileSystem.h"
#include "core/GameConfig.h"
#include "core/Logger.h"
#include "core/Math.h"
#include "core/SystemInfo.h"
#include "player/PlayerController.h"
#include "player/PlayerTypes.h"
#include "render/CrosshairRenderer.h"
#include "render/DebugTextRenderer.h"
#include "render/PlayerModel.h"
#include "render/VulkanDescriptors.h"
#include "render/VulkanHelpers.h"
#include "render/VulkanPipelineBuilder.h"
#include "render/TextureManager.h"
#include "render/VulkanResourceContext.h"
#include "render/VulkanSwapchain.h"
#include "world/Block.h"
#include "world/BlockRegistry.h"
#include "world/BlockRaycast.h"
#include "world/ChunkMesher.h"
#include "world/ChunkStreamingManager.h"
#include "world/WorldSave.h"
#include "world/WorldTime.h"
#include "world/WorldGenerator.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace
{
constexpr std::uint32_t kWindowWidth = 1280;
constexpr std::uint32_t kWindowHeight = 720;
constexpr int kMaxFramesInFlight = 2;
constexpr int kDefaultChunkLoadRadius = 5;
constexpr int kMaxChunkLoadRadius = 64;
constexpr int kDefaultChunkUploadsPerFrame = 1;
constexpr int kMaxChunkUploadsPerFrame = 64;
constexpr int kDefaultChunkBuildThreads = 4;
constexpr int kMaxChunkBuildThreads = 16;
constexpr float kDefaultInteractionDistance = 5.0f;
constexpr std::size_t kSelectionOutlineVertexCount = 24;
constexpr float kRenderFovY = 70.0f * kPi / 180.0f;
constexpr float kRenderNearPlane = 0.05f;
constexpr float kRenderFarPlane = 1200.0f;

struct QueueFamilyIndices
{
    std::optional<std::uint32_t> graphicsFamily;
    std::optional<std::uint32_t> presentFamily;

    bool isComplete() const
    {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

struct ChunkMesh
{
    Buffer vertexBuffer;
    Buffer indexBuffer;
    Buffer fluidVertexBuffer;
    Buffer fluidIndexBuffer;
    std::uint32_t vertexCount = 0;
    std::uint32_t indexCount = 0;
    std::uint32_t fluidVertexCount = 0;
    std::uint32_t fluidIndexCount = 0;
    int chunkX = 0;
    int chunkZ = 0;
    std::vector<SubchunkDraw> subchunks;
    std::vector<SubchunkDraw> fluidSubchunks;
};

struct FrustumPlane
{
    Vec3 normal{};
    float distance = 0.0f;
};

struct ViewFrustum
{
    std::array<FrustumPlane, 6> planes{};
};

struct DeferredChunkBufferDestroy
{
    Buffer vertexBuffer;
    Buffer indexBuffer;
    Buffer fluidVertexBuffer;
    Buffer fluidIndexBuffer;
    std::uint64_t retireFrame = 0;
};

struct DeferredUploadCleanup
{
    Buffer stagingBuffer;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    std::uint64_t retireFrame = 0;
};

struct SkyUniform
{
    alignas(16) float camera[4];      // yaw, pitch, aspect, tan(fovY / 2)
    alignas(16) float sunDirection[4];
    alignas(16) float moonDirection[4];
    alignas(16) float spriteScale[4]; // sun height in NDC, moon height in NDC
};

struct BlockUniform
{
    alignas(16) float viewProjection[16];
};

struct SelectionVertex
{
    float position[3];
};

struct DebugTextOverlay
{
    std::chrono::steady_clock::time_point lastUpdate = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point rightLastUpdate = std::chrono::steady_clock::now();
    std::uint32_t framesSinceUpdate = 0;
    std::vector<std::wstring> lines = {
        L"FPS: 0000 [00.000MS]",
        L"POS: X:0000.000 [0000.000] / Y:0000.000 [0000.000] / Z:0000.000 [0000.000]",
        L"Landform: 0.000 [0.000]",
        L"CAM: YAW +000.000 PIT +00.000",
        L"TIME: 000 DAY 00 H 00 M",
        L"SEED: 0",
    };
    std::vector<std::wstring> rightLines = {
        L"VULKANVOXEL 0.0.0.0",
        L"GPU: N/A",
        L"CPU: N/A",
        L"RAM: 00000 / 00000 MB",
        L"VRAM: N/A",
        L"API: VULKAN 0.0.0",
        L"DRIVER: 0",
        L"DRAWS: 0000",
        L"CHUNKS: 000",
        L"LOAD RADIUS: 0",
        L"UPLOADS/FRAME: 0",
        L"BUILD THREADS: 0",
        L"BUILD JOBS: 0",
        L"BUILD DONE: 0",
        L"DEFERRED DESTROYS: 0",
        L"DEFERRED UPLOADS: 0",
        L"VERTS: 0000000",
        L"INDICES: 0000000",
        L"TRIS: 0000000",
    };
    std::vector<std::wstring> bottomLeftLines;
};

struct FrameProfiler
{
    double frameCpuMs = 0.0;
    double fenceWaitMs = 0.0;
    double acquireMs = 0.0;
    double loadUpdateMs = 0.0;
    double subchunkDoneMs = 0.0;
    double chunkUploadMs = 0.0;
    double uniformMs = 0.0;
    double debugTextMs = 0.0;
    double recordMs = 0.0;
    double submitPresentMs = 0.0;
    std::size_t subchunkResultsProcessed = 0;
    int chunksUploaded = 0;
    std::size_t chunksLoaded = 0;
    std::size_t chunksUnloaded = 0;
    std::size_t chunksQueued = 0;
};

void glfwErrorCallback(int errorCode, const char* description)
{
    std::cerr << "GLFW error " << errorCode << ": " << description << '\n';
}

int floorDiv(int value, int divisor)
{
    int quotient = value / divisor;
    const int remainder = value % divisor;
    if (remainder != 0 && ((remainder < 0) != (divisor < 0)))
    {
        --quotient;
    }
    return quotient;
}

int floorMod(int value, int divisor)
{
    int result = value % divisor;
    if (result < 0)
    {
        result += divisor;
    }
    return result;
}

float wrapWorldPosition(float position)
{
    float result = std::fmod(position, static_cast<float>(kWorldSizeXZ));
    if (result < 0.0f)
    {
        result += static_cast<float>(kWorldSizeXZ);
    }
    return result;
}

int chunkCoordForWorldPosition(float position, int chunkSize)
{
    return floorDiv(static_cast<int>(std::floor(position)), chunkSize);
}

std::uint64_t mixSeed(std::uint64_t value)
{
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ull;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebull;
    value ^= value >> 31;
    return value;
}

std::uint64_t generateRandomWorldSeed()
{
    std::uint64_t seed = static_cast<std::uint64_t>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());
    try
    {
        std::random_device randomDevice;
        seed ^= static_cast<std::uint64_t>(randomDevice()) << 32;
        seed ^= static_cast<std::uint64_t>(randomDevice());
    }
    catch (...)
    {
    }
    return mixSeed(seed);
}

std::wstring formatCoordinateValue(double value)
{
    const bool negative = value < 0.0;
    const double absoluteValue = std::abs(value);
    std::wostringstream stream;
    if (negative)
    {
        stream << L'-';
    }
    stream << std::fixed << std::setprecision(3)
           << std::setw(8) << std::setfill(L'0') << absoluteValue;
    return stream.str();
}

const wchar_t* directionNameForYaw(float yaw)
{
    constexpr std::array<const wchar_t*, 4> directionNames = {
        L"EAST",
        L"SOUTH",
        L"WEST",
        L"NORTH",
    };
    float normalizedYaw = std::remainder(yaw, 2.0f * kPi);
    if (normalizedYaw < 0.0f)
    {
        normalizedYaw += 2.0f * kPi;
    }

    const int directionIndex = static_cast<int>(std::floor(
        (normalizedYaw + kPi / 4.0f) / (kPi / 2.0f))) % static_cast<int>(directionNames.size());
    return directionNames[static_cast<std::size_t>(directionIndex)];
}

class VulkanVoxelApp
{
public:
    void run()
    {
        initLogging();
        initSaveSystem();
        loadSavedWorldState();
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    GLFWwindow* window_ = nullptr;
    VkInstance instance_ = VK_NULL_HANDLE;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties physicalDeviceProperties_{};
    bool memoryBudgetSupported_ = false;
    VulkanResourceContext resourceContext_{};
    TextureManager textureManager_{};
    DebugTextRenderer debugTextRenderer_{};
    PlayerModel playerModel_{};
    std::wstring versionDebugLine_;
    std::wstring gpuDebugLine_;
    std::wstring cpuDebugLine_;
    std::wstring apiDebugLine_;
    std::wstring driverDebugLine_;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue graphicsQueue_ = VK_NULL_HANDLE;
    VkQueue presentQueue_ = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkFormat swapchainImageFormat_ = VK_FORMAT_UNDEFINED;
    VkExtent2D swapchainExtent_{};
    std::vector<VkImage> swapchainImages_;
    std::vector<VkImageView> swapchainImageViews_;
    std::vector<VkFramebuffer> framebuffers_;
    VkRenderPass renderPass_ = VK_NULL_HANDLE;
    Texture depthTexture_{};
    VkDescriptorSetLayout descriptorSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout blockDescriptorSetLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout debugTextDescriptorSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout blockPipelineLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout fluidPipelineLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout playerPipelineLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout selectionPipelineLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout debugTextPipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline graphicsPipeline_ = VK_NULL_HANDLE;
    VkPipeline blockPipeline_ = VK_NULL_HANDLE;
    VkPipeline fluidPipeline_ = VK_NULL_HANDLE;
    VkPipeline playerPipeline_ = VK_NULL_HANDLE;
    VkPipeline selectionPipeline_ = VK_NULL_HANDLE;
    VkPipeline debugTextPipeline_ = VK_NULL_HANDLE;
    VkCommandPool commandPool_ = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> commandBuffers_;
    std::vector<VkSemaphore> imageAvailableSemaphores_;
    std::vector<VkSemaphore> renderFinishedSemaphores_;
    std::vector<VkFence> inFlightFences_;
    std::size_t currentFrame_ = 0;
    std::uint64_t frameCounter_ = 0;
    bool framebufferResized_ = false;

    Texture sunTexture_{};
    Texture moonTexture_{};
    Texture blockTextureArray_{};
    Texture playerTexture_{};
    Texture crosshairTexture_{};
    Texture debugFontAtlasTexture_{};
    VkSampler textureSampler_ = VK_NULL_HANDLE;
    Buffer uniformBuffer_{};
    Buffer blockUniformBuffer_{};
    Buffer debugTextVertexBuffer_{};
    Buffer crosshairVertexBuffer_{};
    Buffer playerVertexBuffer_{};
    Buffer playerIndexBuffer_{};
    Buffer selectionVertexBuffer_{};
    void* uniformMappedMemory_ = nullptr;
    void* blockUniformMappedMemory_ = nullptr;
    void* debugTextVertexMappedMemory_ = nullptr;
    void* crosshairVertexMappedMemory_ = nullptr;
    void* playerVertexMappedMemory_ = nullptr;
    void* selectionVertexMappedMemory_ = nullptr;
    std::uint32_t debugTextVertexCount_ = 0;
    std::uint32_t crosshairVertexCount_ = 0;
    std::uint32_t playerVertexCount_ = 0;
    std::uint32_t playerIndexCount_ = 0;
    std::uint32_t selectionVertexCount_ = 0;
    std::vector<ChunkMesh> chunkMeshes_;
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet_ = VK_NULL_HANDLE;
    VkDescriptorSet blockDescriptorSet_ = VK_NULL_HANDLE;
    VkDescriptorSet playerDescriptorSet_ = VK_NULL_HANDLE;
    VkDescriptorSet crosshairDescriptorSet_ = VK_NULL_HANDLE;
    VkDescriptorSet debugTextDescriptorSet_ = VK_NULL_HANDLE;
    CameraState camera_{};
    PlayerController playerController_{};
    WorldGenerator worldGenerator_{};
    Logger logger_{};
    WorldSave worldSave_{};
    DebugTextOverlay debugTextOverlay_{};
    FrameProfiler frameProfiler_{};
    bool debugTextVisible_ = true;
    int chunkUploadsPerFrame_ = kDefaultChunkUploadsPerFrame;
    float interactionDistance_ = kDefaultInteractionDistance;
    ViewFrustum currentViewFrustum_{};
    std::size_t visibleSubchunkDraws_ = 0;
    std::size_t culledSubchunkDraws_ = 0;
    std::optional<BlockRaycastHit> selectedBlock_;
    std::chrono::steady_clock::time_point lastInteractionRaycastTime_ = std::chrono::steady_clock::now();
    BlockRegistry blockRegistry_{};
    std::vector<std::wstring> blockTextureLayerNames_;
    ChunkMesher chunkMesher_{worldGenerator_, blockRegistry_};
    ChunkStreamingManager chunkStreaming_{chunkMesher_};
    std::unordered_map<ChunkCoord, LoadedChunkData, ChunkCoordHash> loadedChunks_;
    mutable std::mutex loadedChunksMutex_;
    std::unordered_map<ChunkCoord, ChunkVoxelData, ChunkCoordHash> chunkDataCache_;
    std::unordered_set<ChunkCoord, ChunkCoordHash> chunkDataLoadsInProgress_;
    std::mutex chunkDataCacheMutex_;
    std::condition_variable chunkDataCacheCv_;
    std::mutex saveMutex_;
    std::vector<DeferredChunkBufferDestroy> deferredChunkBufferDestroys_;
    std::vector<DeferredUploadCleanup> deferredUploadCleanups_;
    std::size_t meshVertexCount_ = 0;
    std::size_t meshIndexCount_ = 0;
    std::uint64_t worldSeed_ = 0;
    std::uint64_t worldTimeTicks_ = kDefaultWorldTimeTicks;
    double worldTickAccumulator_ = 0.0;
    std::chrono::steady_clock::time_point lastFrameTime_ = std::chrono::steady_clock::now();
    bool cursorCaptured_ = true;
    bool fullscreen_ = false;
    int windowedX_ = 100;
    int windowedY_ = 100;
    int windowedWidth_ = kWindowWidth;
    int windowedHeight_ = kWindowHeight;

    static void framebufferResizeCallback(GLFWwindow* window, int, int)
    {
        auto* app = reinterpret_cast<VulkanVoxelApp*>(glfwGetWindowUserPointer(window));
        app->framebufferResized_ = true;
    }

    static void mouseCallback(GLFWwindow* window, double xPos, double yPos)
    {
        auto* app = reinterpret_cast<VulkanVoxelApp*>(glfwGetWindowUserPointer(window));
        app->handleMouseMove(xPos, yPos);
    }

    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int)
    {
        auto* app = reinterpret_cast<VulkanVoxelApp*>(glfwGetWindowUserPointer(window));
        if (action != GLFW_PRESS)
        {
            return;
        }

        if (button == GLFW_MOUSE_BUTTON_MIDDLE)
        {
            app->releaseCursor();
            return;
        }

        if (!app->cursorCaptured_)
        {
            app->captureCursor();
            return;
        }

        app->handleMouseButton(button);
    }

    static void keyCallback(GLFWwindow* window, int key, int, int action, int)
    {
        auto* app = reinterpret_cast<VulkanVoxelApp*>(glfwGetWindowUserPointer(window));
        if (key == GLFW_KEY_F11 && action == GLFW_PRESS)
        {
            app->toggleFullscreen();
        }
        else if (key == GLFW_KEY_F3 && action == GLFW_PRESS)
        {
            app->debugTextVisible_ = !app->debugTextVisible_;
        }
        else if (key == GLFW_KEY_F && action == GLFW_PRESS)
        {
            app->playerController_.toggleMovementMode();
        }
        else if (key == GLFW_KEY_F5 && action == GLFW_PRESS)
        {
            app->playerController_.cycleCameraViewMode();
        }
    }

    void handleMouseButton(int button)
    {
        if (button == GLFW_MOUSE_BUTTON_LEFT)
        {
            editSelectedBlock(kAirBlockId);
        }
        else if (button == GLFW_MOUSE_BUTTON_RIGHT)
        {
            editSelectedBlock(kRockBlockId);
        }
    }

    void initLogging()
    {
        logger_.initialize(sourcePathWide(L"/logs"));
    }

    void initSaveSystem()
    {
        worldSave_.initialize(sourcePathWide(L"/saves/world"), &logger_);
    }

    void loadSavedWorldState()
    {
        if (const std::optional<WorldSaveState> state = worldSave_.loadWorldState())
        {
            camera_ = state->camera;
            camera_.firstMouse = true;
            playerController_.setMovementMode(state->movementMode);
            playerController_.setCameraViewMode(state->cameraViewMode);
            worldTimeTicks_ = state->worldTimeTicks;
            worldSeed_ = state->worldSeed;
            worldGenerator_.setSeed(worldSeed_);
            return;
        }

        worldSeed_ = generateRandomWorldSeed();
        worldGenerator_.setSeed(worldSeed_);
        saveWorldState();
    }

    void initWindow()
    {
        glfwSetErrorCallback(glfwErrorCallback);
        if (glfwInit() != GLFW_TRUE)
        {
            throw std::runtime_error("Failed to initialize GLFW.");
        }

        if (glfwVulkanSupported() != GLFW_TRUE)
        {
            throw std::runtime_error("Vulkan is not supported by GLFW on this system.");
        }

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

        window_ = glfwCreateWindow(
            kWindowWidth,
            kWindowHeight,
            "VulkanVoxel " VULKAN_VOXEL_VERSION,
            nullptr,
            nullptr);
        if (window_ == nullptr)
        {
            throw std::runtime_error("Failed to create the game window.");
        }

        glfwSetWindowUserPointer(window_, this);
        glfwSetFramebufferSizeCallback(window_, framebufferResizeCallback);
        glfwSetCursorPosCallback(window_, mouseCallback);
        glfwSetMouseButtonCallback(window_, mouseButtonCallback);
        glfwSetKeyCallback(window_, keyCallback);
        glfwSetInputMode(window_, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    }

    void initVulkan()
    {
        createInstance();
        if (glfwCreateWindowSurface(instance_, window_, nullptr, &surface_) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create a Vulkan window surface.");
        }

        pickPhysicalDevice();
        createLogicalDevice();
        resourceContext_.setDevice(physicalDevice_, device_);
        createSwapchain();
        createRenderPass();
        createDepthResources();
        createDescriptorSetLayout();
        createGraphicsPipeline();
        createBlockPipeline();
        createFluidPipeline();
        createPlayerPipeline();
        createSelectionPipeline();
        createDebugTextPipeline();
        createFramebuffers();
        createCommandPool();
        resourceContext_.setTransferContext(commandPool_, graphicsQueue_);
        textureManager_.setContext(device_, resourceContext_);
        createSkyTextures();
        createCrosshairTexture();
        loadPlayerMesh();
        createPlayerTexture();
        loadBlockDefinitions();
        createBlockTextureArray();
        loadWorldConfig();
        chunkStreaming_.setChunkBuildCallback(
            [this](ChunkCoord coord, std::uint64_t generation)
            {
                return buildChunkForStreaming(coord, generation);
            });
        chunkStreaming_.setBuildErrorCallback(
            [this](ChunkCoord coord, const std::string& message)
            {
                logger_.error(
                    "Chunk build failed at (" +
                    std::to_string(coord.x) +
                    ", " +
                    std::to_string(coord.z) +
                    "): " +
                    message);
            });
        startChunkBuildWorkers();
        createChunkMesh();
        cacheStaticDebugText();
        updateRightDebugText();
        createDebugFontAtlasTexture();
        createTextureSampler();
        createUniformBuffer();
        createDescriptorPool();
        createDescriptorSet();
        createBlockDescriptorSet();
        createPlayerDescriptorSet();
        createCrosshairDescriptorSet();
        createDebugTextDescriptorSet();
        createCommandBuffers();
        createSyncObjects();
    }

    void mainLoop()
    {
        while (glfwWindowShouldClose(window_) != GLFW_TRUE)
        {
            glfwPollEvents();
            drawFrame();
        }

        vkDeviceWaitIdle(device_);
    }

    void cleanup()
    {
        stopChunkBuildWorkers();
        saveLoadedChunks(true);
        saveWorldState();
        if (device_ != VK_NULL_HANDLE)
        {
            vkDeviceWaitIdle(device_);
            collectDeferredUploadCleanups(true);
            collectDeferredChunkBufferDestroys(true);
        }

        cleanupSwapchain();

        resourceContext_.destroyBuffer(uniformBuffer_);
        resourceContext_.destroyBuffer(blockUniformBuffer_);
        resourceContext_.destroyBuffer(debugTextVertexBuffer_);
        resourceContext_.destroyBuffer(crosshairVertexBuffer_);
        resourceContext_.destroyBuffer(playerVertexBuffer_);
        resourceContext_.destroyBuffer(playerIndexBuffer_);
        resourceContext_.destroyBuffer(selectionVertexBuffer_);
        destroyAllChunkMeshes();

        if (descriptorPool_ != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorPool(device_, descriptorPool_, nullptr);
        }

        if (textureSampler_ != VK_NULL_HANDLE)
        {
            vkDestroySampler(device_, textureSampler_, nullptr);
        }

        resourceContext_.destroyTexture(sunTexture_);
        resourceContext_.destroyTexture(moonTexture_);
        resourceContext_.destroyTexture(blockTextureArray_);
        resourceContext_.destroyTexture(playerTexture_);
        resourceContext_.destroyTexture(crosshairTexture_);
        resourceContext_.destroyTexture(debugFontAtlasTexture_);
        RemoveFontResourceExW(sourcePathWide(L"/assets/fonts/VCR_OSD_MONO.ttf").c_str(), FR_PRIVATE, nullptr);

        if (descriptorSetLayout_ != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorSetLayout(device_, descriptorSetLayout_, nullptr);
        }
        if (blockDescriptorSetLayout_ != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorSetLayout(device_, blockDescriptorSetLayout_, nullptr);
        }
        if (debugTextDescriptorSetLayout_ != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorSetLayout(device_, debugTextDescriptorSetLayout_, nullptr);
        }

        for (std::size_t i = 0; i < kMaxFramesInFlight; ++i)
        {
            if (renderFinishedSemaphores_[i] != VK_NULL_HANDLE)
            {
                vkDestroySemaphore(device_, renderFinishedSemaphores_[i], nullptr);
            }
            if (imageAvailableSemaphores_[i] != VK_NULL_HANDLE)
            {
                vkDestroySemaphore(device_, imageAvailableSemaphores_[i], nullptr);
            }
            if (inFlightFences_[i] != VK_NULL_HANDLE)
            {
                vkDestroyFence(device_, inFlightFences_[i], nullptr);
            }
        }

        if (commandPool_ != VK_NULL_HANDLE)
        {
            vkDestroyCommandPool(device_, commandPool_, nullptr);
        }

        if (device_ != VK_NULL_HANDLE)
        {
            vkDestroyDevice(device_, nullptr);
        }

        if (surface_ != VK_NULL_HANDLE)
        {
            vkDestroySurfaceKHR(instance_, surface_, nullptr);
        }

        if (instance_ != VK_NULL_HANDLE)
        {
            vkDestroyInstance(instance_, nullptr);
        }

        if (window_ != nullptr)
        {
            glfwDestroyWindow(window_);
        }
        glfwTerminate();
    }

    void createInstance()
    {
        VkApplicationInfo applicationInfo{};
        applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        applicationInfo.pApplicationName = "VulkanVoxel";
        applicationInfo.applicationVersion = VK_MAKE_VERSION(0, 0, 0);
        applicationInfo.pEngineName = "VulkanVoxel";
        applicationInfo.engineVersion = VK_MAKE_VERSION(0, 0, 0);
        applicationInfo.apiVersion = VK_API_VERSION_1_2;

        std::uint32_t extensionCount = 0;
        const char** extensions = glfwGetRequiredInstanceExtensions(&extensionCount);
        if (extensions == nullptr || extensionCount == 0)
        {
            throw std::runtime_error("GLFW could not provide Vulkan instance extensions.");
        }

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &applicationInfo;
        createInfo.enabledExtensionCount = extensionCount;
        createInfo.ppEnabledExtensionNames = extensions;

        if (vkCreateInstance(&createInfo, nullptr, &instance_) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create Vulkan instance.");
        }
    }

    void pickPhysicalDevice()
    {
        std::uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance_, &deviceCount, nullptr);
        if (deviceCount == 0)
        {
            throw std::runtime_error("Failed to find a GPU with Vulkan support.");
        }

        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance_, &deviceCount, devices.data());

        for (VkPhysicalDevice device : devices)
        {
            if (isDeviceSuitable(device))
            {
                physicalDevice_ = device;
                vkGetPhysicalDeviceProperties(physicalDevice_, &physicalDeviceProperties_);
                return;
            }
        }

        throw std::runtime_error("Failed to find a suitable Vulkan GPU.");
    }

    bool isDeviceSuitable(VkPhysicalDevice device) const
    {
        QueueFamilyIndices indices = findQueueFamilies(device);
        bool extensionsSupported = checkDeviceExtensionSupport(device);
        bool swapchainAdequate = false;
        if (extensionsSupported)
        {
            SwapchainSupportDetails swapchainSupport = querySwapchainSupport(device, surface_);
            swapchainAdequate = !swapchainSupport.formats.empty() && !swapchainSupport.presentModes.empty();
        }

        return indices.isComplete() && extensionsSupported && swapchainAdequate;
    }

    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) const
    {
        QueueFamilyIndices indices{};

        std::uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

        for (std::uint32_t i = 0; i < queueFamilyCount; ++i)
        {
            if ((queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0)
            {
                indices.graphicsFamily = i;
            }

            VkBool32 presentSupport = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface_, &presentSupport);
            if (presentSupport == VK_TRUE)
            {
                indices.presentFamily = i;
            }

            if (indices.isComplete())
            {
                break;
            }
        }

        return indices;
    }

    bool checkDeviceExtensionSupport(VkPhysicalDevice device) const
    {
        std::uint32_t extensionCount = 0;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

        bool swapchainFound = false;
        for (const VkExtensionProperties& extension : availableExtensions)
        {
            if (std::strcmp(extension.extensionName, VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0)
            {
                swapchainFound = true;
                break;
            }
        }

        return swapchainFound;
    }

    bool isDeviceExtensionAvailable(VkPhysicalDevice device, const char* extensionName) const
    {
        std::uint32_t extensionCount = 0;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

        for (const VkExtensionProperties& extension : availableExtensions)
        {
            if (std::strcmp(extension.extensionName, extensionName) == 0)
            {
                return true;
            }
        }

        return false;
    }

    void createLogicalDevice()
    {
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice_);
        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::array<std::uint32_t, 2> uniqueFamilies = {
            indices.graphicsFamily.value(),
            indices.presentFamily.value(),
        };

        float queuePriority = 1.0f;
        for (std::uint32_t queueFamily : uniqueFamilies)
        {
            if (!queueCreateInfos.empty() && queueCreateInfos.front().queueFamilyIndex == queueFamily)
            {
                continue;
            }

            VkDeviceQueueCreateInfo queueCreateInfo{};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);
        }

        VkPhysicalDeviceFeatures deviceFeatures{};
        std::vector<const char*> deviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
        memoryBudgetSupported_ = isDeviceExtensionAvailable(physicalDevice_, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
        if (memoryBudgetSupported_)
        {
            deviceExtensions.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
        }

        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.queueCreateInfoCount = static_cast<std::uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();
        createInfo.pEnabledFeatures = &deviceFeatures;
        createInfo.enabledExtensionCount = static_cast<std::uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

        if (vkCreateDevice(physicalDevice_, &createInfo, nullptr, &device_) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create Vulkan logical device.");
        }

        vkGetDeviceQueue(device_, indices.graphicsFamily.value(), 0, &graphicsQueue_);
        vkGetDeviceQueue(device_, indices.presentFamily.value(), 0, &presentQueue_);
    }

    void createSwapchain()
    {
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice_);
        int width = 0;
        int height = 0;
        glfwGetFramebufferSize(window_, &width, &height);

        VulkanSwapchainResources resources = createVulkanSwapchain({
            physicalDevice_,
            device_,
            surface_,
            indices.graphicsFamily.value(),
            indices.presentFamily.value(),
            static_cast<std::uint32_t>(width),
            static_cast<std::uint32_t>(height),
        });

        swapchain_ = resources.swapchain;
        swapchainImageFormat_ = resources.imageFormat;
        swapchainExtent_ = resources.extent;
        swapchainImages_ = std::move(resources.images);
        swapchainImageViews_ = std::move(resources.imageViews);
    }

    void createRenderPass()
    {
        VkAttachmentDescription colorAttachment{};
        colorAttachment.format = swapchainImageFormat_;
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentReference colorAttachmentReference{};
        colorAttachmentReference.attachment = 0;
        colorAttachmentReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentDescription depthAttachment{};
        depthAttachment.format = findDepthFormat();
        depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference depthAttachmentReference{};
        depthAttachmentReference.attachment = 1;
        depthAttachmentReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentReference;
        subpass.pDepthStencilAttachment = &depthAttachmentReference;

        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask =
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.dstStageMask =
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.dstAccessMask =
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        std::array<VkAttachmentDescription, 2> attachments = {
            colorAttachment,
            depthAttachment,
        };

        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = static_cast<std::uint32_t>(attachments.size());
        renderPassInfo.pAttachments = attachments.data();
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;

        if (vkCreateRenderPass(device_, &renderPassInfo, nullptr, &renderPass_) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create Vulkan render pass.");
        }
    }

    VkFormat findSupportedFormat(
        const std::vector<VkFormat>& candidates,
        VkImageTiling tiling,
        VkFormatFeatureFlags features) const
    {
        for (VkFormat format : candidates)
        {
            VkFormatProperties properties{};
            vkGetPhysicalDeviceFormatProperties(physicalDevice_, format, &properties);

            if (tiling == VK_IMAGE_TILING_LINEAR &&
                (properties.linearTilingFeatures & features) == features)
            {
                return format;
            }

            if (tiling == VK_IMAGE_TILING_OPTIMAL &&
                (properties.optimalTilingFeatures & features) == features)
            {
                return format;
            }
        }

        throw std::runtime_error("Failed to find a supported Vulkan format.");
    }

    VkFormat findDepthFormat() const
    {
        return findSupportedFormat(
            {
                VK_FORMAT_D32_SFLOAT,
                VK_FORMAT_D32_SFLOAT_S8_UINT,
                VK_FORMAT_D24_UNORM_S8_UINT,
            },
            VK_IMAGE_TILING_OPTIMAL,
            VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
    }

    void createDepthResources()
    {
        depthTexture_.format = findDepthFormat();
        depthTexture_.width = swapchainExtent_.width;
        depthTexture_.height = swapchainExtent_.height;

        resourceContext_.createImage(
            depthTexture_.width,
            depthTexture_.height,
            depthTexture_.format,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            depthTexture_.image,
            depthTexture_.memory);

        depthTexture_.view = createVulkanImageView(
            device_,
            depthTexture_.image,
            depthTexture_.format,
            VK_IMAGE_ASPECT_DEPTH_BIT);
        depthTexture_.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    }

    void createDescriptorSetLayout()
    {
        const std::array<VkDescriptorSetLayoutBinding, 3> bindings = {
            descriptorSetLayoutBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT),
            descriptorSetLayoutBinding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT),
            descriptorSetLayoutBinding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT),
        };
        descriptorSetLayout_ = createVulkanDescriptorSetLayout(
            device_,
            bindings,
            "Failed to create Vulkan descriptor set layout.");

        const std::array<VkDescriptorSetLayoutBinding, 2> blockBindings = {
            descriptorSetLayoutBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT),
            descriptorSetLayoutBinding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT),
        };
        blockDescriptorSetLayout_ = createVulkanDescriptorSetLayout(
            device_,
            blockBindings,
            "Failed to create Vulkan block descriptor set layout.");

        const std::array<VkDescriptorSetLayoutBinding, 1> debugTextBindings = {
            descriptorSetLayoutBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT),
        };
        debugTextDescriptorSetLayout_ = createVulkanDescriptorSetLayout(
            device_,
            debugTextBindings,
            "Failed to create Vulkan debug text descriptor set layout.");
    }

    void createGraphicsPipeline()
    {
        const std::string shaderDir = std::string(VULKAN_VOXEL_BINARY_DIR) + "/shaders/";
        GraphicsPipelineConfig config{};
        config.vertexShaderPath = shaderDir + "sky.vert.spv";
        config.fragmentShaderPath = shaderDir + "sky.frag.spv";
        config.extent = swapchainExtent_;
        config.renderPass = renderPass_;
        config.descriptorSetLayout = descriptorSetLayout_;

        const GraphicsPipelineBundle bundle = createGraphicsPipelineBundle(device_, config);
        pipelineLayout_ = bundle.layout;
        graphicsPipeline_ = bundle.pipeline;
    }

    void createBlockPipeline()
    {
        const std::string shaderDir = std::string(VULKAN_VOXEL_BINARY_DIR) + "/shaders/";

        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(BlockVertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        std::array<VkVertexInputAttributeDescription, 4> attributeDescriptions{};
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(BlockVertex, position);
        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(BlockVertex, uv);
        attributeDescriptions[2].binding = 0;
        attributeDescriptions[2].location = 2;
        attributeDescriptions[2].format = VK_FORMAT_R32_SFLOAT;
        attributeDescriptions[2].offset = offsetof(BlockVertex, ao);
        attributeDescriptions[3].binding = 0;
        attributeDescriptions[3].location = 3;
        attributeDescriptions[3].format = VK_FORMAT_R32_SFLOAT;
        attributeDescriptions[3].offset = offsetof(BlockVertex, textureLayer);

        GraphicsPipelineConfig config{};
        config.vertexShaderPath = shaderDir + "block.vert.spv";
        config.fragmentShaderPath = shaderDir + "block.frag.spv";
        config.extent = swapchainExtent_;
        config.renderPass = renderPass_;
        config.descriptorSetLayout = blockDescriptorSetLayout_;
        config.cullMode = VK_CULL_MODE_BACK_BIT;
        config.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        config.depthTestEnable = true;
        config.depthWriteEnable = true;
        config.alphaBlendEnable = false;
        config.vertexBindingDescription = &bindingDescription;
        config.vertexAttributeDescriptions = attributeDescriptions.data();
        config.vertexAttributeDescriptionCount = static_cast<std::uint32_t>(attributeDescriptions.size());
        config.layoutErrorMessage = "Failed to create Vulkan block pipeline layout.";
        config.pipelineErrorMessage = "Failed to create Vulkan block graphics pipeline.";

        const GraphicsPipelineBundle bundle = createGraphicsPipelineBundle(device_, config);
        blockPipelineLayout_ = bundle.layout;
        blockPipeline_ = bundle.pipeline;
    }

    void createFluidPipeline()
    {
        const std::string shaderDir = std::string(VULKAN_VOXEL_BINARY_DIR) + "/shaders/";

        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(BlockVertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        std::array<VkVertexInputAttributeDescription, 4> attributeDescriptions{};
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(BlockVertex, position);
        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(BlockVertex, uv);
        attributeDescriptions[2].binding = 0;
        attributeDescriptions[2].location = 2;
        attributeDescriptions[2].format = VK_FORMAT_R32_SFLOAT;
        attributeDescriptions[2].offset = offsetof(BlockVertex, ao);
        attributeDescriptions[3].binding = 0;
        attributeDescriptions[3].location = 3;
        attributeDescriptions[3].format = VK_FORMAT_R32_SFLOAT;
        attributeDescriptions[3].offset = offsetof(BlockVertex, textureLayer);

        GraphicsPipelineConfig config{};
        config.vertexShaderPath = shaderDir + "block.vert.spv";
        config.fragmentShaderPath = shaderDir + "block.frag.spv";
        config.extent = swapchainExtent_;
        config.renderPass = renderPass_;
        config.descriptorSetLayout = blockDescriptorSetLayout_;
        config.cullMode = VK_CULL_MODE_NONE;
        config.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        config.depthTestEnable = true;
        config.depthWriteEnable = false;
        config.alphaBlendEnable = true;
        config.vertexBindingDescription = &bindingDescription;
        config.vertexAttributeDescriptions = attributeDescriptions.data();
        config.vertexAttributeDescriptionCount = static_cast<std::uint32_t>(attributeDescriptions.size());
        config.layoutErrorMessage = "Failed to create Vulkan fluid pipeline layout.";
        config.pipelineErrorMessage = "Failed to create Vulkan fluid graphics pipeline.";

        const GraphicsPipelineBundle bundle = createGraphicsPipelineBundle(device_, config);
        fluidPipelineLayout_ = bundle.layout;
        fluidPipeline_ = bundle.pipeline;
    }

    void createPlayerPipeline()
    {
        const std::string shaderDir = std::string(VULKAN_VOXEL_BINARY_DIR) + "/shaders/";

        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(BlockVertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions{};
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(BlockVertex, position);
        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(BlockVertex, uv);
        attributeDescriptions[2].binding = 0;
        attributeDescriptions[2].location = 2;
        attributeDescriptions[2].format = VK_FORMAT_R32_SFLOAT;
        attributeDescriptions[2].offset = offsetof(BlockVertex, ao);

        GraphicsPipelineConfig config{};
        config.vertexShaderPath = shaderDir + "player.vert.spv";
        config.fragmentShaderPath = shaderDir + "player.frag.spv";
        config.extent = swapchainExtent_;
        config.renderPass = renderPass_;
        config.descriptorSetLayout = blockDescriptorSetLayout_;
        config.cullMode = VK_CULL_MODE_BACK_BIT;
        config.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        config.depthTestEnable = true;
        config.depthWriteEnable = true;
        config.vertexBindingDescription = &bindingDescription;
        config.vertexAttributeDescriptions = attributeDescriptions.data();
        config.vertexAttributeDescriptionCount = static_cast<std::uint32_t>(attributeDescriptions.size());
        config.layoutErrorMessage = "Failed to create Vulkan player pipeline layout.";
        config.pipelineErrorMessage = "Failed to create Vulkan player graphics pipeline.";

        const GraphicsPipelineBundle bundle = createGraphicsPipelineBundle(device_, config);
        playerPipelineLayout_ = bundle.layout;
        playerPipeline_ = bundle.pipeline;
    }

    void createSelectionPipeline()
    {
        const std::string shaderDir = std::string(VULKAN_VOXEL_BINARY_DIR) + "/shaders/";

        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(SelectionVertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        std::array<VkVertexInputAttributeDescription, 1> attributeDescriptions{};
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(SelectionVertex, position);

        GraphicsPipelineConfig config{};
        config.vertexShaderPath = shaderDir + "selection.vert.spv";
        config.fragmentShaderPath = shaderDir + "selection.frag.spv";
        config.extent = swapchainExtent_;
        config.renderPass = renderPass_;
        config.descriptorSetLayout = blockDescriptorSetLayout_;
        config.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
        config.depthTestEnable = true;
        config.depthWriteEnable = false;
        config.alphaBlendEnable = false;
        config.vertexBindingDescription = &bindingDescription;
        config.vertexAttributeDescriptions = attributeDescriptions.data();
        config.vertexAttributeDescriptionCount = static_cast<std::uint32_t>(attributeDescriptions.size());
        config.layoutErrorMessage = "Failed to create Vulkan selection pipeline layout.";
        config.pipelineErrorMessage = "Failed to create Vulkan selection graphics pipeline.";

        const GraphicsPipelineBundle bundle = createGraphicsPipelineBundle(device_, config);
        selectionPipelineLayout_ = bundle.layout;
        selectionPipeline_ = bundle.pipeline;
    }

    void createDebugTextPipeline()
    {
        const std::string shaderDir = std::string(VULKAN_VOXEL_BINARY_DIR) + "/shaders/";

        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(DebugTextVertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions{};
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(DebugTextVertex, position);
        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(DebugTextVertex, uv);
        attributeDescriptions[2].binding = 0;
        attributeDescriptions[2].location = 2;
        attributeDescriptions[2].format = VK_FORMAT_R32G32B32A32_SFLOAT;
        attributeDescriptions[2].offset = offsetof(DebugTextVertex, color);

        GraphicsPipelineConfig config{};
        config.vertexShaderPath = shaderDir + "debug_text.vert.spv";
        config.fragmentShaderPath = shaderDir + "debug_text.frag.spv";
        config.extent = swapchainExtent_;
        config.renderPass = renderPass_;
        config.descriptorSetLayout = debugTextDescriptorSetLayout_;
        config.vertexBindingDescription = &bindingDescription;
        config.vertexAttributeDescriptions = attributeDescriptions.data();
        config.vertexAttributeDescriptionCount = static_cast<std::uint32_t>(attributeDescriptions.size());
        config.layoutErrorMessage = "Failed to create Vulkan debug text pipeline layout.";
        config.pipelineErrorMessage = "Failed to create Vulkan debug text graphics pipeline.";

        const GraphicsPipelineBundle bundle = createGraphicsPipelineBundle(device_, config);
        debugTextPipelineLayout_ = bundle.layout;
        debugTextPipeline_ = bundle.pipeline;
    }

    void createFramebuffers()
    {
        framebuffers_.resize(swapchainImageViews_.size());
        for (std::size_t i = 0; i < swapchainImageViews_.size(); ++i)
        {
            std::array<VkImageView, 2> attachments = {
                swapchainImageViews_[i],
                depthTexture_.view,
            };

            VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass_;
            framebufferInfo.attachmentCount = static_cast<std::uint32_t>(attachments.size());
            framebufferInfo.pAttachments = attachments.data();
            framebufferInfo.width = swapchainExtent_.width;
            framebufferInfo.height = swapchainExtent_.height;
            framebufferInfo.layers = 1;

            if (vkCreateFramebuffer(device_, &framebufferInfo, nullptr, &framebuffers_[i]) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create Vulkan framebuffer.");
            }
        }
    }

    void createCommandPool()
    {
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice_);

        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

        if (vkCreateCommandPool(device_, &poolInfo, nullptr, &commandPool_) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create Vulkan command pool.");
        }
    }

    void createSkyTextures()
    {
        sunTexture_ = createTexture(sourcePathWide(L"/assets/textures/sky/Sun.png"));
        moonTexture_ = createTexture(sourcePathWide(L"/assets/textures/sky/Moon.png"));
    }

    void createCrosshairTexture()
    {
        crosshairTexture_ = createTexture(sourcePathWide(L"/assets/textures/ui/Crosshair.png"));
    }

    void createPlayerTexture()
    {
        playerTexture_ = createTexture(sourcePathWide(L"/assets/textures/character/Character.png"));
    }

    void loadPlayerMesh()
    {
        playerModel_.loadFromFile(sourcePath("/assets/textures/character/Character.mesh"));
        playerIndexCount_ = static_cast<std::uint32_t>(playerModel_.indexCount());
    }

    void loadBlockDefinitions()
    {
        blockTextureLayerNames_.clear();
        blockRegistry_.loadFromConfig(sourcePath("/config/blocks.json"));
    }

    std::uint32_t addBlockTextureLayer(const std::wstring& relativePath)
    {
        for (std::uint32_t i = 0; i < blockTextureLayerNames_.size(); ++i)
        {
            if (blockTextureLayerNames_[i] == relativePath)
            {
                return i;
            }
        }

        blockTextureLayerNames_.push_back(relativePath);
        return static_cast<std::uint32_t>(blockTextureLayerNames_.size() - 1);
    }

    std::uint32_t resolveBlockTextureLayer(const std::string& name, const wchar_t* postfix)
    {
        const std::wstring baseRelativePath = L"/assets/textures/block/" + asciiToWide(name) + L".png";
        const std::wstring specificRelativePath =
            L"/assets/textures/block/" + asciiToWide(name) + postfix + L".png";
        std::wstring selectedRelativePath =
            fileExists(sourcePathWide(specificRelativePath)) ? specificRelativePath : baseRelativePath;
        const std::wstring postfixText(postfix);
        if (!fileExists(sourcePathWide(selectedRelativePath)) &&
            (postfixText == L"_top" || postfixText == L"_bottom"))
        {
            const std::wstring topBottomRelativePath =
                L"/assets/textures/block/" + asciiToWide(name) + L"_topbottom.png";
            if (fileExists(sourcePathWide(topBottomRelativePath)))
            {
                selectedRelativePath = topBottomRelativePath;
            }
        }
        if (!fileExists(sourcePathWide(selectedRelativePath)))
        {
            throw std::runtime_error("Missing block texture.");
        }

        return addBlockTextureLayer(selectedRelativePath);
    }

    void createBlockTextureArray()
    {
        for (BlockDefinition& definition : blockRegistry_.mutableDefinitions())
        {
            if (definition.renderShape == BlockRenderShape::None || definition.id == kAirBlockId)
            {
                continue;
            }

            definition.textureLayers[static_cast<std::size_t>(BlockFace::Top)] =
                resolveBlockTextureLayer(definition.name, L"_top");
            definition.textureLayers[static_cast<std::size_t>(BlockFace::Side)] =
                resolveBlockTextureLayer(definition.name, L"_side");
            definition.textureLayers[static_cast<std::size_t>(BlockFace::Bottom)] =
                resolveBlockTextureLayer(definition.name, L"_bottom");
        }
        const std::wstring waterTexturePath = L"/assets/textures/fluid/water.png";
        if (!fileExists(sourcePathWide(waterTexturePath)))
        {
            throw std::runtime_error("Missing water fluid texture.");
        }
        chunkMesher_.setWaterTextureLayer(addBlockTextureLayer(waterTexturePath));

        std::vector<std::wstring> texturePaths;
        texturePaths.reserve(blockTextureLayerNames_.size());
        for (const std::wstring& relativePath : blockTextureLayerNames_)
        {
            texturePaths.push_back(sourcePathWide(relativePath));
        }
        if (texturePaths.empty())
        {
            throw std::runtime_error("No block textures were loaded.");
        }

        blockTextureArray_ = finalizeTextureUpload(
            textureManager_.createTextureArrayFromFiles(texturePaths, VK_FORMAT_R8G8B8A8_SRGB));
    }

    void loadWorldConfig()
    {
        const TerrainDensityConfig defaultTerrainDensity{};
        const TerrainDensityConfig minTerrainDensity{
            {0.0f, 0.0f},
            {0, 0, 1.0f, 0.0f, 0.0f, 0.0f},
            {false, 0},
        };
        const TerrainDensityConfig maxTerrainDensity{
            {1.0f, static_cast<float>(kChunkHeight) * 4.0f},
            {kMaxTerrainDensityOctaves, 1024, 8.0f, 256.0f, 1.0f, 4.0f},
            {true, 1024},
        };
        const WorldConfig config = loadWorldConfigFile(
            sourcePath("/config/world.json"),
            {kDefaultChunkLoadRadius, kDefaultChunkUploadsPerFrame, kDefaultChunkBuildThreads, defaultTerrainDensity},
            {0, 1, 1, minTerrainDensity},
            {kMaxChunkLoadRadius, kMaxChunkUploadsPerFrame, kMaxChunkBuildThreads, maxTerrainDensity});

        chunkUploadsPerFrame_ = config.chunkUploadsPerFrame;
        chunkStreaming_.setLoadRadius(config.chunkLoadRadius);
        chunkStreaming_.setBuildThreadCount(config.chunkBuildThreads);
        worldGenerator_.setTerrainDensityConfig(config.terrainDensity);
        const std::string landformCurvePath = sourcePath("/assets/worldgen/landform_curve.bin");
        if (!worldGenerator_.loadLandformCurveFile(landformCurvePath))
        {
            logger_.error("Failed to load landform curve binary: " + landformCurvePath);
            throw std::runtime_error("Failed to load landform curve binary.");
        }
    }

    void createChunkMesh()
    {
        static_assert(kAirBlockId == 0 && kBlockIdCount == 65536);
        updateLoadedChunks();
    }

    void startChunkBuildWorkers()
    {
        chunkStreaming_.startWorkers();
    }

    void stopChunkBuildWorkers()
    {
        chunkStreaming_.stopWorkers();
    }

    ChunkBuildResult buildChunkForStreaming(ChunkCoord coord, std::uint64_t generation)
    {
        std::array<ChunkVoxelData, 9> neighborVoxels;
        for (int dz = -1; dz <= 1; ++dz)
        {
            for (int dx = -1; dx <= 1; ++dx)
            {
                neighborVoxels[static_cast<std::size_t>((dz + 1) * 3 + (dx + 1))] =
                    ensureChunkData({coord.x + dx, coord.z + dz});
            }
        }

        GeneratedChunkColumn column = buildMeshingColumnFromChunkData(coord, neighborVoxels);
        ChunkBuildResult result = chunkMesher_.buildChunkMeshFromColumn(coord, generation, column);

        ChunkVoxelData& centerVoxels = neighborVoxels[4];
        result.blockIds = std::move(centerVoxels.blockIds);
        result.fluidIds = std::move(centerVoxels.fluidIds);
        result.fluidAmounts = std::move(centerVoxels.fluidAmounts);
        return result;
    }

    ChunkVoxelData ensureChunkData(ChunkCoord coord)
    {
        while (true)
        {
            {
                std::lock_guard<std::mutex> lock(loadedChunksMutex_);
                const auto loadedIt = loadedChunks_.find(coord);
                if (loadedIt != loadedChunks_.end() &&
                    loadedIt->second.blockIds.size() == kChunkBlockCount &&
                    loadedIt->second.fluidIds.size() == kChunkBlockCount &&
                    loadedIt->second.fluidAmounts.size() == kChunkBlockCount)
                {
                    return {
                        loadedIt->second.blockIds,
                        loadedIt->second.fluidIds,
                        loadedIt->second.fluidAmounts,
                    };
                }
            }

            {
                std::unique_lock<std::mutex> lock(chunkDataCacheMutex_);
                const auto cacheIt = chunkDataCache_.find(coord);
                if (cacheIt != chunkDataCache_.end() && cacheIt->second.valid())
                {
                    return cacheIt->second;
                }

                if (!chunkDataLoadsInProgress_.contains(coord))
                {
                    chunkDataLoadsInProgress_.insert(coord);
                    break;
                }

                chunkDataCacheCv_.wait(lock, [&]()
                {
                    return !chunkDataLoadsInProgress_.contains(coord) || chunkDataCache_.contains(coord);
                });
            }
        }

        ChunkVoxelData voxels;
        try
        {
            voxels = loadOrGenerateChunkData(coord);
        }
        catch (...)
        {
            {
                std::lock_guard<std::mutex> lock(chunkDataCacheMutex_);
                chunkDataLoadsInProgress_.erase(coord);
            }
            chunkDataCacheCv_.notify_all();
            throw;
        }

        if (!voxels.valid())
        {
            {
                std::lock_guard<std::mutex> lock(chunkDataCacheMutex_);
                chunkDataLoadsInProgress_.erase(coord);
            }
            chunkDataCacheCv_.notify_all();
            throw std::runtime_error("Chunk data load/generation produced invalid voxel data.");
        }

        {
            std::lock_guard<std::mutex> lock(loadedChunksMutex_);
            const auto loadedIt = loadedChunks_.find(coord);
            if (loadedIt != loadedChunks_.end() &&
                loadedIt->second.blockIds.size() == kChunkBlockCount &&
                loadedIt->second.fluidIds.size() == kChunkBlockCount &&
                loadedIt->second.fluidAmounts.size() == kChunkBlockCount)
            {
                voxels = {
                    loadedIt->second.blockIds,
                    loadedIt->second.fluidIds,
                    loadedIt->second.fluidAmounts,
                };
            }
        }

        {
            std::lock_guard<std::mutex> lock(chunkDataCacheMutex_);
            chunkDataCache_[coord] = voxels;
            chunkDataLoadsInProgress_.erase(coord);
        }
        chunkDataCacheCv_.notify_all();
        return voxels;
    }

    ChunkVoxelData loadOrGenerateChunkData(ChunkCoord coord)
    {
        {
            std::lock_guard<std::mutex> lock(loadedChunksMutex_);
            const auto loadedIt = loadedChunks_.find(coord);
            if (loadedIt != loadedChunks_.end() &&
                loadedIt->second.blockIds.size() == kChunkBlockCount &&
                loadedIt->second.fluidIds.size() == kChunkBlockCount &&
                loadedIt->second.fluidAmounts.size() == kChunkBlockCount)
            {
                return {
                    loadedIt->second.blockIds,
                    loadedIt->second.fluidIds,
                    loadedIt->second.fluidAmounts,
                };
            }
        }

        if (std::optional<ChunkVoxelData> savedVoxels = loadChunkFromDisk(coord))
        {
            if (savedVoxels->valid())
            {
                return std::move(*savedVoxels);
            }
        }

        ChunkVoxelData generatedVoxels = worldGenerator_.generateChunkVoxels(coord);
        saveChunkToDisk(coord, generatedVoxels);
        return generatedVoxels;
    }

    GeneratedChunkColumn buildMeshingColumnFromChunkData(
        ChunkCoord coord,
        const std::array<ChunkVoxelData, 9>& neighborVoxels) const
    {
        const int chunkBaseX = coord.x * kChunkSizeX;
        const int chunkBaseZ = coord.z * kChunkSizeZ;
        GeneratedChunkColumn column{};

        for (int localPaddedZ = 0; localPaddedZ < GeneratedChunkColumn::kDepth; ++localPaddedZ)
        {
            for (int localPaddedX = 0; localPaddedX < GeneratedChunkColumn::kWidth; ++localPaddedX)
            {
                const int worldX = chunkBaseX + localPaddedX - GeneratedChunkColumn::kPadding;
                const int worldZ = chunkBaseZ + localPaddedZ - GeneratedChunkColumn::kPadding;
                const ChunkCoord sourceCoord = chunkCoordForBlock(worldX, worldZ);
                const int neighborX = sourceCoord.x - coord.x + 1;
                const int neighborZ = sourceCoord.z - coord.z + 1;
                if (neighborX < 0 || neighborX >= 3 || neighborZ < 0 || neighborZ >= 3)
                {
                    throw std::runtime_error("Meshing column source chunk is outside neighbor data range.");
                }

                const ChunkVoxelData& sourceVoxels =
                    neighborVoxels[static_cast<std::size_t>(neighborZ * 3 + neighborX)];
                if (!sourceVoxels.valid())
                {
                    throw std::runtime_error("Meshing column source chunk data is invalid.");
                }
                const int sourceLocalX = floorMod(worldX, kChunkSizeX);
                const int sourceLocalZ = floorMod(worldZ, kChunkSizeZ);
                for (int y = 0; y < kChunkHeight; ++y)
                {
                    const std::size_t sourceIndex = chunkBlockIndex(sourceLocalX, y, sourceLocalZ);
                    const std::uint16_t blockId = sourceVoxels.blockIds[sourceIndex];
                    std::uint8_t fluidId = sourceVoxels.fluidIds[sourceIndex];
                    std::uint8_t fluidAmount = std::min(sourceVoxels.fluidAmounts[sourceIndex], kMaxFluidAmount);
                    if (blockId != kAirBlockId || fluidId != kWaterFluidId || fluidAmount == 0)
                    {
                        fluidId = kNoFluidId;
                        fluidAmount = 0;
                    }

                    column.blockAt(localPaddedX, y, localPaddedZ) = blockId;
                    column.fluidIdAt(localPaddedX, y, localPaddedZ) = fluidId;
                    column.fluidAt(localPaddedX, y, localPaddedZ) = fluidAmount;
                }
            }
        }

        return column;
    }

    ChunkLoadUpdateStats updateLoadedChunks()
    {
        const int centerChunkX = chunkCoordForWorldPosition(camera_.position.x, kChunkSizeX);
        const int centerChunkZ = chunkCoordForWorldPosition(camera_.position.z, kChunkSizeZ);
        return chunkStreaming_.updateLoadedChunks(
            centerChunkX,
            centerChunkZ,
            [this](ChunkCoord coord)
            {
                unloadLoadedChunkData(coord);
                destroyChunkMeshes(coord);
            });
    }

    void uploadChunkMesh(
        int chunkX,
        int chunkZ,
        const std::vector<BlockVertex>& vertices,
        const std::vector<std::uint32_t>& indices,
        const std::vector<SubchunkDraw>& subchunkDraws,
        const std::vector<BlockVertex>& fluidVertices,
        const std::vector<std::uint32_t>& fluidIndices,
        const std::vector<SubchunkDraw>& fluidSubchunkDraws)
    {
        if (indices.empty() && fluidIndices.empty())
        {
            return;
        }

        ChunkMesh mesh{};
        mesh.chunkX = chunkX;
        mesh.chunkZ = chunkZ;
        mesh.vertexCount = static_cast<std::uint32_t>(vertices.size());
        mesh.indexCount = static_cast<std::uint32_t>(indices.size());
        mesh.fluidVertexCount = static_cast<std::uint32_t>(fluidVertices.size());
        mesh.fluidIndexCount = static_cast<std::uint32_t>(fluidIndices.size());
        mesh.subchunks = subchunkDraws;
        mesh.fluidSubchunks = fluidSubchunkDraws;
        meshVertexCount_ += vertices.size() + fluidVertices.size();
        meshIndexCount_ += indices.size() + fluidIndices.size();

        if (!indices.empty())
        {
            createDeviceLocalBuffer(
                vertices.data(),
                sizeof(BlockVertex) * vertices.size(),
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                mesh.vertexBuffer);
            createDeviceLocalBuffer(
                indices.data(),
                sizeof(std::uint32_t) * indices.size(),
                VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                mesh.indexBuffer);
        }

        if (!fluidIndices.empty())
        {
            createDeviceLocalBuffer(
                fluidVertices.data(),
                sizeof(BlockVertex) * fluidVertices.size(),
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                mesh.fluidVertexBuffer);
            createDeviceLocalBuffer(
                fluidIndices.data(),
                sizeof(std::uint32_t) * fluidIndices.size(),
                VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                mesh.fluidIndexBuffer);
        }

        chunkMeshes_.push_back(std::move(mesh));
    }

    std::size_t processCompletedSubchunkBuilds()
    {
        return chunkStreaming_.processCompletedSubchunkBuilds();
    }

    int processCompletedChunkUploads()
    {
        int uploadedCount = 0;
        for (int uploaded = 0; uploaded < chunkUploadsPerFrame_; ++uploaded)
        {
            ChunkBuildResult result{};
            if (!chunkStreaming_.popCompletedChunkBuild(result))
            {
                return uploadedCount;
            }

            if (!chunkStreaming_.shouldAcceptCompletedChunk(result.coord))
            {
                continue;
            }

            destroyChunkMeshes(result.coord);
            uploadChunkMesh(
                result.coord.x,
                result.coord.z,
                result.vertices,
                result.indices,
                result.subchunks,
                result.fluidVertices,
                result.fluidIndices,
                result.fluidSubchunks);
            registerLoadedChunkData(
                result.coord,
                std::move(result.blockIds),
                std::move(result.fluidIds),
                std::move(result.fluidAmounts));
            chunkStreaming_.markChunkLoaded(result.coord);
            ++uploadedCount;
        }
        return uploadedCount;
    }

    void processCompletedChunkBuilds()
    {
        processCompletedSubchunkBuilds();
        processCompletedChunkUploads();
    }

    void registerLoadedChunkData(
        ChunkCoord coord,
        std::vector<std::uint16_t> blockIds,
        std::vector<std::uint8_t> fluidIds,
        std::vector<std::uint8_t> fluidAmounts)
    {
        if (blockIds.size() != kChunkBlockCount ||
            fluidIds.size() != kChunkBlockCount ||
            fluidAmounts.size() != kChunkBlockCount)
        {
            logger_.warn("Skipping loaded chunk registration with invalid voxel data");
            return;
        }

        cacheChunkData(coord, {blockIds, fluidIds, fluidAmounts});

        std::lock_guard<std::mutex> lock(loadedChunksMutex_);
        bool dirty = false;
        const auto loadedIt = loadedChunks_.find(coord);
        if (loadedIt != loadedChunks_.end())
        {
            dirty = loadedIt->second.dirty;
        }
        loadedChunks_[coord] = {
            coord,
            std::move(blockIds),
            std::move(fluidIds),
            std::move(fluidAmounts),
            dirty,
        };
    }

    void unloadLoadedChunkData(ChunkCoord coord)
    {
        std::optional<LoadedChunkData> chunkData;
        {
            std::lock_guard<std::mutex> lock(loadedChunksMutex_);
            auto loadedIt = loadedChunks_.find(coord);
            if (loadedIt == loadedChunks_.end())
            {
                return;
            }
            chunkData = std::move(loadedIt->second);
            loadedChunks_.erase(loadedIt);
        }

        if (chunkData->dirty)
        {
            saveChunkToDisk(coord, {chunkData->blockIds, chunkData->fluidIds, chunkData->fluidAmounts});
        }
        cacheChunkData(coord, {chunkData->blockIds, chunkData->fluidIds, chunkData->fluidAmounts});
    }

    void cacheChunkData(ChunkCoord coord, const ChunkVoxelData& voxels)
    {
        if (!voxels.valid())
        {
            return;
        }

        {
            std::lock_guard<std::mutex> lock(chunkDataCacheMutex_);
            chunkDataCache_[coord] = voxels;
        }
        chunkDataCacheCv_.notify_all();
    }

    void saveLoadedChunks(bool onlyDirty)
    {
        std::vector<LoadedChunkData> chunksToSave;
        {
            std::lock_guard<std::mutex> lock(loadedChunksMutex_);
            chunksToSave.reserve(loadedChunks_.size());
            for (auto& [_, chunkData] : loadedChunks_)
            {
                if (!onlyDirty || chunkData.dirty)
                {
                    chunksToSave.push_back(chunkData);
                }
            }
        }

        for (const LoadedChunkData& chunkData : chunksToSave)
        {
            saveChunkToDisk(chunkData.coord, {chunkData.blockIds, chunkData.fluidIds, chunkData.fluidAmounts});
        }
    }

    void saveWorldState()
    {
        WorldSaveState state{};
        state.camera = camera_;
        state.camera.position.x = wrapWorldPosition(state.camera.position.x);
        state.camera.position.z = wrapWorldPosition(state.camera.position.z);
        state.movementMode = playerController_.movementMode();
        state.cameraViewMode = playerController_.cameraViewMode();
        state.worldTimeTicks = worldTimeTicks_;
        state.worldSeed = worldSeed_;

        try
        {
            std::lock_guard<std::mutex> lock(saveMutex_);
            worldSave_.saveWorldState(state);
        }
        catch (const std::exception& exception)
        {
            logger_.error(std::string("Failed to save world state: ") + exception.what());
        }
    }

    std::optional<ChunkVoxelData> loadChunkFromDisk(ChunkCoord coord)
    {
        try
        {
            std::lock_guard<std::mutex> lock(saveMutex_);
            return worldSave_.loadChunk(coord);
        }
        catch (const std::exception& exception)
        {
            logger_.error(std::string("Failed to load chunk from disk: ") + exception.what());
            return std::nullopt;
        }
    }

    bool saveChunkToDisk(ChunkCoord coord, const ChunkVoxelData& voxels)
    {
        try
        {
            std::lock_guard<std::mutex> lock(saveMutex_);
            worldSave_.saveChunk(coord, voxels);
            return true;
        }
        catch (const std::exception& exception)
        {
            logger_.error(std::string("Failed to save chunk to disk: ") + exception.what());
            return false;
        }
    }

    std::pair<std::size_t, std::size_t> chunkBuildQueueSizes()
    {
        return chunkStreaming_.queueSizes();
    }

    void createDebugFontAtlasTexture()
    {
        AddFontResourceExW(sourcePathWide(L"/assets/fonts/VCR_OSD_MONO.ttf").c_str(), FR_PRIVATE, nullptr);
        debugFontAtlasTexture_ = createTextureFromPixels(
            debugTextRenderer_.renderGlyphAtlas(),
            kDebugFontAtlasWidth,
            kDebugFontAtlasHeight,
            VK_FORMAT_R8G8B8A8_UNORM);
    }

    Texture finalizeTextureUpload(TextureUpload upload)
    {
        for (VkCommandBuffer commandBuffer : upload.commandBuffers)
        {
            deferUploadCommandBuffer(commandBuffer);
        }
        deferUploadStagingBuffer(upload.stagingBuffer);
        return upload.texture;
    }

    Texture createTexture(const std::wstring& path)
    {
        return finalizeTextureUpload(textureManager_.createTextureFromFile(path, VK_FORMAT_R8G8B8A8_SRGB));
    }

    Texture createTextureFromPixels(
        const std::vector<std::uint8_t>& pixels,
        std::uint32_t width,
        std::uint32_t height,
        VkFormat format)
    {
        return finalizeTextureUpload(textureManager_.createTextureFromPixels(pixels, width, height, format));
    }

    void createTextureSampler()
    {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(physicalDevice_, &properties);

        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_NEAREST;
        samplerInfo.minFilter = VK_FILTER_NEAREST;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.anisotropyEnable = VK_FALSE;
        samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        samplerInfo.mipLodBias = -0.5f;
        samplerInfo.minLod = 0.0f;
        samplerInfo.maxLod = VK_LOD_CLAMP_NONE;

        if (vkCreateSampler(device_, &samplerInfo, nullptr, &textureSampler_) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create Vulkan texture sampler.");
        }
    }

    void createUniformBuffer()
    {
        uniformBuffer_ = resourceContext_.createBuffer(
            sizeof(SkyUniform),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        vkMapMemory(device_, uniformBuffer_.memory, 0, sizeof(SkyUniform), 0, &uniformMappedMemory_);

        blockUniformBuffer_ = resourceContext_.createBuffer(
            sizeof(BlockUniform),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        vkMapMemory(device_, blockUniformBuffer_.memory, 0, sizeof(BlockUniform), 0, &blockUniformMappedMemory_);

        debugTextVertexBuffer_ = resourceContext_.createBuffer(
            sizeof(DebugTextVertex) * kMaxDebugTextVertices,
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        vkMapMemory(
            device_,
            debugTextVertexBuffer_.memory,
            0,
            sizeof(DebugTextVertex) * kMaxDebugTextVertices,
            0,
            &debugTextVertexMappedMemory_);

        crosshairVertexBuffer_ = resourceContext_.createBuffer(
            sizeof(DebugTextVertex) * kCrosshairVertexCount,
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        vkMapMemory(
            device_,
            crosshairVertexBuffer_.memory,
            0,
            sizeof(DebugTextVertex) * kCrosshairVertexCount,
            0,
            &crosshairVertexMappedMemory_);
        updateCrosshairVertices();

        selectionVertexBuffer_ = resourceContext_.createBuffer(
            sizeof(SelectionVertex) * kSelectionOutlineVertexCount,
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        vkMapMemory(
            device_,
            selectionVertexBuffer_.memory,
            0,
            sizeof(SelectionVertex) * kSelectionOutlineVertexCount,
            0,
            &selectionVertexMappedMemory_);

        if (playerModel_.isLoaded())
        {
            playerVertexBuffer_ = resourceContext_.createBuffer(
                sizeof(BlockVertex) * playerModel_.vertexCount(),
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

            vkMapMemory(
                device_,
                playerVertexBuffer_.memory,
                0,
                sizeof(BlockVertex) * playerModel_.vertexCount(),
                0,
                &playerVertexMappedMemory_);

            createDeviceLocalBuffer(
                playerModel_.indices().data(),
                sizeof(std::uint32_t) * playerModel_.indices().size(),
                VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                playerIndexBuffer_);
        }
    }

    void createDescriptorPool()
    {
        const std::array<VkDescriptorPoolSize, 2> poolSizes = {{
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 6},
        }};
        descriptorPool_ = createVulkanDescriptorPool(
            device_,
            poolSizes,
            5,
            "Failed to create Vulkan descriptor pool.");
    }

    void createDescriptorSet()
    {
        descriptorSet_ = allocateVulkanDescriptorSet(
            device_,
            descriptorPool_,
            descriptorSetLayout_,
            "Failed to allocate Vulkan descriptor set.");

        const VkDescriptorBufferInfo bufferInfo = descriptorBufferInfo(uniformBuffer_.buffer, sizeof(SkyUniform));
        const VkDescriptorImageInfo sunImageInfo = descriptorImageInfo(sunTexture_.view, textureSampler_);
        const VkDescriptorImageInfo moonImageInfo = descriptorImageInfo(moonTexture_.view, textureSampler_);

        const std::array<VkWriteDescriptorSet, 3> descriptorWrites = {
            writeBufferDescriptor(descriptorSet_, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, bufferInfo),
            writeImageDescriptor(descriptorSet_, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, sunImageInfo),
            writeImageDescriptor(descriptorSet_, 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, moonImageInfo),
        };
        updateVulkanDescriptorSets(device_, descriptorWrites);
    }

    void createBlockDescriptorSet()
    {
        blockDescriptorSet_ = allocateVulkanDescriptorSet(
            device_,
            descriptorPool_,
            blockDescriptorSetLayout_,
            "Failed to allocate Vulkan block descriptor set.");

        const VkDescriptorBufferInfo bufferInfo = descriptorBufferInfo(blockUniformBuffer_.buffer, sizeof(BlockUniform));
        const VkDescriptorImageInfo imageInfo = descriptorImageInfo(blockTextureArray_.view, textureSampler_);

        const std::array<VkWriteDescriptorSet, 2> descriptorWrites = {
            writeBufferDescriptor(blockDescriptorSet_, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, bufferInfo),
            writeImageDescriptor(blockDescriptorSet_, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, imageInfo),
        };
        updateVulkanDescriptorSets(device_, descriptorWrites);
    }

    void createPlayerDescriptorSet()
    {
        playerDescriptorSet_ = allocateVulkanDescriptorSet(
            device_,
            descriptorPool_,
            blockDescriptorSetLayout_,
            "Failed to allocate Vulkan player descriptor set.");

        const VkDescriptorBufferInfo bufferInfo = descriptorBufferInfo(blockUniformBuffer_.buffer, sizeof(BlockUniform));
        const VkDescriptorImageInfo imageInfo = descriptorImageInfo(playerTexture_.view, textureSampler_);

        const std::array<VkWriteDescriptorSet, 2> descriptorWrites = {
            writeBufferDescriptor(playerDescriptorSet_, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, bufferInfo),
            writeImageDescriptor(playerDescriptorSet_, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, imageInfo),
        };
        updateVulkanDescriptorSets(device_, descriptorWrites);
    }

    void createCrosshairDescriptorSet()
    {
        crosshairDescriptorSet_ = allocateVulkanDescriptorSet(
            device_,
            descriptorPool_,
            debugTextDescriptorSetLayout_,
            "Failed to allocate Vulkan crosshair descriptor set.");

        const VkDescriptorImageInfo imageInfo = descriptorImageInfo(crosshairTexture_.view, textureSampler_);
        const std::array<VkWriteDescriptorSet, 1> descriptorWrites = {
            writeImageDescriptor(crosshairDescriptorSet_, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, imageInfo),
        };
        updateVulkanDescriptorSets(device_, descriptorWrites);
    }

    void createDebugTextDescriptorSet()
    {
        debugTextDescriptorSet_ = allocateVulkanDescriptorSet(
            device_,
            descriptorPool_,
            debugTextDescriptorSetLayout_,
            "Failed to allocate Vulkan debug text descriptor set.");

        const VkDescriptorImageInfo imageInfo = descriptorImageInfo(debugFontAtlasTexture_.view, textureSampler_);
        const std::array<VkWriteDescriptorSet, 1> descriptorWrites = {
            writeImageDescriptor(debugTextDescriptorSet_, 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, imageInfo),
        };
        updateVulkanDescriptorSets(device_, descriptorWrites);
    }

    void cacheStaticDebugText()
    {
        versionDebugLine_ = L"VULKANVOXEL " + narrowToWide(VULKAN_VOXEL_VERSION);
        gpuDebugLine_ = L"GPU: " + narrowToWide(physicalDeviceProperties_.deviceName);
        cpuDebugLine_ = L"CPU: " + getCpuBrandString();
        apiDebugLine_ = L"API: VULKAN " + versionString(physicalDeviceProperties_.apiVersion);
        driverDebugLine_ = L"DRIVER: " + std::to_wstring(physicalDeviceProperties_.driverVersion);
    }

    void createCommandBuffers()
    {
        commandBuffers_.resize(kMaxFramesInFlight);

        VkCommandBufferAllocateInfo allocateInfo{};
        allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocateInfo.commandPool = commandPool_;
        allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocateInfo.commandBufferCount = static_cast<std::uint32_t>(commandBuffers_.size());

        if (vkAllocateCommandBuffers(device_, &allocateInfo, commandBuffers_.data()) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to allocate Vulkan command buffers.");
        }
    }

    void createSyncObjects()
    {
        imageAvailableSemaphores_.resize(kMaxFramesInFlight);
        renderFinishedSemaphores_.resize(kMaxFramesInFlight);
        inFlightFences_.resize(kMaxFramesInFlight);

        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (std::size_t i = 0; i < kMaxFramesInFlight; ++i)
        {
            if (vkCreateSemaphore(device_, &semaphoreInfo, nullptr, &imageAvailableSemaphores_[i]) != VK_SUCCESS ||
                vkCreateSemaphore(device_, &semaphoreInfo, nullptr, &renderFinishedSemaphores_[i]) != VK_SUCCESS ||
                vkCreateFence(device_, &fenceInfo, nullptr, &inFlightFences_[i]) != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to create Vulkan synchronization objects.");
            }
        }
    }

    void deferUploadCommandBuffer(VkCommandBuffer commandBuffer)
    {
        deferredUploadCleanups_.push_back({
            {},
            commandBuffer,
            frameCounter_ + static_cast<std::uint64_t>(kMaxFramesInFlight),
        });
    }

    void deferUploadStagingBuffer(Buffer& stagingBuffer)
    {
        deferredUploadCleanups_.push_back({
            stagingBuffer,
            VK_NULL_HANDLE,
            frameCounter_ + static_cast<std::uint64_t>(kMaxFramesInFlight),
        });
        stagingBuffer = {};
    }

    void createDeviceLocalBuffer(
        const void* sourceData,
        VkDeviceSize size,
        VkBufferUsageFlags usage,
        Buffer& destinationBuffer)
    {
        Buffer stagingBuffer = resourceContext_.createBuffer(
            size,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        void* data = nullptr;
        vkMapMemory(device_, stagingBuffer.memory, 0, size, 0, &data);
        std::memcpy(data, sourceData, static_cast<std::size_t>(size));
        vkUnmapMemory(device_, stagingBuffer.memory);

        destinationBuffer = resourceContext_.createBuffer(
            size,
            usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        deferUploadCommandBuffer(resourceContext_.copyBuffer(stagingBuffer.buffer, destinationBuffer.buffer, size));
        deferUploadStagingBuffer(stagingBuffer);
    }

    void uploadTexturePixels(Texture& texture, const std::vector<std::uint8_t>& pixels)
    {
        TextureUpload upload = textureManager_.uploadTexturePixels(texture, pixels);
        for (VkCommandBuffer commandBuffer : upload.commandBuffers)
        {
            deferUploadCommandBuffer(commandBuffer);
        }
        deferUploadStagingBuffer(upload.stagingBuffer);
        texture = upload.texture;
    }

    void drawFrame()
    {
        FrameProfiler currentProfile = frameProfiler_;
        const auto frameStart = std::chrono::steady_clock::now();
        auto mark = []()
        {
            return std::chrono::steady_clock::now();
        };
        auto elapsedMs = [](std::chrono::steady_clock::time_point begin)
        {
            return std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - begin).count();
        };

        auto sectionStart = mark();
        vkWaitForFences(device_, 1, &inFlightFences_[currentFrame_], VK_TRUE, UINT64_MAX);
        currentProfile.fenceWaitMs = elapsedMs(sectionStart);

        collectDeferredUploadCleanups(false);
        collectDeferredChunkBufferDestroys(false);

        std::uint32_t imageIndex = 0;
        sectionStart = mark();
        VkResult result = vkAcquireNextImageKHR(
            device_,
            swapchain_,
            UINT64_MAX,
            imageAvailableSemaphores_[currentFrame_],
            VK_NULL_HANDLE,
            &imageIndex);
        currentProfile.acquireMs = elapsedMs(sectionStart);

        if (result == VK_ERROR_OUT_OF_DATE_KHR)
        {
            recreateSwapchain();
            return;
        }
        if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
        {
            throw std::runtime_error("Failed to acquire a Vulkan swapchain image.");
        }

        vkResetFences(device_, 1, &inFlightFences_[currentFrame_]);
        vkResetCommandBuffer(commandBuffers_[currentFrame_], 0);

        updateCameraMovement();
        updatePlayerRenderMesh();

        sectionStart = mark();
        const ChunkLoadUpdateStats loadStats = updateLoadedChunks();
        currentProfile.loadUpdateMs = elapsedMs(sectionStart);
        currentProfile.chunksQueued = loadStats.queued;
        currentProfile.chunksUnloaded = loadStats.unloaded;

        sectionStart = mark();
        currentProfile.subchunkResultsProcessed = processCompletedSubchunkBuilds();
        currentProfile.subchunkDoneMs = elapsedMs(sectionStart);

        sectionStart = mark();
        currentProfile.chunksUploaded = processCompletedChunkUploads();
        currentProfile.chunkUploadMs = elapsedMs(sectionStart);
        currentProfile.chunksLoaded = static_cast<std::size_t>(currentProfile.chunksUploaded);

        updateBlockSelection(false);

        sectionStart = mark();
        updateUniformBuffer();
        currentProfile.uniformMs = elapsedMs(sectionStart);

        frameProfiler_ = currentProfile;
        sectionStart = mark();
        updateDebugTextOverlay();
        currentProfile.debugTextMs = elapsedMs(sectionStart);
        frameProfiler_.debugTextMs = currentProfile.debugTextMs;

        sectionStart = mark();
        recordCommandBuffer(commandBuffers_[currentFrame_], imageIndex);
        currentProfile.recordMs = elapsedMs(sectionStart);
        sectionStart = mark();

        VkSemaphore waitSemaphores[] = {imageAvailableSemaphores_[currentFrame_]};
        VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        VkSemaphore signalSemaphores[] = {renderFinishedSemaphores_[currentFrame_]};

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers_[currentFrame_];
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;

        if (vkQueueSubmit(graphicsQueue_, 1, &submitInfo, inFlightFences_[currentFrame_]) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to submit Vulkan draw command buffer.");
        }

        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &swapchain_;
        presentInfo.pImageIndices = &imageIndex;

        result = vkQueuePresentKHR(presentQueue_, &presentInfo);
        currentProfile.submitPresentMs = elapsedMs(sectionStart);
        currentProfile.frameCpuMs = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - frameStart).count();
        frameProfiler_ = currentProfile;
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized_)
        {
            framebufferResized_ = false;
            recreateSwapchain();
        }
        else if (result != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to present a Vulkan swapchain image.");
        }

        currentFrame_ = (currentFrame_ + 1) % kMaxFramesInFlight;
        ++frameCounter_;
    }

    void updateCameraMovement()
    {
        const auto now = std::chrono::steady_clock::now();
        const float rawDeltaSeconds = static_cast<float>(std::chrono::duration<double>(now - lastFrameTime_).count());
        lastFrameTime_ = now;
        advanceWorldTime(rawDeltaSeconds);

        PlayerInputState input{};
        input.moveForward = glfwGetKey(window_, GLFW_KEY_W) == GLFW_PRESS;
        input.moveBackward = glfwGetKey(window_, GLFW_KEY_S) == GLFW_PRESS;
        input.moveRight = glfwGetKey(window_, GLFW_KEY_D) == GLFW_PRESS;
        input.moveLeft = glfwGetKey(window_, GLFW_KEY_A) == GLFW_PRESS;
        input.moveUp = glfwGetKey(window_, GLFW_KEY_SPACE) == GLFW_PRESS;
        input.moveDown =
            glfwGetKey(window_, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
            glfwGetKey(window_, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;

        playerController_.update(
            camera_,
            input,
            rawDeltaSeconds,
            [this](int x, int y, int z)
            {
                return isWorldBlockSolid(x, y, z);
            });
    }

    void advanceWorldTime(float deltaSeconds)
    {
        const float clampedDeltaSeconds = std::clamp(deltaSeconds, 0.0f, 0.25f);
        worldTickAccumulator_ += static_cast<double>(clampedDeltaSeconds) * kWorldTicksPerRealSecond;
        const auto elapsedTicks = static_cast<std::uint64_t>(worldTickAccumulator_);
        if (elapsedTicks == 0)
        {
            return;
        }

        worldTimeTicks_ += elapsedTicks;
        worldTickAccumulator_ -= static_cast<double>(elapsedTicks);
    }

    bool isWorldBlockSolid(int x, int y, int z) const
    {
        if (y < 0 || y >= kChunkHeight)
        {
            return false;
        }
        if (!isChunkLoadedForBlock(x, z))
        {
            return true;
        }

        const std::optional<std::uint16_t> blockId = loadedBlockIdAt(x, y, z);
        return blockId.has_value() && blockRegistry_.isCollision(*blockId);
    }

    bool isRaycastBlockSolid(int x, int y, int z) const
    {
        if (y < 0 || y >= kChunkHeight || !isChunkLoadedForBlock(x, z))
        {
            return false;
        }

        const std::optional<std::uint16_t> blockId = loadedBlockIdAt(x, y, z);
        return blockId.has_value() && blockRegistry_.isRaycastTarget(*blockId);
    }

    bool isChunkLoadedForBlock(int x, int z) const
    {
        std::lock_guard<std::mutex> lock(loadedChunksMutex_);
        return loadedChunks_.contains(chunkCoordForBlock(x, z));
    }

    std::optional<std::uint16_t> loadedBlockIdAt(int x, int y, int z) const
    {
        if (y < 0 || y >= kChunkHeight)
        {
            return std::nullopt;
        }

        const ChunkCoord coord = chunkCoordForBlock(x, z);
        const int localX = floorMod(x, kChunkSizeX);
        const int localZ = floorMod(z, kChunkSizeZ);

        std::lock_guard<std::mutex> lock(loadedChunksMutex_);
        const auto loadedIt = loadedChunks_.find(coord);
        if (loadedIt == loadedChunks_.end() ||
            loadedIt->second.blockIds.size() != kChunkBlockCount)
        {
            return std::nullopt;
        }

        return loadedIt->second.blockIds[chunkBlockIndex(localX, y, localZ)];
    }

    bool setLoadedBlockIdAt(int x, int y, int z, std::uint16_t blockId)
    {
        if (y < 0 || y >= kChunkHeight)
        {
            return false;
        }

        const ChunkCoord coord = chunkCoordForBlock(x, z);
        const int localX = floorMod(x, kChunkSizeX);
        const int localZ = floorMod(z, kChunkSizeZ);

        std::lock_guard<std::mutex> lock(loadedChunksMutex_);
        auto loadedIt = loadedChunks_.find(coord);
        if (loadedIt == loadedChunks_.end() ||
            loadedIt->second.blockIds.size() != kChunkBlockCount)
        {
            return false;
        }

        const std::size_t voxelIndex = chunkBlockIndex(localX, y, localZ);
        loadedIt->second.blockIds[voxelIndex] = blockId;
        if (blockRegistry_.isCollision(blockId) &&
            loadedIt->second.fluidIds.size() == kChunkBlockCount &&
            loadedIt->second.fluidAmounts.size() == kChunkBlockCount)
        {
            loadedIt->second.fluidIds[voxelIndex] = kNoFluidId;
            loadedIt->second.fluidAmounts[voxelIndex] = 0;
        }
        loadedIt->second.dirty = true;

        {
            std::lock_guard<std::mutex> cacheLock(chunkDataCacheMutex_);
            auto cacheIt = chunkDataCache_.find(coord);
            if (cacheIt != chunkDataCache_.end() && cacheIt->second.valid())
            {
                cacheIt->second.blockIds[voxelIndex] = loadedIt->second.blockIds[voxelIndex];
                if (loadedIt->second.fluidIds.size() == kChunkBlockCount &&
                    loadedIt->second.fluidAmounts.size() == kChunkBlockCount)
                {
                    cacheIt->second.fluidIds[voxelIndex] = loadedIt->second.fluidIds[voxelIndex];
                    cacheIt->second.fluidAmounts[voxelIndex] = loadedIt->second.fluidAmounts[voxelIndex];
                }
            }
        }
        chunkDataCacheCv_.notify_all();
        return true;
    }

    ChunkCoord chunkCoordForBlock(int x, int z) const
    {
        return {
            floorDiv(x, kChunkSizeX),
            floorDiv(z, kChunkSizeZ),
        };
    }

    Vec3 playerFeetPosition() const
    {
        return playerController_.playerFeetPosition(camera_);
    }

    Vec3 playerEyePosition() const
    {
        return playerController_.playerEyePosition(camera_);
    }

    bool isThirdPersonView() const
    {
        return playerController_.isThirdPersonView();
    }

    Vec3 renderCameraForward() const
    {
        return playerController_.renderCameraForward(camera_);
    }

    Vec3 renderCameraPosition() const
    {
        return playerController_.renderCameraPosition(camera_);
    }

    static FrustumPlane makeFrustumPlane(Vec3 a, Vec3 b, Vec3 c, Vec3 insidePoint)
    {
        FrustumPlane plane{};
        plane.normal = normalize(cross(b - a, c - a));
        plane.distance = -dot(plane.normal, a);
        if (dot(plane.normal, insidePoint) + plane.distance < 0.0f)
        {
            plane.normal = -plane.normal;
            plane.distance = -plane.distance;
        }
        return plane;
    }

    static ViewFrustum makeViewFrustum(
        Vec3 position,
        Vec3 forward,
        float fovY,
        float aspect,
        float nearPlane,
        float farPlane)
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
        const float nearHalfHeight = std::tan(fovY * 0.5f) * nearPlane;
        const float nearHalfWidth = nearHalfHeight * aspect;
        const float farHalfHeight = std::tan(fovY * 0.5f) * farPlane;
        const float farHalfWidth = farHalfHeight * aspect;
        const Vec3 nearCenter = position + forward * nearPlane;
        const Vec3 farCenter = position + forward * farPlane;
        const Vec3 insidePoint = position + forward * ((nearPlane + farPlane) * 0.5f);

        const Vec3 nearTopLeft = nearCenter + up * nearHalfHeight - right * nearHalfWidth;
        const Vec3 nearTopRight = nearCenter + up * nearHalfHeight + right * nearHalfWidth;
        const Vec3 nearBottomLeft = nearCenter - up * nearHalfHeight - right * nearHalfWidth;
        const Vec3 nearBottomRight = nearCenter - up * nearHalfHeight + right * nearHalfWidth;
        const Vec3 farTopLeft = farCenter + up * farHalfHeight - right * farHalfWidth;
        const Vec3 farTopRight = farCenter + up * farHalfHeight + right * farHalfWidth;
        const Vec3 farBottomLeft = farCenter - up * farHalfHeight - right * farHalfWidth;
        const Vec3 farBottomRight = farCenter - up * farHalfHeight + right * farHalfWidth;

        ViewFrustum frustum{};
        frustum.planes[0] = makeFrustumPlane(nearTopLeft, nearTopRight, nearBottomRight, insidePoint);
        frustum.planes[1] = makeFrustumPlane(farTopRight, farTopLeft, farBottomLeft, insidePoint);
        frustum.planes[2] = makeFrustumPlane(nearTopLeft, nearBottomLeft, farBottomLeft, insidePoint);
        frustum.planes[3] = makeFrustumPlane(nearBottomRight, nearTopRight, farBottomRight, insidePoint);
        frustum.planes[4] = makeFrustumPlane(nearTopRight, nearTopLeft, farTopLeft, insidePoint);
        frustum.planes[5] = makeFrustumPlane(nearBottomLeft, nearBottomRight, farBottomRight, insidePoint);
        return frustum;
    }

    static bool isAabbVisibleInFrustum(const ViewFrustum& frustum, Vec3 minimum, Vec3 maximum)
    {
        for (const FrustumPlane& plane : frustum.planes)
        {
            const Vec3 positiveVertex{
                plane.normal.x >= 0.0f ? maximum.x : minimum.x,
                plane.normal.y >= 0.0f ? maximum.y : minimum.y,
                plane.normal.z >= 0.0f ? maximum.z : minimum.z,
            };
            if (dot(plane.normal, positiveVertex) + plane.distance < 0.0f)
            {
                return false;
            }
        }
        return true;
    }

    static bool isSubchunkVisible(const ViewFrustum& frustum, const SubchunkDraw& draw)
    {
        const Vec3 minimum{
            static_cast<float>(draw.chunkX * kChunkSizeX),
            static_cast<float>(draw.subchunkY * kSubchunkSize),
            static_cast<float>(draw.chunkZ * kChunkSizeZ),
        };
        const Vec3 maximum{
            minimum.x + static_cast<float>(kChunkSizeX),
            minimum.y + static_cast<float>(kSubchunkSize),
            minimum.z + static_cast<float>(kChunkSizeZ),
        };
        return isAabbVisibleInFrustum(frustum, minimum, maximum);
    }

    void updateBlockSelection(bool force)
    {
        const auto now = std::chrono::steady_clock::now();
        if (!force && now - lastInteractionRaycastTime_ < std::chrono::milliseconds(50))
        {
            return;
        }
        lastInteractionRaycastTime_ = now;

        selectedBlock_ = raycastBlocks(
            playerEyePosition(),
            cameraForward(camera_.yaw, camera_.pitch),
            interactionDistance_,
            [this](int x, int y, int z)
            {
                return isRaycastBlockSolid(x, y, z);
            },
            [this](int x, int y, int z)
            {
                return y >= 0 && y < kChunkHeight && isChunkLoadedForBlock(x, z);
            });
        updateSelectionOutlineVertices();
    }

    void editSelectedBlock(std::uint16_t blockId)
    {
        updateBlockSelection(true);
        if (!selectedBlock_)
        {
            return;
        }

        int x = selectedBlock_->x;
        int y = selectedBlock_->y;
        int z = selectedBlock_->z;
        if (blockId != kAirBlockId)
        {
            if (selectedBlock_->normalX == 0 &&
                selectedBlock_->normalY == 0 &&
                selectedBlock_->normalZ == 0)
            {
                return;
            }
            x += selectedBlock_->normalX;
            y += selectedBlock_->normalY;
            z += selectedBlock_->normalZ;
        }

        if (y < 0 || y >= kChunkHeight || !isChunkLoadedForBlock(x, z))
        {
            return;
        }
        if (blockId != kAirBlockId && isWorldBlockSolid(x, y, z))
        {
            return;
        }

        if (!setLoadedBlockIdAt(x, y, z, blockId))
        {
            return;
        }
        remeshEditedBlock(x, y, z);
        updateBlockSelection(true);
    }

    void remeshEditedBlock(int x, int, int z)
    {
        const ChunkCoord coord = chunkCoordForBlock(x, z);
        remeshChunk(coord);

        const int localX = x - coord.x * kChunkSizeX;
        const int localZ = z - coord.z * kChunkSizeZ;
        if (localX == 0)
        {
            remeshChunk({coord.x - 1, coord.z});
        }
        else if (localX == kChunkSizeX - 1)
        {
            remeshChunk({coord.x + 1, coord.z});
        }

        if (localZ == 0)
        {
            remeshChunk({coord.x, coord.z - 1});
        }
        else if (localZ == kChunkSizeZ - 1)
        {
            remeshChunk({coord.x, coord.z + 1});
        }
    }

    void remeshChunk(ChunkCoord coord)
    {
        chunkStreaming_.rebuildLoadedChunk(coord);
    }

    void updateSelectionOutlineVertices()
    {
        if (selectionVertexMappedMemory_ == nullptr)
        {
            selectionVertexCount_ = 0;
            return;
        }
        if (!selectedBlock_)
        {
            selectionVertexCount_ = 0;
            return;
        }

        const float minX = static_cast<float>(selectedBlock_->x) - 0.002f;
        const float minY = static_cast<float>(selectedBlock_->y) - 0.002f;
        const float minZ = static_cast<float>(selectedBlock_->z) - 0.002f;
        const float maxX = static_cast<float>(selectedBlock_->x + 1) + 0.002f;
        const float maxY = static_cast<float>(selectedBlock_->y + 1) + 0.002f;
        const float maxZ = static_cast<float>(selectedBlock_->z + 1) + 0.002f;

        const std::array<Vec3, 8> corners = {{
            {minX, minY, minZ},
            {maxX, minY, minZ},
            {maxX, maxY, minZ},
            {minX, maxY, minZ},
            {minX, minY, maxZ},
            {maxX, minY, maxZ},
            {maxX, maxY, maxZ},
            {minX, maxY, maxZ},
        }};
        constexpr std::array<std::array<int, 2>, 12> edges = {{
            {{0, 1}}, {{1, 2}}, {{2, 3}}, {{3, 0}},
            {{4, 5}}, {{5, 6}}, {{6, 7}}, {{7, 4}},
            {{0, 4}}, {{1, 5}}, {{2, 6}}, {{3, 7}},
        }};

        std::array<SelectionVertex, kSelectionOutlineVertexCount> vertices{};
        std::size_t vertexIndex = 0;
        for (const auto& edge : edges)
        {
            const Vec3 a = corners[static_cast<std::size_t>(edge[0])];
            const Vec3 b = corners[static_cast<std::size_t>(edge[1])];
            vertices[vertexIndex++] = {{a.x, a.y, a.z}};
            vertices[vertexIndex++] = {{b.x, b.y, b.z}};
        }

        std::memcpy(selectionVertexMappedMemory_, vertices.data(), sizeof(vertices));
        selectionVertexCount_ = static_cast<std::uint32_t>(vertices.size());
    }

    void updatePlayerRenderMesh()
    {
        if (!playerModel_.isLoaded() || !isThirdPersonView() || playerVertexMappedMemory_ == nullptr)
        {
            playerVertexCount_ = 0;
            return;
        }

        const std::vector<BlockVertex>& renderVertices =
            playerModel_.updateRenderVertices(playerFeetPosition(), camera_.yaw, kPlayerHeight);

        std::memcpy(
            playerVertexMappedMemory_,
            renderVertices.data(),
            sizeof(BlockVertex) * renderVertices.size());
        playerVertexCount_ = static_cast<std::uint32_t>(renderVertices.size());
    }

    void updateCrosshairVertices()
    {
        if (crosshairVertexMappedMemory_ == nullptr)
        {
            crosshairVertexCount_ = 0;
            return;
        }

        const std::vector<DebugTextVertex> vertices = buildCrosshairVertices(
            crosshairTexture_.width,
            crosshairTexture_.height,
            swapchainExtent_.width,
            swapchainExtent_.height);

        if (!vertices.empty())
        {
            std::memcpy(
                crosshairVertexMappedMemory_,
                vertices.data(),
                sizeof(DebugTextVertex) * vertices.size());
        }
        crosshairVertexCount_ = static_cast<std::uint32_t>(vertices.size());
    }

    void updateDebugTextOverlay()
    {
        ++debugTextOverlay_.framesSinceUpdate;

        const auto now = std::chrono::steady_clock::now();
        const auto elapsed = now - debugTextOverlay_.lastUpdate;
        if (elapsed < std::chrono::milliseconds(50))
        {
            return;
        }

        const double elapsedSeconds = std::chrono::duration<double>(elapsed).count();
        const double fps = static_cast<double>(debugTextOverlay_.framesSinceUpdate) / elapsedSeconds;
        const double frameMs = 1000.0 / std::max(fps, 0.001);
        const int fpsInteger = std::clamp(static_cast<int>(std::lround(fps)), 0, 9999);

        auto profilerLine = [](const wchar_t* label, double milliseconds)
        {
            std::wostringstream line;
            line << label << L": " << std::fixed << std::setprecision(3) << milliseconds << L"MS";
            return line.str();
        };

        std::wostringstream fpsText;
        fpsText << L"FPS: " << std::setw(4) << std::setfill(L'0') << fpsInteger
                << L" [" << std::setw(6) << std::setfill(L'0') << std::fixed << std::setprecision(3)
                << frameMs << L"MS]";

        const WorldTimeParts timeParts = splitWorldTime(worldTimeTicks_);
        std::wostringstream timeText;
        timeText << L"TIME: " << timeParts.day
                 << L" DAY " << std::setw(2) << std::setfill(L'0') << timeParts.hour
                 << L" H " << std::setw(2) << std::setfill(L'0') << timeParts.minute
                 << L" M";

        std::wostringstream positionText;
        positionText << L"POS: X:" << formatCoordinateValue(camera_.position.x)
                     << L" [" << formatCoordinateValue(wrapWorldPosition(camera_.position.x)) << L"] / Y:"
                     << formatCoordinateValue(camera_.position.y)
                     << L" [" << formatCoordinateValue(camera_.position.y) << L"] / Z:"
                     << formatCoordinateValue(camera_.position.z)
                     << L" [" << formatCoordinateValue(wrapWorldPosition(camera_.position.z)) << L"]";

        std::wostringstream landformText;
        const int playerBlockX = static_cast<int>(std::floor(camera_.position.x));
        const int playerBlockZ = static_cast<int>(std::floor(camera_.position.z));
        landformText << std::fixed << std::setprecision(3)
                     << L"Landform: " << worldGenerator_.landformRawAt(playerBlockX, playerBlockZ)
                     << L" [" << worldGenerator_.landformCenterOffsetAt(playerBlockX, playerBlockZ) << L"]";

        const double yawDegrees = static_cast<double>(camera_.yaw) * 180.0 / static_cast<double>(kPi);
        const double pitchDegrees = static_cast<double>(camera_.pitch) * 180.0 / static_cast<double>(kPi);
        std::wostringstream cameraText;
        cameraText << std::showpos << std::fixed << std::setprecision(3)
                   << L"CAM: YAW " << std::setw(8) << yawDegrees
                   << L" PIT " << std::setw(7) << pitchDegrees
                   << std::noshowpos << L" [" << directionNameForYaw(camera_.yaw) << L"]";

        debugTextOverlay_.lines = {
            fpsText.str(),
            positionText.str(),
            landformText.str(),
            cameraText.str(),
            timeText.str(),
            L"SEED: " + std::to_wstring(worldSeed_),
        };

        const FrameProfiler profile = frameProfiler_;
        debugTextOverlay_.bottomLeftLines = {
            profilerLine(L"FRAME CPU", profile.frameCpuMs),
            profilerLine(L"FENCE", profile.fenceWaitMs),
            profilerLine(L"ACQUIRE", profile.acquireMs),
            profilerLine(L"LOAD", profile.loadUpdateMs),
            profilerLine(L"DONE", profile.subchunkDoneMs),
            profilerLine(L"UPLOAD", profile.chunkUploadMs),
            profilerLine(L"UNIFORM", profile.uniformMs),
            profilerLine(L"DEBUG", profile.debugTextMs),
            profilerLine(L"RECORD", profile.recordMs),
            profilerLine(L"SUBMIT", profile.submitPresentMs),
        };

        const bool shouldUpdateRightText =
            now - debugTextOverlay_.rightLastUpdate >= std::chrono::milliseconds(500);
        if (shouldUpdateRightText)
        {
            updateRightDebugText();
            debugTextOverlay_.rightLastUpdate = now;
        }

        debugTextOverlay_.lastUpdate = now;
        debugTextOverlay_.framesSinceUpdate = 0;

        rebuildDebugTextVertices();
    }

    void rebuildDebugTextVertices()
    {
        std::vector<DebugTextVertex> vertices = debugTextRenderer_.buildVertices(
            debugTextOverlay_.lines,
            debugTextOverlay_.rightLines,
            debugTextOverlay_.bottomLeftLines,
            swapchainExtent_.width,
            swapchainExtent_.height);

        debugTextVertexCount_ = static_cast<std::uint32_t>(vertices.size());
        if (debugTextVertexCount_ > 0)
        {
            std::memcpy(
                debugTextVertexMappedMemory_,
                vertices.data(),
                sizeof(DebugTextVertex) * vertices.size());
        }
    }

    void updateRightDebugText()
    {
        const std::uint64_t processRamMb = getProcessRamUsageMb();
        const std::uint64_t totalRamMb = getTotalRamMb();

        std::wstring vramLine = L"VRAM: N/A";
        if (memoryBudgetSupported_)
        {
            VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{};
            budget.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;

            VkPhysicalDeviceMemoryProperties2 memoryProperties{};
            memoryProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
            memoryProperties.pNext = &budget;
            vkGetPhysicalDeviceMemoryProperties2(physicalDevice_, &memoryProperties);

            for (std::uint32_t i = 0; i < memoryProperties.memoryProperties.memoryHeapCount; ++i)
            {
                if ((memoryProperties.memoryProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0)
                {
                    vramLine = L"VRAM: " +
                        std::to_wstring(static_cast<std::uint64_t>(budget.heapUsage[i] / (1024ull * 1024ull))) +
                        L" / " +
                        std::to_wstring(static_cast<std::uint64_t>(budget.heapBudget[i] / (1024ull * 1024ull))) +
                        L" MB";
                    break;
                }
            }
        }

        std::size_t blockDrawCalls = 0;
        for (const ChunkMesh& mesh : chunkMeshes_)
        {
            blockDrawCalls += mesh.subchunks.size() + mesh.fluidSubchunks.size();
        }
        const std::size_t debugDrawCalls = debugTextVisible_ ? 1 : 0;
        const std::size_t playerDrawCalls = playerVertexCount_ > 0 && playerIndexCount_ > 0 ? 1 : 0;
        const std::size_t crosshairDrawCalls = crosshairVertexCount_ > 0 ? 1 : 0;
        const std::size_t drawCalls = 1 + blockDrawCalls + playerDrawCalls + crosshairDrawCalls + debugDrawCalls;
        const std::size_t totalVertices = 12 + meshVertexCount_ + playerVertexCount_ + crosshairVertexCount_ + debugTextVertexCount_;
        const std::size_t totalIndices = meshIndexCount_ + playerIndexCount_;
        const std::size_t triangles = meshIndexCount_ / 3 + playerIndexCount_ / 3 + 4 + crosshairVertexCount_ / 3 + debugTextVertexCount_ / 3;
        const auto [pendingBuilds, completedBuilds] = chunkBuildQueueSizes();

        debugTextOverlay_.rightLines = {
            versionDebugLine_,
            gpuDebugLine_,
            cpuDebugLine_,
            L"RAM: " + std::to_wstring(processRamMb) + L" / " + std::to_wstring(totalRamMb) + L" MB",
            vramLine,
            apiDebugLine_,
            driverDebugLine_,
            L"DRAWS: " + std::to_wstring(drawCalls),
            L"VISIBLE SUBCHUNKS: " + std::to_wstring(visibleSubchunkDraws_),
            L"CULLED SUBCHUNKS: " + std::to_wstring(culledSubchunkDraws_),
            L"CHUNKS: " + std::to_wstring(chunkStreaming_.loadedChunkCount()),
            L"LOAD RADIUS: " + std::to_wstring(chunkStreaming_.loadRadius()),
            L"UPLOADS/FRAME: " + std::to_wstring(chunkUploadsPerFrame_),
            L"BUILD THREADS: " + std::to_wstring(chunkStreaming_.buildThreadCount()),
            L"BUILD JOBS: " + std::to_wstring(pendingBuilds),
            L"BUILD DONE: " + std::to_wstring(completedBuilds),
            L"DEFERRED DESTROYS: " + std::to_wstring(deferredChunkBufferDestroys_.size()),
            L"DEFERRED UPLOADS: " + std::to_wstring(deferredUploadCleanups_.size()),
            L"VERTS: " + std::to_wstring(totalVertices),
            L"INDICES: " + std::to_wstring(totalIndices),
            L"TRIS: " + std::to_wstring(triangles),
        };
    }

    void updateUniformBuffer()
    {
        SkyUniform uniform{};
        const float aspect = static_cast<float>(swapchainExtent_.width) / static_cast<float>(swapchainExtent_.height);
        const Vec3 renderPosition = renderCameraPosition();
        const Vec3 renderForward = renderCameraForward();
        const auto [renderYaw, renderPitch] = yawPitchFromForward(renderForward);

        uniform.camera[0] = renderYaw;
        uniform.camera[1] = renderPitch;
        uniform.camera[2] = aspect;
        uniform.camera[3] = std::tan(kRenderFovY * 0.5f);

        const Vec3 sunDirection = sunDirectionFromWorldTime(worldTimeTicks_);
        uniform.sunDirection[0] = sunDirection.x;
        uniform.sunDirection[1] = sunDirection.y;
        uniform.sunDirection[2] = sunDirection.z;
        uniform.sunDirection[3] = 0.0f;

        uniform.moonDirection[0] = -uniform.sunDirection[0];
        uniform.moonDirection[1] = -uniform.sunDirection[1];
        uniform.moonDirection[2] = -uniform.sunDirection[2];
        uniform.moonDirection[3] = 0.0f;

        uniform.spriteScale[0] = 0.10f;
        uniform.spriteScale[1] = 0.10f;
        uniform.spriteScale[2] = 0.0f;
        uniform.spriteScale[3] = 0.0f;

        std::memcpy(uniformMappedMemory_, &uniform, sizeof(uniform));

        BlockUniform blockUniform{};
        const Mat4 view = makeViewMatrixFromForward(renderPosition, renderForward);
        const Mat4 projection = makePerspectiveMatrix(kRenderFovY, aspect, kRenderNearPlane, kRenderFarPlane);
        const Mat4 viewProjection = multiply(projection, view);
        currentViewFrustum_ =
            makeViewFrustum(renderPosition, renderForward, kRenderFovY, aspect, kRenderNearPlane, kRenderFarPlane);
        std::memcpy(blockUniform.viewProjection, viewProjection.m, sizeof(blockUniform.viewProjection));
        std::memcpy(blockUniformMappedMemory_, &blockUniform, sizeof(blockUniform));
    }

    Vec3 skyClearColor() const
    {
        const float sunAltitude = sunDirectionFromWorldTime(worldTimeTicks_).y;
        const Vec3 nightColor{0.025f, 0.035f, 0.08f};
        const Vec3 dawnColor{0.78f, 0.43f, 0.24f};
        const Vec3 dayColor{0.46f, 0.72f, 1.0f};

        if (sunAltitude <= -0.08f)
        {
            return nightColor;
        }
        if (sunAltitude < 0.16f)
        {
            const float blend = (sunAltitude + 0.08f) / 0.24f;
            return nightColor * (1.0f - blend) + dawnColor * blend;
        }

        const float blend = std::clamp((sunAltitude - 0.16f) / 0.34f, 0.0f, 1.0f);
        return dawnColor * (1.0f - blend) + dayColor * blend;
    }

    void recordCommandBuffer(VkCommandBuffer commandBuffer, std::uint32_t imageIndex)
    {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to begin recording Vulkan command buffer.");
        }

        std::array<VkClearValue, 2> clearValues{};
        const Vec3 clearColor = skyClearColor();
        clearValues[0].color = {{clearColor.x, clearColor.y, clearColor.z, 1.0f}};
        clearValues[1].depthStencil = {1.0f, 0};

        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass_;
        renderPassInfo.framebuffer = framebuffers_[imageIndex];
        renderPassInfo.renderArea.offset = {0, 0};
        renderPassInfo.renderArea.extent = swapchainExtent_;
        renderPassInfo.clearValueCount = static_cast<std::uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline_);
        vkCmdBindDescriptorSets(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipelineLayout_,
            0,
            1,
            &descriptorSet_,
            0,
            nullptr);
        vkCmdDraw(commandBuffer, 6, 2, 0, 0);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, blockPipeline_);
        vkCmdBindDescriptorSets(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            blockPipelineLayout_,
            0,
            1,
            &blockDescriptorSet_,
            0,
            nullptr);
        std::size_t visibleSubchunkDraws = 0;
        std::size_t culledSubchunkDraws = 0;
        for (const ChunkMesh& mesh : chunkMeshes_)
        {
            if (mesh.indexCount == 0)
            {
                continue;
            }
            VkBuffer vertexBuffers[] = {mesh.vertexBuffer.buffer};
            VkDeviceSize offsets[] = {0};
            vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
            vkCmdBindIndexBuffer(commandBuffer, mesh.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
            for (const SubchunkDraw& draw : mesh.subchunks)
            {
                if (!isSubchunkVisible(currentViewFrustum_, draw))
                {
                    ++culledSubchunkDraws;
                    continue;
                }
                ++visibleSubchunkDraws;
                vkCmdDrawIndexed(
                    commandBuffer,
                    draw.range.indexCount,
                    1,
                    draw.range.firstIndex,
                    draw.range.vertexOffset,
                0);
            }
        }

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, fluidPipeline_);
        vkCmdBindDescriptorSets(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            fluidPipelineLayout_,
            0,
            1,
            &blockDescriptorSet_,
            0,
            nullptr);
        for (const ChunkMesh& mesh : chunkMeshes_)
        {
            if (mesh.fluidIndexCount == 0)
            {
                continue;
            }
            VkBuffer vertexBuffers[] = {mesh.fluidVertexBuffer.buffer};
            VkDeviceSize offsets[] = {0};
            vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
            vkCmdBindIndexBuffer(commandBuffer, mesh.fluidIndexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
            for (const SubchunkDraw& draw : mesh.fluidSubchunks)
            {
                if (!isSubchunkVisible(currentViewFrustum_, draw))
                {
                    ++culledSubchunkDraws;
                    continue;
                }
                ++visibleSubchunkDraws;
                vkCmdDrawIndexed(
                    commandBuffer,
                    draw.range.indexCount,
                    1,
                    draw.range.firstIndex,
                    draw.range.vertexOffset,
                    0);
            }
        }
        visibleSubchunkDraws_ = visibleSubchunkDraws;
        culledSubchunkDraws_ = culledSubchunkDraws;

        if (selectionVertexCount_ > 0)
        {
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, selectionPipeline_);
            vkCmdBindDescriptorSets(
                commandBuffer,
                VK_PIPELINE_BIND_POINT_GRAPHICS,
                selectionPipelineLayout_,
                0,
                1,
                &blockDescriptorSet_,
                0,
                nullptr);
            VkBuffer vertexBuffers[] = {selectionVertexBuffer_.buffer};
            VkDeviceSize offsets[] = {0};
            vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
            vkCmdDraw(commandBuffer, selectionVertexCount_, 1, 0, 0);
        }

        if (playerVertexCount_ > 0 && playerIndexCount_ > 0)
        {
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, playerPipeline_);
            vkCmdBindDescriptorSets(
                commandBuffer,
                VK_PIPELINE_BIND_POINT_GRAPHICS,
                playerPipelineLayout_,
                0,
                1,
                &playerDescriptorSet_,
                0,
                nullptr);
            VkBuffer vertexBuffers[] = {playerVertexBuffer_.buffer};
            VkDeviceSize offsets[] = {0};
            vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
            vkCmdBindIndexBuffer(commandBuffer, playerIndexBuffer_.buffer, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(commandBuffer, playerIndexCount_, 1, 0, 0, 0);
        }

        if (crosshairVertexCount_ > 0)
        {
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, debugTextPipeline_);
            vkCmdBindDescriptorSets(
                commandBuffer,
                VK_PIPELINE_BIND_POINT_GRAPHICS,
                debugTextPipelineLayout_,
                0,
                1,
                &crosshairDescriptorSet_,
                0,
                nullptr);
            VkBuffer vertexBuffers[] = {crosshairVertexBuffer_.buffer};
            VkDeviceSize offsets[] = {0};
            vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
            vkCmdDraw(commandBuffer, crosshairVertexCount_, 1, 0, 0);
        }

        if (debugTextVisible_)
        {
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, debugTextPipeline_);
            vkCmdBindDescriptorSets(
                commandBuffer,
                VK_PIPELINE_BIND_POINT_GRAPHICS,
                debugTextPipelineLayout_,
                0,
                1,
                &debugTextDescriptorSet_,
                0,
                nullptr);
            VkBuffer vertexBuffers[] = {debugTextVertexBuffer_.buffer};
            VkDeviceSize offsets[] = {0};
            vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
            vkCmdDraw(commandBuffer, debugTextVertexCount_, 1, 0, 0);
        }
        vkCmdEndRenderPass(commandBuffer);

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to record Vulkan command buffer.");
        }
    }

    void recreateSwapchain()
    {
        int width = 0;
        int height = 0;
        glfwGetFramebufferSize(window_, &width, &height);
        while (width == 0 || height == 0)
        {
            glfwGetFramebufferSize(window_, &width, &height);
            glfwWaitEvents();
        }

        vkDeviceWaitIdle(device_);
        collectDeferredUploadCleanups(true);
        collectDeferredChunkBufferDestroys(true);
        cleanupSwapchain();
        createSwapchain();
        createRenderPass();
        createDepthResources();
        createGraphicsPipeline();
        createBlockPipeline();
        createFluidPipeline();
        createPlayerPipeline();
        createSelectionPipeline();
        createDebugTextPipeline();
        createFramebuffers();
        updateCrosshairVertices();
    }

    void cleanupSwapchain()
    {
        for (VkFramebuffer framebuffer : framebuffers_)
        {
            vkDestroyFramebuffer(device_, framebuffer, nullptr);
        }
        framebuffers_.clear();

        if (graphicsPipeline_ != VK_NULL_HANDLE)
        {
            vkDestroyPipeline(device_, graphicsPipeline_, nullptr);
            graphicsPipeline_ = VK_NULL_HANDLE;
        }
        if (blockPipeline_ != VK_NULL_HANDLE)
        {
            vkDestroyPipeline(device_, blockPipeline_, nullptr);
            blockPipeline_ = VK_NULL_HANDLE;
        }
        if (fluidPipeline_ != VK_NULL_HANDLE)
        {
            vkDestroyPipeline(device_, fluidPipeline_, nullptr);
            fluidPipeline_ = VK_NULL_HANDLE;
        }
        if (playerPipeline_ != VK_NULL_HANDLE)
        {
            vkDestroyPipeline(device_, playerPipeline_, nullptr);
            playerPipeline_ = VK_NULL_HANDLE;
        }
        if (selectionPipeline_ != VK_NULL_HANDLE)
        {
            vkDestroyPipeline(device_, selectionPipeline_, nullptr);
            selectionPipeline_ = VK_NULL_HANDLE;
        }
        if (debugTextPipeline_ != VK_NULL_HANDLE)
        {
            vkDestroyPipeline(device_, debugTextPipeline_, nullptr);
            debugTextPipeline_ = VK_NULL_HANDLE;
        }

        if (pipelineLayout_ != VK_NULL_HANDLE)
        {
            vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
            pipelineLayout_ = VK_NULL_HANDLE;
        }
        if (blockPipelineLayout_ != VK_NULL_HANDLE)
        {
            vkDestroyPipelineLayout(device_, blockPipelineLayout_, nullptr);
            blockPipelineLayout_ = VK_NULL_HANDLE;
        }
        if (fluidPipelineLayout_ != VK_NULL_HANDLE)
        {
            vkDestroyPipelineLayout(device_, fluidPipelineLayout_, nullptr);
            fluidPipelineLayout_ = VK_NULL_HANDLE;
        }
        if (playerPipelineLayout_ != VK_NULL_HANDLE)
        {
            vkDestroyPipelineLayout(device_, playerPipelineLayout_, nullptr);
            playerPipelineLayout_ = VK_NULL_HANDLE;
        }
        if (selectionPipelineLayout_ != VK_NULL_HANDLE)
        {
            vkDestroyPipelineLayout(device_, selectionPipelineLayout_, nullptr);
            selectionPipelineLayout_ = VK_NULL_HANDLE;
        }
        if (debugTextPipelineLayout_ != VK_NULL_HANDLE)
        {
            vkDestroyPipelineLayout(device_, debugTextPipelineLayout_, nullptr);
            debugTextPipelineLayout_ = VK_NULL_HANDLE;
        }

        if (renderPass_ != VK_NULL_HANDLE)
        {
            vkDestroyRenderPass(device_, renderPass_, nullptr);
            renderPass_ = VK_NULL_HANDLE;
        }

        resourceContext_.destroyTexture(depthTexture_);

        VulkanSwapchainResources resources{};
        resources.swapchain = swapchain_;
        resources.imageFormat = swapchainImageFormat_;
        resources.extent = swapchainExtent_;
        resources.images = std::move(swapchainImages_);
        resources.imageViews = std::move(swapchainImageViews_);
        destroyVulkanSwapchain(device_, resources);
        swapchain_ = resources.swapchain;
        swapchainImageFormat_ = resources.imageFormat;
        swapchainExtent_ = resources.extent;
        swapchainImages_ = std::move(resources.images);
        swapchainImageViews_ = std::move(resources.imageViews);
    }

    void handleMouseMove(double xPos, double yPos)
    {
        if (!cursorCaptured_)
        {
            return;
        }

        if (camera_.firstMouse)
        {
            camera_.lastMouseX = xPos;
            camera_.lastMouseY = yPos;
            camera_.firstMouse = false;
            return;
        }

        const double deltaX = xPos - camera_.lastMouseX;
        const double deltaY = camera_.lastMouseY - yPos;
        camera_.lastMouseX = xPos;
        camera_.lastMouseY = yPos;

        constexpr float sensitivity = 0.0022f;
        camera_.yaw += static_cast<float>(deltaX) * sensitivity;
        camera_.pitch += static_cast<float>(deltaY) * sensitivity;
        camera_.yaw = std::remainder(camera_.yaw, 2.0f * kPi);

        constexpr float pitchLimit = 89.0f * kPi / 180.0f;
        camera_.pitch = std::clamp(camera_.pitch, -pitchLimit, pitchLimit);
    }

    void releaseCursor()
    {
        cursorCaptured_ = false;
        camera_.firstMouse = true;
        glfwSetInputMode(window_, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    }

    void captureCursor()
    {
        cursorCaptured_ = true;
        camera_.firstMouse = true;
        glfwSetInputMode(window_, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    }

    void toggleFullscreen()
    {
        fullscreen_ = !fullscreen_;
        if (fullscreen_)
        {
            glfwGetWindowPos(window_, &windowedX_, &windowedY_);
            glfwGetWindowSize(window_, &windowedWidth_, &windowedHeight_);

            GLFWmonitor* monitor = glfwGetPrimaryMonitor();
            const GLFWvidmode* mode = glfwGetVideoMode(monitor);
            glfwSetWindowMonitor(
                window_,
                monitor,
                0,
                0,
                mode->width,
                mode->height,
                mode->refreshRate);
        }
        else
        {
            glfwSetWindowMonitor(
                window_,
                nullptr,
                windowedX_,
                windowedY_,
                windowedWidth_,
                windowedHeight_,
                0);
        }

        framebufferResized_ = true;
        camera_.firstMouse = true;
    }

    void collectDeferredChunkBufferDestroys(bool force)
    {
        for (auto it = deferredChunkBufferDestroys_.begin(); it != deferredChunkBufferDestroys_.end();)
        {
            if (!force && it->retireFrame > frameCounter_)
            {
                ++it;
                continue;
            }

            resourceContext_.destroyBuffer(it->vertexBuffer);
            resourceContext_.destroyBuffer(it->indexBuffer);
            resourceContext_.destroyBuffer(it->fluidVertexBuffer);
            resourceContext_.destroyBuffer(it->fluidIndexBuffer);
            it = deferredChunkBufferDestroys_.erase(it);
        }
    }

    void collectDeferredUploadCleanups(bool force)
    {
        for (auto it = deferredUploadCleanups_.begin(); it != deferredUploadCleanups_.end();)
        {
            if (!force && it->retireFrame > frameCounter_)
            {
                ++it;
                continue;
            }

            resourceContext_.destroyBuffer(it->stagingBuffer);
            if (it->commandBuffer != VK_NULL_HANDLE)
            {
                vkFreeCommandBuffers(device_, commandPool_, 1, &it->commandBuffer);
                it->commandBuffer = VK_NULL_HANDLE;
            }
            it = deferredUploadCleanups_.erase(it);
        }
    }

    void destroyChunkMeshes(ChunkCoord coord)
    {
        for (auto it = chunkMeshes_.begin(); it != chunkMeshes_.end();)
        {
            if (it->chunkX != coord.x || it->chunkZ != coord.z)
            {
                ++it;
                continue;
            }

            deferredChunkBufferDestroys_.push_back({
                it->vertexBuffer,
                it->indexBuffer,
                it->fluidVertexBuffer,
                it->fluidIndexBuffer,
                frameCounter_ + static_cast<std::uint64_t>(kMaxFramesInFlight),
            });
            meshVertexCount_ -= std::min(
                meshVertexCount_,
                static_cast<std::size_t>(it->vertexCount) + static_cast<std::size_t>(it->fluidVertexCount));
            meshIndexCount_ -= std::min(
                meshIndexCount_,
                static_cast<std::size_t>(it->indexCount) + static_cast<std::size_t>(it->fluidIndexCount));
            it = chunkMeshes_.erase(it);
        }
    }

    void destroyAllChunkMeshes()
    {
        for (ChunkMesh& mesh : chunkMeshes_)
        {
            resourceContext_.destroyBuffer(mesh.vertexBuffer);
            resourceContext_.destroyBuffer(mesh.indexBuffer);
            resourceContext_.destroyBuffer(mesh.fluidVertexBuffer);
            resourceContext_.destroyBuffer(mesh.fluidIndexBuffer);
        }
        chunkMeshes_.clear();
        collectDeferredChunkBufferDestroys(true);
        chunkStreaming_.reset();
        clearChunkDataCache();
        meshVertexCount_ = 0;
        meshIndexCount_ = 0;
    }

    void clearChunkDataCache()
    {
        {
            std::lock_guard<std::mutex> lock(chunkDataCacheMutex_);
            chunkDataCache_.clear();
            chunkDataLoadsInProgress_.clear();
        }
        chunkDataCacheCv_.notify_all();
    }


};
}

int main()
{
    HRESULT comResult = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    if (FAILED(comResult))
    {
        std::cerr << "Failed to initialize COM for texture loading.\n";
        return EXIT_FAILURE;
    }

    try
    {
        VulkanVoxelApp app;
        app.run();
    }
    catch (const std::exception& exception)
    {
        std::cerr << exception.what() << '\n';
        CoUninitialize();
        return EXIT_FAILURE;
    }

    CoUninitialize();
    return EXIT_SUCCESS;
}
