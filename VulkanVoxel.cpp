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
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kWindowWidth = 1280;
constexpr int kWindowHeight = 720;
constexpr int kMaxFramesInFlight = 2;
constexpr std::size_t kMaxOverlayVertexCount = 1048576;
constexpr float kMoveSpeed = 60.0f;
constexpr float kMouseSensitivity = 0.08f;
constexpr float kPi = 3.14159265358979323846f;
constexpr const char* kWindowTitle = "VulkanVoxel";
constexpr float kOverlayPixelSize = 3.0f;
constexpr float kOverlayMargin = 16.0f;
constexpr float kOverlayLineGap = 8.0f;

using GlyphRows = std::array<std::uint8_t, 7>;

GlyphRows GetGlyph(char c) {
    switch (c) {
    case 'A': return {0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11};
    case 'B': return {0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E};
    case 'C': return {0x0F, 0x10, 0x10, 0x10, 0x10, 0x10, 0x0F};
    case 'D': return {0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E};
    case 'E': return {0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F};
    case 'F': return {0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10};
    case 'G': return {0x0F, 0x10, 0x10, 0x13, 0x11, 0x11, 0x0F};
    case 'H': return {0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11};
    case 'I': return {0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E};
    case 'J': return {0x01, 0x01, 0x01, 0x01, 0x11, 0x11, 0x0E};
    case 'K': return {0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11};
    case 'L': return {0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F};
    case 'M': return {0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11};
    case 'N': return {0x11, 0x11, 0x19, 0x15, 0x13, 0x11, 0x11};
    case 'O': return {0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E};
    case 'P': return {0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10};
    case 'Q': return {0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D};
    case 'R': return {0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11};
    case 'S': return {0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E};
    case 'T': return {0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04};
    case 'U': return {0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E};
    case 'V': return {0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04};
    case 'W': return {0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0A};
    case 'X': return {0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11};
    case 'Y': return {0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04};
    case 'Z': return {0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F};
    case '0': return {0x1F, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1F};
    case '1': return {0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E};
    case '2': return {0x1F, 0x01, 0x01, 0x1F, 0x10, 0x10, 0x1F};
    case '3': return {0x1F, 0x01, 0x01, 0x0F, 0x01, 0x01, 0x1F};
    case '4': return {0x11, 0x11, 0x11, 0x1F, 0x01, 0x01, 0x01};
    case '5': return {0x1F, 0x10, 0x10, 0x1F, 0x01, 0x01, 0x1F};
    case '6': return {0x1F, 0x10, 0x10, 0x1F, 0x11, 0x11, 0x1F};
    case '7': return {0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08};
    case '8': return {0x1F, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x1F};
    case '9': return {0x1F, 0x11, 0x11, 0x1F, 0x01, 0x01, 0x1F};
    case '[': return {0x0E, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0E};
    case ']': return {0x0E, 0x02, 0x02, 0x02, 0x02, 0x02, 0x0E};
    case '/': return {0x01, 0x01, 0x02, 0x04, 0x08, 0x10, 0x10};
    case '-': return {0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00};
    case '.': return {0x00, 0x00, 0x00, 0x00, 0x00, 0x06, 0x06};
    case ':': return {0x00, 0x04, 0x04, 0x00, 0x04, 0x04, 0x00};
    case ' ': return {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
    default:  return {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
    }
}

bool IsSupportedOverlayChar(char c) {
    return (c >= 'A' && c <= 'Z') ||
           (c >= '0' && c <= '9') ||
           c == ' ' || c == ':' || c == '.' || c == '[' || c == ']' || c == '/' || c == '-';
}

std::string SanitizeOverlayText(const std::string& text) {
    std::string result;
    result.reserve(text.size());

    for (unsigned char c : text) {
        const char upper = static_cast<char>(std::toupper(c));
        result.push_back(IsSupportedOverlayChar(upper) ? upper : ' ');
    }

    return result;
}

float GetOverlayTextWidth(const std::string& text, float pixelSize) {
    const float glyphWidth = 5.0f * pixelSize;
    const float glyphSpacing = pixelSize;
    if (text.empty()) {
        return 0.0f;
    }

    return static_cast<float>(text.size()) * glyphWidth +
           static_cast<float>(text.size() - 1) * glyphSpacing;
}

void AppendOverlayQuad(
    std::vector<OverlayVertex>& vertices,
    float leftPixels,
    float topPixels,
    float rightPixels,
    float bottomPixels,
    float red,
    float green,
    float blue,
    const VkExtent2D& extent
) {
    const float left = (leftPixels / static_cast<float>(extent.width)) * 2.0f - 1.0f;
    const float right = (rightPixels / static_cast<float>(extent.width)) * 2.0f - 1.0f;
    const float top = (topPixels / static_cast<float>(extent.height)) * 2.0f - 1.0f;
    const float bottom = (bottomPixels / static_cast<float>(extent.height)) * 2.0f - 1.0f;

    const OverlayVertex v0{{left, top}, {red, green, blue}};
    const OverlayVertex v1{{right, top}, {red, green, blue}};
    const OverlayVertex v2{{right, bottom}, {red, green, blue}};
    const OverlayVertex v3{{left, bottom}, {red, green, blue}};

    vertices.push_back(v0);
    vertices.push_back(v1);
    vertices.push_back(v2);
    vertices.push_back(v0);
    vertices.push_back(v2);
    vertices.push_back(v3);
}

void AppendGlyph(
    std::vector<OverlayVertex>& vertices,
    char c,
    float startX,
    float startY,
    float pixelSize,
    float red,
    float green,
    float blue,
    const VkExtent2D& extent
) {
    const GlyphRows rows = GetGlyph(c);

    for (std::size_t y = 0; y < rows.size(); ++y) {
        for (int x = 0; x < 5; ++x) {
            const std::uint8_t mask = static_cast<std::uint8_t>(1u << (4 - x));
            if ((rows[y] & mask) == 0) {
                continue;
            }

            const float left = startX + static_cast<float>(x) * pixelSize;
            const float top = startY + static_cast<float>(y) * pixelSize;
            AppendOverlayQuad(
                vertices,
                left,
                top,
                left + pixelSize,
                top + pixelSize,
                red,
                green,
                blue,
                extent
            );
        }
    }
}

void AppendOutlinedGlyph(
    std::vector<OverlayVertex>& vertices,
    char c,
    float startX,
    float startY,
    float pixelSize,
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
            c,
            startX + offset.x * pixelSize,
            startY + offset.y * pixelSize,
            pixelSize,
            0.0f,
            0.0f,
            0.0f,
            extent
        );
    }

    AppendGlyph(vertices, c, startX, startY, pixelSize, 1.0f, 1.0f, 1.0f, extent);
}

void AppendOutlinedText(
    std::vector<OverlayVertex>& vertices,
    const std::string& text,
    float startX,
    float startY,
    float pixelSize,
    const VkExtent2D& extent
) {
    const float glyphWidth = 5.0f * pixelSize;
    const float glyphSpacing = pixelSize;
    float cursorX = startX;

    for (char c : text) {
        AppendOutlinedGlyph(vertices, c, cursorX, startY, pixelSize, extent);
        cursorX += glyphWidth + glyphSpacing;
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
    CreateInstance();
    CreateSurface();
    PickPhysicalDevice();
    CreateLogicalDevice();
    CreateSwapChain();
    LoadStaticDebugInfo();
    RefreshSystemUsageStats();
    CreateImageViews();
    CreateCommandPool();
    CreateDescriptorSetLayout();
    CreateRenderPass();
    CreateDepthResources();
    CreateTextureImage();
    CreateTextureImageView();
    CreateTextureSampler();
    BuildWorldMesh();
    CreateOverlayBuffer();
    CreateUniformBuffers();
    CreateDescriptorPool();
    CreateDescriptorSets();
    CreatePipelines();
    CreateFramebuffers();
    CreateCommandBuffers();
    CreateSyncObjects();
}

void VulkanVoxelApp::MainLoop() {
    auto lastFrameTime = std::chrono::steady_clock::now();

    while (glfwWindowShouldClose(window_) == GLFW_FALSE) {
        const auto now = std::chrono::steady_clock::now();
        const float deltaTime = std::chrono::duration<float>(now - lastFrameTime).count();
        lastFrameTime = now;

        glfwPollEvents();
        ProcessInput(deltaTime);
        UpdateOverlayText(deltaTime);
        DrawFrame();
    }

    if (device_ != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device_);
    }
}

void VulkanVoxelApp::Cleanup() {
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
        if (worldVertexBuffer_ != VK_NULL_HANDLE) {
            vkDestroyBuffer(device_, worldVertexBuffer_, nullptr);
        }
        if (worldVertexBufferMemory_ != VK_NULL_HANDLE) {
            vkFreeMemory(device_, worldVertexBufferMemory_, nullptr);
        }

        if (textureSampler_ != VK_NULL_HANDLE) {
            vkDestroySampler(device_, textureSampler_, nullptr);
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

    if (overlayPipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_, overlayPipeline_, nullptr);
        overlayPipeline_ = VK_NULL_HANDLE;
    }
    if (overlayPipelineLayout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device_, overlayPipelineLayout_, nullptr);
        overlayPipelineLayout_ = VK_NULL_HANDLE;
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
    if (isF3Pressed && !previousF3Pressed_) {
        debugOverlayEnabled_ = !debugOverlayEnabled_;
        RebuildOverlayVertices();
        overlayDirty_ = true;
    }
    previousF3Pressed_ = isF3Pressed;

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
    cameraPitch_ = std::clamp(cameraPitch_, -89.0f, 89.0f);

    Vec3 movement{};
    if (glfwGetKey(window_, GLFW_KEY_W) == GLFW_PRESS) {
        movement = movement + GetHorizontalForwardVector();
    }
    if (glfwGetKey(window_, GLFW_KEY_S) == GLFW_PRESS) {
        movement = movement - GetHorizontalForwardVector();
    }
    if (glfwGetKey(window_, GLFW_KEY_D) == GLFW_PRESS) {
        movement = movement + GetRightVector();
    }
    if (glfwGetKey(window_, GLFW_KEY_A) == GLFW_PRESS) {
        movement = movement - GetRightVector();
    }
    if (glfwGetKey(window_, GLFW_KEY_SPACE) == GLFW_PRESS) {
        movement.y += 1.0f;
    }
    if (glfwGetKey(window_, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS ||
        glfwGetKey(window_, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS) {
        movement.y -= 1.0f;
    }

    if (Length(movement) > 0.0f) {
        movement = Normalize(movement);
        cameraPosition_ = cameraPosition_ + movement * (kMoveSpeed * deltaTime);
    }

    cameraPosition_.x = std::clamp(cameraPosition_.x, 0.0f, static_cast<float>(kWorldSizeX));
    cameraPosition_.y = std::clamp(cameraPosition_.y, 1.0f, static_cast<float>(kWorldSizeY - 1));
    cameraPosition_.z = std::clamp(cameraPosition_.z, 0.0f, static_cast<float>(kWorldSizeZ));
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

void VulkanVoxelApp::UpdateOverlayText(float deltaTime) {
    ++frameCounter_;
    fpsAccumulatorSeconds_ += deltaTime;
    overlayRefreshAccumulatorSeconds_ += deltaTime;
    bool statsUpdated = false;

    if (fpsAccumulatorSeconds_ >= 0.5) {
        currentFps_ = static_cast<std::uint32_t>(
            static_cast<double>(frameCounter_) / fpsAccumulatorSeconds_ + 0.5
        );
        currentFrameTimeMs_ = currentFps_ > 0 ? 1000.0 / static_cast<double>(currentFps_) : 0.0;
        frameCounter_ = 0;
        fpsAccumulatorSeconds_ = 0.0;
        RefreshSystemUsageStats();
        statsUpdated = true;
    }

    if (!debugOverlayEnabled_) {
        if (overlayVertexCount_ != 0) {
            overlayVertices_.clear();
            overlayVertexCount_ = 0;
            overlayDirty_ = true;
        }
        return;
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

    if (!debugOverlayEnabled_) {
        overlayVertexCount_ = 0;
        return;
    }

    const int chunkX = std::clamp(static_cast<int>(cameraPosition_.x) / kChunkSizeX, 0, kWorldChunkCountX - 1);
    const int chunkZ = std::clamp(static_cast<int>(cameraPosition_.z) / kChunkSizeZ, 0, kWorldChunkCountZ - 1);
    const int subChunk = std::clamp(static_cast<int>(cameraPosition_.y) / kSubChunkSize, 0, kSubChunkCountY - 1);
    const Vec3 forward = GetForwardVector();
    const std::uint32_t faceCount = worldVertexCount_ / 6;
    const std::uint32_t triangleCount = worldVertexCount_ / 3;
    std::vector<std::string> leftLines;
    std::vector<std::string> rightLines;

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
        "LOOK: X " + FormatFloat(forward.x, 2) +
        " Y " + FormatFloat(forward.y, 2) +
        " Z " + FormatFloat(forward.z, 2)
    );
    leftLines.push_back(
        "CHUNK: X " + std::to_string(chunkX) +
        " Z " + std::to_string(chunkZ) +
        " SUB: " + std::to_string(subChunk)
    );
    leftLines.push_back("LOADED CHUNKS: " + std::to_string(world_.GetLoadedChunkCount()));
    leftLines.push_back("FACE COUNT: " + std::to_string(faceCount));

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
    rightLines.push_back("API: " + apiVersionString_);
    rightLines.push_back("DRIVER: " + driverVersionString_);
    rightLines.push_back("RENDERER: " + rendererName_);
    rightLines.push_back("PRESENT: " + presentModeString_);
    rightLines.push_back("DRAW CALLS: " + std::to_string(drawCallCount_));
    rightLines.push_back("VERTEX COUNT: " + std::to_string(worldVertexCount_));
    rightLines.push_back("TRIANGLE COUNT: " + std::to_string(triangleCount));

    float leftY = kOverlayMargin;
    for (const std::string& rawLine : leftLines) {
        const std::string line = SanitizeOverlayText(rawLine);
        AppendOutlinedText(
            overlayVertices_,
            line,
            kOverlayMargin,
            leftY,
            kOverlayPixelSize,
            swapChainExtent_
        );
        leftY += 7.0f * kOverlayPixelSize + kOverlayLineGap;
    }

    float rightY = kOverlayMargin;
    for (const std::string& rawLine : rightLines) {
        const std::string line = SanitizeOverlayText(rawLine);
        const float startX = static_cast<float>(swapChainExtent_.width) -
                             kOverlayMargin -
                             GetOverlayTextWidth(line, kOverlayPixelSize);
        AppendOutlinedText(
            overlayVertices_,
            line,
            startX,
            rightY,
            kOverlayPixelSize,
            swapChainExtent_
        );
        rightY += 7.0f * kOverlayPixelSize + kOverlayLineGap;
    }

    if (overlayVertices_.size() > kMaxOverlayVertexCount) {
        throw std::runtime_error("Overlay vertex buffer is too small.");
    }

    overlayVertexCount_ = static_cast<std::uint32_t>(overlayVertices_.size());
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

int main() {
    VulkanVoxelApp app;

    try {
        return app.Run();
    } catch (const std::exception& e) {
        OutputDebugStringA(e.what());
        OutputDebugStringA("\n");
        std::cerr << e.what() << '\n';
        return 1;
    }
}
