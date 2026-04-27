#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define NOMINMAX
#include <Windows.h>
#include <psapi.h>
#include <wincodec.h>
#include <wrl/client.h>

#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstddef>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <fstream>
#include <iomanip>
#include <intrin.h>
#include <iostream>
#include <iterator>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace
{
constexpr std::uint32_t kWindowWidth = 1280;
constexpr std::uint32_t kWindowHeight = 720;
constexpr int kMaxFramesInFlight = 2;
constexpr float kPi = 3.14159265358979323846f;
constexpr int kChunkSizeX = 16;
constexpr int kChunkSizeZ = 16;
constexpr int kSubchunkSize = 16;
constexpr int kChunkHeight = 512;
constexpr int kSubchunksPerChunk = kChunkHeight / kSubchunkSize;
constexpr int kTerrainBaseHeight = 192;
constexpr int kTerrainHeightRange = 40;
constexpr int kDefaultChunkLoadRadius = 5;
constexpr int kMaxChunkLoadRadius = 64;
constexpr int kDefaultChunkUploadsPerFrame = 1;
constexpr int kMaxChunkUploadsPerFrame = 64;
constexpr int kDefaultChunkBuildThreads = 4;
constexpr int kMaxChunkBuildThreads = 16;
constexpr std::size_t kBlockIdCount =
    static_cast<std::size_t>(std::numeric_limits<std::uint16_t>::max()) + 1;
constexpr std::uint16_t kAirBlockId = 0;
constexpr std::uint16_t kRockBlockId = 1;
constexpr std::uint16_t kDirtBlockId = 2;
constexpr std::uint16_t kGrassBlockId = 3;
constexpr std::uint16_t kBedrockBlockId = std::numeric_limits<std::uint16_t>::max();
constexpr int kDebugGlyphFirst = 32;
constexpr int kDebugGlyphLast = 126;
constexpr int kDebugGlyphCount = kDebugGlyphLast - kDebugGlyphFirst + 1;
constexpr int kDebugGlyphColumns = 16;
constexpr int kDebugGlyphCellWidth = 64;
constexpr int kDebugGlyphCellHeight = 64;
constexpr std::uint32_t kDebugFontAtlasWidth = kDebugGlyphColumns * kDebugGlyphCellWidth;
constexpr std::uint32_t kDebugFontAtlasHeight =
    ((kDebugGlyphCount + kDebugGlyphColumns - 1) / kDebugGlyphColumns) * kDebugGlyphCellHeight;
constexpr std::size_t kMaxDebugTextVertices = 48000;

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

struct ImagePixels
{
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<std::uint8_t> rgba;
};

struct QueueFamilyIndices
{
    std::optional<std::uint32_t> graphicsFamily;
    std::optional<std::uint32_t> presentFamily;

    bool isComplete() const
    {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

struct SwapchainSupportDetails
{
    VkSurfaceCapabilitiesKHR capabilities{};
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

struct Texture
{
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
};

struct Buffer
{
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
};

struct BlockVertex
{
    float position[3];
    float uv[2];
    float ao = 1.0f;
    float textureLayer = 0.0f;
};

struct FaceMaskCell
{
    bool filled = false;
    std::uint32_t textureLayer = 0;
    std::array<std::uint8_t, 4> ao{};

    bool operator==(const FaceMaskCell& other) const
    {
        return filled == other.filled && (!filled || (textureLayer == other.textureLayer && ao == other.ao));
    }
};

enum class BlockFace : std::uint8_t
{
    Top = 0,
    Side = 1,
    Bottom = 2,
};

struct BlockDefinition
{
    std::uint16_t id = kAirBlockId;
    std::string name;
    bool solid = false;
    std::array<std::uint32_t, 3> textureLayers{};
};

struct ChunkColumnData
{
    std::array<int, kChunkSizeX * kChunkSizeZ> highestSolidY{};
};

struct MeshRange
{
    std::uint32_t vertexCount = 0;
    std::uint32_t firstIndex = 0;
    std::uint32_t indexCount = 0;
    std::int32_t vertexOffset = 0;
};

struct SubchunkDraw
{
    int chunkX = 0;
    int chunkZ = 0;
    int subchunkY = 0;
    MeshRange range{};
};

struct ChunkMesh
{
    Buffer vertexBuffer;
    Buffer indexBuffer;
    std::uint32_t vertexCount = 0;
    std::uint32_t indexCount = 0;
    int chunkX = 0;
    int chunkZ = 0;
    std::vector<SubchunkDraw> subchunks;
};

struct DeferredChunkBufferDestroy
{
    Buffer vertexBuffer;
    Buffer indexBuffer;
    std::uint64_t retireFrame = 0;
};

struct DeferredUploadCleanup
{
    Buffer stagingBuffer;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    std::uint64_t retireFrame = 0;
};

struct ChunkCoord
{
    int x = 0;
    int z = 0;

    bool operator==(ChunkCoord other) const
    {
        return x == other.x && z == other.z;
    }
};

struct ChunkCoordHash
{
    std::size_t operator()(ChunkCoord coord) const
    {
        const std::uint64_t x = static_cast<std::uint32_t>(coord.x);
        const std::uint64_t z = static_cast<std::uint32_t>(coord.z);
        return static_cast<std::size_t>((x << 32) ^ z);
    }
};

struct ChunkBuildResult
{
    ChunkCoord coord{};
    std::vector<BlockVertex> vertices;
    std::vector<std::uint32_t> indices;
    std::vector<SubchunkDraw> subchunks;
};

struct ChunkBuildRequest
{
    ChunkCoord coord{};
    int subchunkY = 0;
    std::int64_t priorityDistanceSq = 0;
    std::uint64_t generation = 0;
};

struct SubchunkBuildResult
{
    ChunkCoord coord{};
    int subchunkY = 0;
    std::uint64_t generation = 0;
    std::vector<BlockVertex> vertices;
    std::vector<std::uint32_t> indices;
};

struct PendingChunkMesh
{
    std::uint64_t generation = 0;
    int completedSubchunks = 0;
    std::array<bool, kSubchunksPerChunk> received{};
    std::array<std::vector<BlockVertex>, kSubchunksPerChunk> vertices;
    std::array<std::vector<std::uint32_t>, kSubchunksPerChunk> indices;
};

struct ChunkBuildRequestPriority
{
    bool operator()(const ChunkBuildRequest& lhs, const ChunkBuildRequest& rhs) const
    {
        if (lhs.priorityDistanceSq != rhs.priorityDistanceSq)
        {
            return lhs.priorityDistanceSq > rhs.priorityDistanceSq;
        }
        return lhs.generation < rhs.generation;
    }
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

struct CameraState
{
    float yaw = 0.0f;   // +X is east, so the initial view looks east.
    float pitch = 0.0f;
    Vec3 position{80.0f, 310.0f, 80.0f};
    bool firstMouse = true;
    double lastMouseX = 0.0;
    double lastMouseY = 0.0;
};

struct DebugTextOverlay
{
    std::chrono::steady_clock::time_point lastUpdate = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point rightLastUpdate = std::chrono::steady_clock::now();
    std::uint32_t framesSinceUpdate = 0;
    std::vector<std::wstring> lines = {
        L"FPS: 0000 [00.000MS]",
        L"POS: 0000.00 / 0000.00 / 0000.00",
        L"CAM: YAW +000.000 PIT +00.000",
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
    std::vector<std::wstring> bottomLeftLines = {
        L"CPU FRAME: 00.000MS",
        L"FENCE WAIT: 00.000MS",
        L"ACQUIRE: 00.000MS",
        L"LOAD UPDATE: 00.000MS",
        L"SUBCHUNK DONE: 00.000MS",
        L"CHUNK UPLOAD: 00.000MS",
        L"DEBUG TEXT: 00.000MS",
        L"RECORD: 00.000MS",
        L"SUBMIT+PRESENT: 00.000MS",
    };
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

struct ChunkLoadUpdateStats
{
    std::size_t loaded = 0;
    std::size_t unloaded = 0;
    std::size_t queued = 0;
};

struct DebugGlyph
{
    float u0 = 0.0f;
    float v0 = 0.0f;
    float u1 = 0.0f;
    float v1 = 0.0f;
    float advance = 0.0f;
    float width = 0.0f;
    float height = 0.0f;
};

struct DebugTextVertex
{
    float position[2];
    float uv[2];
    float color[4];
};

void glfwErrorCallback(int errorCode, const char* description)
{
    std::cerr << "GLFW error " << errorCode << ": " << description << '\n';
}

std::vector<char> readBinaryFile(const std::string& path)
{
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + path);
    }

    const std::streamsize fileSize = file.tellg();
    std::vector<char> buffer(static_cast<std::size_t>(fileSize));
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    return buffer;
}

std::uint32_t findMemoryType(
    VkPhysicalDevice physicalDevice,
    std::uint32_t typeFilter,
    VkMemoryPropertyFlags properties)
{
    VkPhysicalDeviceMemoryProperties memoryProperties{};
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

    for (std::uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i)
    {
        if ((typeFilter & (1u << i)) != 0 &&
            (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties)
        {
            return i;
        }
    }

    throw std::runtime_error("Failed to find a suitable Vulkan memory type.");
}

ImagePixels loadPngWithWic(const std::wstring& path)
{
    ComPtr<IWICImagingFactory> factory;
    HRESULT result = CoCreateInstance(
        CLSID_WICImagingFactory,
        nullptr,
        CLSCTX_INPROC_SERVER,
        IID_PPV_ARGS(&factory));
    if (FAILED(result))
    {
        throw std::runtime_error("Failed to create WIC imaging factory.");
    }

    ComPtr<IWICBitmapDecoder> decoder;
    result = factory->CreateDecoderFromFilename(
        path.c_str(),
        nullptr,
        GENERIC_READ,
        WICDecodeMetadataCacheOnLoad,
        &decoder);
    if (FAILED(result))
    {
        throw std::runtime_error("Failed to open PNG texture.");
    }

    ComPtr<IWICBitmapFrameDecode> frame;
    result = decoder->GetFrame(0, &frame);
    if (FAILED(result))
    {
        throw std::runtime_error("Failed to decode PNG frame.");
    }

    ComPtr<IWICFormatConverter> converter;
    result = factory->CreateFormatConverter(&converter);
    if (FAILED(result))
    {
        throw std::runtime_error("Failed to create WIC format converter.");
    }

    result = converter->Initialize(
        frame.Get(),
        GUID_WICPixelFormat32bppRGBA,
        WICBitmapDitherTypeNone,
        nullptr,
        0.0,
        WICBitmapPaletteTypeCustom);
    if (FAILED(result))
    {
        throw std::runtime_error("Failed to convert PNG to RGBA.");
    }

    ImagePixels image{};
    result = converter->GetSize(&image.width, &image.height);
    if (FAILED(result) || image.width == 0 || image.height == 0)
    {
        throw std::runtime_error("PNG texture has an invalid size.");
    }

    const std::uint32_t stride = image.width * 4;
    image.rgba.resize(static_cast<std::size_t>(stride) * image.height);
    result = converter->CopyPixels(nullptr, stride, static_cast<UINT>(image.rgba.size()), image.rgba.data());
    if (FAILED(result))
    {
        throw std::runtime_error("Failed to copy PNG pixels.");
    }

    return image;
}

std::wstring sourcePathWide(const wchar_t* relativePath)
{
    return std::wstring(VULKAN_VOXEL_SOURCE_DIR_WIDE) + relativePath;
}

std::wstring sourcePathWide(const std::wstring& relativePath)
{
    return std::wstring(VULKAN_VOXEL_SOURCE_DIR_WIDE) + relativePath;
}

std::string sourcePath(const char* relativePath)
{
    return std::string(VULKAN_VOXEL_SOURCE_DIR) + relativePath;
}

std::wstring asciiToWide(const std::string& text)
{
    return std::wstring(text.begin(), text.end());
}

bool fileExists(const std::wstring& path)
{
    const DWORD attributes = GetFileAttributesW(path.c_str());
    return attributes != INVALID_FILE_ATTRIBUTES && (attributes & FILE_ATTRIBUTE_DIRECTORY) == 0;
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

int chunkCoordForWorldPosition(float position, int chunkSize)
{
    return floorDiv(static_cast<int>(std::floor(position)), chunkSize);
}

std::int64_t chunkDistanceSq(ChunkCoord coord, int centerChunkX, int centerChunkZ)
{
    const std::int64_t dx = static_cast<std::int64_t>(coord.x) - centerChunkX;
    const std::int64_t dz = static_cast<std::int64_t>(coord.z) - centerChunkZ;
    return dx * dx + dz * dz;
}

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
    const float length = std::sqrt(dot(value, value));
    if (length <= 0.00001f)
    {
        return {};
    }

    return value * (1.0f / length);
}

std::uint8_t vertexAo(bool side1, bool side2, bool corner)
{
    if (side1 && side2)
    {
        return 0;
    }

    return static_cast<std::uint8_t>(
        3 - static_cast<int>(side1) - static_cast<int>(side2) - static_cast<int>(corner));
}

float aoFactor(std::uint8_t ao)
{
    return 0.55f + 0.15f * static_cast<float>(ao);
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

std::wstring formatFixedWidth(double value, int width, int precision)
{
    std::wostringstream stream;
    stream << std::fixed << std::setprecision(precision) << std::setw(width) << std::setfill(L'0') << value;
    return stream.str();
}

std::wstring versionString(std::uint32_t version)
{
    std::wostringstream stream;
    stream << VK_VERSION_MAJOR(version) << L'.'
           << VK_VERSION_MINOR(version) << L'.'
           << VK_VERSION_PATCH(version);
    return stream.str();
}

std::wstring narrowToWide(const char* text)
{
    if (text == nullptr)
    {
        return L"";
    }

    const int length = MultiByteToWideChar(CP_UTF8, 0, text, -1, nullptr, 0);
    if (length <= 1)
    {
        return L"";
    }

    std::wstring result(static_cast<std::size_t>(length), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text, -1, result.data(), length);
    result.pop_back();
    return result;
}

std::wstring getCpuBrandString()
{
    int cpuInfo[4] = {};
    __cpuid(cpuInfo, 0x80000000);
    const unsigned int maxExtendedId = static_cast<unsigned int>(cpuInfo[0]);
    if (maxExtendedId < 0x80000004)
    {
        return L"N/A";
    }

    char brand[49] = {};
    for (unsigned int id = 0; id < 3; ++id)
    {
        int brandInfo[4] = {};
        __cpuid(brandInfo, static_cast<int>(0x80000002 + id));
        std::memcpy(brand + id * 16, brandInfo, sizeof(brandInfo));
    }

    std::string trimmed = brand;
    trimmed.erase(trimmed.begin(), std::find_if(trimmed.begin(), trimmed.end(), [](unsigned char ch)
    {
        return !std::isspace(ch);
    }));
    trimmed.erase(std::find_if(trimmed.rbegin(), trimmed.rend(), [](unsigned char ch)
    {
        return !std::isspace(ch);
    }).base(), trimmed.end());

    return narrowToWide(trimmed.c_str());
}

class VulkanVoxelApp
{
public:
    void run()
    {
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
    VkPipelineLayout debugTextPipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline graphicsPipeline_ = VK_NULL_HANDLE;
    VkPipeline blockPipeline_ = VK_NULL_HANDLE;
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
    Texture debugFontAtlasTexture_{};
    VkSampler textureSampler_ = VK_NULL_HANDLE;
    Buffer uniformBuffer_{};
    Buffer blockUniformBuffer_{};
    Buffer debugTextVertexBuffer_{};
    void* uniformMappedMemory_ = nullptr;
    void* blockUniformMappedMemory_ = nullptr;
    void* debugTextVertexMappedMemory_ = nullptr;
    std::uint32_t debugTextVertexCount_ = 0;
    std::vector<ChunkMesh> chunkMeshes_;
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet_ = VK_NULL_HANDLE;
    VkDescriptorSet blockDescriptorSet_ = VK_NULL_HANDLE;
    VkDescriptorSet debugTextDescriptorSet_ = VK_NULL_HANDLE;
    CameraState camera_{};
    DebugTextOverlay debugTextOverlay_{};
    FrameProfiler frameProfiler_{};
    bool debugTextVisible_ = true;
    int chunkLoadRadius_ = kDefaultChunkLoadRadius;
    int chunkUploadsPerFrame_ = kDefaultChunkUploadsPerFrame;
    int chunkBuildThreads_ = kDefaultChunkBuildThreads;
    int loadedCenterChunkX_ = std::numeric_limits<int>::min();
    int loadedCenterChunkZ_ = std::numeric_limits<int>::min();
    std::vector<BlockDefinition> blockDefinitions_;
    std::array<int, kBlockIdCount> blockDefinitionIndices_{};
    std::vector<std::wstring> blockTextureLayerNames_;
    std::unordered_set<ChunkCoord, ChunkCoordHash> desiredChunks_;
    std::unordered_set<ChunkCoord, ChunkCoordHash> loadedChunks_;
    std::unordered_set<ChunkCoord, ChunkCoordHash> queuedChunks_;
    std::vector<std::thread> chunkBuildWorkers_;
    std::mutex chunkBuildMutex_;
    std::condition_variable chunkBuildCv_;
    std::vector<ChunkBuildRequest> pendingChunkBuilds_;
    std::deque<SubchunkBuildResult> completedSubchunkBuilds_;
    std::deque<ChunkBuildResult> completedChunkBuilds_;
    std::atomic_bool chunkBuildWorkerRunning_ = false;
    std::uint64_t chunkBuildGeneration_ = 0;
    int chunkBuildPriorityCenterX_ = 0;
    int chunkBuildPriorityCenterZ_ = 0;
    std::unordered_map<ChunkCoord, std::uint64_t, ChunkCoordHash> queuedChunkGenerations_;
    std::unordered_map<ChunkCoord, PendingChunkMesh, ChunkCoordHash> pendingChunkMeshes_;
    std::vector<DeferredChunkBufferDestroy> deferredChunkBufferDestroys_;
    std::vector<DeferredUploadCleanup> deferredUploadCleanups_;
    std::size_t meshVertexCount_ = 0;
    std::size_t meshIndexCount_ = 0;
    std::array<DebugGlyph, kDebugGlyphCount> debugGlyphs_{};
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
        }
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
        createSwapchain();
        createImageViews();
        createRenderPass();
        createDepthResources();
        createDescriptorSetLayout();
        createGraphicsPipeline();
        createBlockPipeline();
        createDebugTextPipeline();
        createFramebuffers();
        createCommandPool();
        createSkyTextures();
        loadBlockDefinitions();
        createBlockTextureArray();
        loadWorldConfig();
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
        if (device_ != VK_NULL_HANDLE)
        {
            vkDeviceWaitIdle(device_);
            collectDeferredUploadCleanups(true);
            collectDeferredChunkBufferDestroys(true);
        }

        cleanupSwapchain();

        destroyBuffer(uniformBuffer_);
        destroyBuffer(blockUniformBuffer_);
        destroyBuffer(debugTextVertexBuffer_);
        destroyAllChunkMeshes();

        if (descriptorPool_ != VK_NULL_HANDLE)
        {
            vkDestroyDescriptorPool(device_, descriptorPool_, nullptr);
        }

        if (textureSampler_ != VK_NULL_HANDLE)
        {
            vkDestroySampler(device_, textureSampler_, nullptr);
        }

        destroyTexture(sunTexture_);
        destroyTexture(moonTexture_);
        destroyTexture(blockTextureArray_);
        destroyTexture(debugFontAtlasTexture_);
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
            SwapchainSupportDetails swapchainSupport = querySwapchainSupport(device);
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

    SwapchainSupportDetails querySwapchainSupport(VkPhysicalDevice device) const
    {
        SwapchainSupportDetails details{};
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface_, &details.capabilities);

        std::uint32_t formatCount = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface_, &formatCount, nullptr);
        if (formatCount != 0)
        {
            details.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface_, &formatCount, details.formats.data());
        }

        std::uint32_t presentModeCount = 0;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface_, &presentModeCount, nullptr);
        if (presentModeCount != 0)
        {
            details.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface_, &presentModeCount, details.presentModes.data());
        }

        return details;
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
        SwapchainSupportDetails support = querySwapchainSupport(physicalDevice_);
        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(support.formats);
        VkPresentModeKHR presentMode = chooseSwapPresentMode(support.presentModes);
        VkExtent2D extent = chooseSwapExtent(support.capabilities);

        std::uint32_t imageCount = support.capabilities.minImageCount + 1;
        if (support.capabilities.maxImageCount > 0 && imageCount > support.capabilities.maxImageCount)
        {
            imageCount = support.capabilities.maxImageCount;
        }

        QueueFamilyIndices indices = findQueueFamilies(physicalDevice_);
        std::array<std::uint32_t, 2> queueFamilyIndices = {
            indices.graphicsFamily.value(),
            indices.presentFamily.value(),
        };

        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = surface_;
        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        if (indices.graphicsFamily != indices.presentFamily)
        {
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = static_cast<std::uint32_t>(queueFamilyIndices.size());
            createInfo.pQueueFamilyIndices = queueFamilyIndices.data();
        }
        else
        {
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }

        createInfo.preTransform = support.capabilities.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = presentMode;
        createInfo.clipped = VK_TRUE;
        createInfo.oldSwapchain = VK_NULL_HANDLE;

        if (vkCreateSwapchainKHR(device_, &createInfo, nullptr, &swapchain_) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create Vulkan swapchain.");
        }

        vkGetSwapchainImagesKHR(device_, swapchain_, &imageCount, nullptr);
        swapchainImages_.resize(imageCount);
        vkGetSwapchainImagesKHR(device_, swapchain_, &imageCount, swapchainImages_.data());

        swapchainImageFormat_ = surfaceFormat.format;
        swapchainExtent_ = extent;
    }

    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) const
    {
        for (const VkSurfaceFormatKHR& format : availableFormats)
        {
            if (format.format == VK_FORMAT_B8G8R8A8_SRGB &&
                format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            {
                return format;
            }
        }

        return availableFormats.front();
    }

    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) const
    {
        for (VkPresentModeKHR presentMode : availablePresentModes)
        {
            if (presentMode == VK_PRESENT_MODE_IMMEDIATE_KHR)
            {
                return presentMode;
            }
        }

        for (VkPresentModeKHR presentMode : availablePresentModes)
        {
            if (presentMode == VK_PRESENT_MODE_MAILBOX_KHR)
            {
                return presentMode;
            }
        }

        return VK_PRESENT_MODE_FIFO_KHR;
    }

    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) const
    {
        if (capabilities.currentExtent.width != std::numeric_limits<std::uint32_t>::max())
        {
            return capabilities.currentExtent;
        }

        int width = 0;
        int height = 0;
        glfwGetFramebufferSize(window_, &width, &height);

        VkExtent2D actualExtent = {
            static_cast<std::uint32_t>(width),
            static_cast<std::uint32_t>(height),
        };

        actualExtent.width = std::clamp(
            actualExtent.width,
            capabilities.minImageExtent.width,
            capabilities.maxImageExtent.width);
        actualExtent.height = std::clamp(
            actualExtent.height,
            capabilities.minImageExtent.height,
            capabilities.maxImageExtent.height);

        return actualExtent;
    }

    void createImageViews()
    {
        swapchainImageViews_.resize(swapchainImages_.size());
        for (std::size_t i = 0; i < swapchainImages_.size(); ++i)
        {
            swapchainImageViews_[i] = createImageView(swapchainImages_[i], swapchainImageFormat_);
        }
    }

    VkImageView createImageView(
        VkImage image,
        VkFormat format,
        VkImageAspectFlags aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        VkImageViewType viewType = VK_IMAGE_VIEW_TYPE_2D,
        std::uint32_t layerCount = 1)
    {
        VkImageViewCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        createInfo.image = image;
        createInfo.viewType = viewType;
        createInfo.format = format;
        createInfo.subresourceRange.aspectMask = aspectMask;
        createInfo.subresourceRange.baseMipLevel = 0;
        createInfo.subresourceRange.levelCount = 1;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.layerCount = layerCount;

        VkImageView imageView = VK_NULL_HANDLE;
        if (vkCreateImageView(device_, &createInfo, nullptr, &imageView) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create Vulkan image view.");
        }

        return imageView;
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

        createImage(
            depthTexture_.width,
            depthTexture_.height,
            depthTexture_.format,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            depthTexture_.image,
            depthTexture_.memory);

        depthTexture_.view = createImageView(depthTexture_.image, depthTexture_.format, VK_IMAGE_ASPECT_DEPTH_BIT);
        depthTexture_.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    }

    void createDescriptorSetLayout()
    {
        VkDescriptorSetLayoutBinding uniformBinding{};
        uniformBinding.binding = 0;
        uniformBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uniformBinding.descriptorCount = 1;
        uniformBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        VkDescriptorSetLayoutBinding sunTextureBinding{};
        sunTextureBinding.binding = 1;
        sunTextureBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sunTextureBinding.descriptorCount = 1;
        sunTextureBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutBinding moonTextureBinding{};
        moonTextureBinding.binding = 2;
        moonTextureBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        moonTextureBinding.descriptorCount = 1;
        moonTextureBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        std::array<VkDescriptorSetLayoutBinding, 3> bindings = {
            uniformBinding,
            sunTextureBinding,
            moonTextureBinding,
        };

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<std::uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();

        if (vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr, &descriptorSetLayout_) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create Vulkan descriptor set layout.");
        }

        VkDescriptorSetLayoutBinding blockUniformBinding{};
        blockUniformBinding.binding = 0;
        blockUniformBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        blockUniformBinding.descriptorCount = 1;
        blockUniformBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        VkDescriptorSetLayoutBinding blockTextureBinding{};
        blockTextureBinding.binding = 1;
        blockTextureBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        blockTextureBinding.descriptorCount = 1;
        blockTextureBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        std::array<VkDescriptorSetLayoutBinding, 2> blockBindings = {
            blockUniformBinding,
            blockTextureBinding,
        };

        VkDescriptorSetLayoutCreateInfo blockLayoutInfo{};
        blockLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        blockLayoutInfo.bindingCount = static_cast<std::uint32_t>(blockBindings.size());
        blockLayoutInfo.pBindings = blockBindings.data();

        if (vkCreateDescriptorSetLayout(device_, &blockLayoutInfo, nullptr, &blockDescriptorSetLayout_) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create Vulkan block descriptor set layout.");
        }

        VkDescriptorSetLayoutBinding debugTextTextureBinding{};
        debugTextTextureBinding.binding = 0;
        debugTextTextureBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        debugTextTextureBinding.descriptorCount = 1;
        debugTextTextureBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo debugTextLayoutInfo{};
        debugTextLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        debugTextLayoutInfo.bindingCount = 1;
        debugTextLayoutInfo.pBindings = &debugTextTextureBinding;

        if (vkCreateDescriptorSetLayout(device_, &debugTextLayoutInfo, nullptr, &debugTextDescriptorSetLayout_) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create Vulkan debug text descriptor set layout.");
        }
    }

    void createGraphicsPipeline()
    {
        const std::string shaderDir = std::string(VULKAN_VOXEL_BINARY_DIR) + "/shaders/";
        std::vector<char> vertexShaderCode = readBinaryFile(shaderDir + "sky.vert.spv");
        std::vector<char> fragmentShaderCode = readBinaryFile(shaderDir + "sky.frag.spv");

        VkShaderModule vertexShaderModule = createShaderModule(vertexShaderCode);
        VkShaderModule fragmentShaderModule = createShaderModule(fragmentShaderCode);

        VkPipelineShaderStageCreateInfo vertexShaderStageInfo{};
        vertexShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertexShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertexShaderStageInfo.module = vertexShaderModule;
        vertexShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo fragmentShaderStageInfo{};
        fragmentShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragmentShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragmentShaderStageInfo.module = fragmentShaderModule;
        fragmentShaderStageInfo.pName = "main";

        std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages = {
            vertexShaderStageInfo,
            fragmentShaderStageInfo,
        };

        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = static_cast<float>(swapchainExtent_.height);
        viewport.width = static_cast<float>(swapchainExtent_.width);
        viewport.height = -static_cast<float>(swapchainExtent_.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        VkRect2D scissor{};
        scissor.offset = {0, 0};
        scissor.extent = swapchainExtent_;

        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.pViewports = &viewport;
        viewportState.scissorCount = 1;
        viewportState.pScissors = &scissor;

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_NONE;
        rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;

        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable = VK_FALSE;
        depthStencil.depthWriteEnable = VK_FALSE;
        depthStencil.stencilTestEnable = VK_FALSE;

        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT |
            VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT |
            VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_TRUE;
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout_;

        if (vkCreatePipelineLayout(device_, &pipelineLayoutInfo, nullptr, &pipelineLayout_) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create Vulkan pipeline layout.");
        }

        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = static_cast<std::uint32_t>(shaderStages.size());
        pipelineInfo.pStages = shaderStages.data();
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pDepthStencilState = &depthStencil;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.layout = pipelineLayout_;
        pipelineInfo.renderPass = renderPass_;
        pipelineInfo.subpass = 0;

        if (vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline_) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create Vulkan graphics pipeline.");
        }

        vkDestroyShaderModule(device_, fragmentShaderModule, nullptr);
        vkDestroyShaderModule(device_, vertexShaderModule, nullptr);
    }

    void createBlockPipeline()
    {
        const std::string shaderDir = std::string(VULKAN_VOXEL_BINARY_DIR) + "/shaders/";
        std::vector<char> vertexShaderCode = readBinaryFile(shaderDir + "block.vert.spv");
        std::vector<char> fragmentShaderCode = readBinaryFile(shaderDir + "block.frag.spv");

        VkShaderModule vertexShaderModule = createShaderModule(vertexShaderCode);
        VkShaderModule fragmentShaderModule = createShaderModule(fragmentShaderCode);

        VkPipelineShaderStageCreateInfo vertexShaderStageInfo{};
        vertexShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertexShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertexShaderStageInfo.module = vertexShaderModule;
        vertexShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo fragmentShaderStageInfo{};
        fragmentShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragmentShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragmentShaderStageInfo.module = fragmentShaderModule;
        fragmentShaderStageInfo.pName = "main";

        std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages = {
            vertexShaderStageInfo,
            fragmentShaderStageInfo,
        };

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

        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<std::uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = static_cast<float>(swapchainExtent_.height);
        viewport.width = static_cast<float>(swapchainExtent_.width);
        viewport.height = -static_cast<float>(swapchainExtent_.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        VkRect2D scissor{};
        scissor.offset = {0, 0};
        scissor.extent = swapchainExtent_;

        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.pViewports = &viewport;
        viewportState.scissorCount = 1;
        viewportState.pScissors = &scissor;

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        // Mesh vertices are authored outward CCW.
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthWriteEnable = VK_TRUE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
        depthStencil.depthBoundsTestEnable = VK_FALSE;
        depthStencil.stencilTestEnable = VK_FALSE;

        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT |
            VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT |
            VK_COLOR_COMPONENT_A_BIT;

        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &blockDescriptorSetLayout_;

        if (vkCreatePipelineLayout(device_, &pipelineLayoutInfo, nullptr, &blockPipelineLayout_) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create Vulkan block pipeline layout.");
        }

        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = static_cast<std::uint32_t>(shaderStages.size());
        pipelineInfo.pStages = shaderStages.data();
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pDepthStencilState = &depthStencil;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.layout = blockPipelineLayout_;
        pipelineInfo.renderPass = renderPass_;
        pipelineInfo.subpass = 0;

        if (vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &blockPipeline_) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create Vulkan block graphics pipeline.");
        }

        vkDestroyShaderModule(device_, fragmentShaderModule, nullptr);
        vkDestroyShaderModule(device_, vertexShaderModule, nullptr);
    }

    void createDebugTextPipeline()
    {
        const std::string shaderDir = std::string(VULKAN_VOXEL_BINARY_DIR) + "/shaders/";
        std::vector<char> vertexShaderCode = readBinaryFile(shaderDir + "debug_text.vert.spv");
        std::vector<char> fragmentShaderCode = readBinaryFile(shaderDir + "debug_text.frag.spv");

        VkShaderModule vertexShaderModule = createShaderModule(vertexShaderCode);
        VkShaderModule fragmentShaderModule = createShaderModule(fragmentShaderCode);

        VkPipelineShaderStageCreateInfo vertexShaderStageInfo{};
        vertexShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertexShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertexShaderStageInfo.module = vertexShaderModule;
        vertexShaderStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo fragmentShaderStageInfo{};
        fragmentShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragmentShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragmentShaderStageInfo.module = fragmentShaderModule;
        fragmentShaderStageInfo.pName = "main";

        std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages = {
            vertexShaderStageInfo,
            fragmentShaderStageInfo,
        };

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

        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<std::uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = static_cast<float>(swapchainExtent_.height);
        viewport.width = static_cast<float>(swapchainExtent_.width);
        viewport.height = -static_cast<float>(swapchainExtent_.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        VkRect2D scissor{};
        scissor.offset = {0, 0};
        scissor.extent = swapchainExtent_;

        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.pViewports = &viewport;
        viewportState.scissorCount = 1;
        viewportState.pScissors = &scissor;

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_NONE;
        rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;

        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable = VK_FALSE;
        depthStencil.depthWriteEnable = VK_FALSE;
        depthStencil.stencilTestEnable = VK_FALSE;

        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT |
            VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT |
            VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_TRUE;
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &debugTextDescriptorSetLayout_;

        if (vkCreatePipelineLayout(device_, &pipelineLayoutInfo, nullptr, &debugTextPipelineLayout_) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create Vulkan debug text pipeline layout.");
        }

        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = static_cast<std::uint32_t>(shaderStages.size());
        pipelineInfo.pStages = shaderStages.data();
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pDepthStencilState = &depthStencil;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.layout = debugTextPipelineLayout_;
        pipelineInfo.renderPass = renderPass_;
        pipelineInfo.subpass = 0;

        if (vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &debugTextPipeline_) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create Vulkan debug text graphics pipeline.");
        }

        vkDestroyShaderModule(device_, fragmentShaderModule, nullptr);
        vkDestroyShaderModule(device_, vertexShaderModule, nullptr);
    }

    VkShaderModule createShaderModule(const std::vector<char>& code)
    {
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const std::uint32_t*>(code.data());

        VkShaderModule shaderModule = VK_NULL_HANDLE;
        if (vkCreateShaderModule(device_, &createInfo, nullptr, &shaderModule) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create Vulkan shader module.");
        }

        return shaderModule;
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

    void addBlockDefinition(std::uint16_t id, std::string name, bool solid)
    {
        BlockDefinition definition{};
        definition.id = id;
        definition.name = std::move(name);
        definition.solid = solid;
        blockDefinitionIndices_[id] = static_cast<int>(blockDefinitions_.size());
        blockDefinitions_.push_back(std::move(definition));
    }

    static std::optional<std::uint32_t> readJsonUInt(const std::string& object, const char* key)
    {
        const std::string pattern = "\"" + std::string(key) + "\"";
        const std::size_t keyPos = object.find(pattern);
        if (keyPos == std::string::npos)
        {
            return std::nullopt;
        }

        const std::size_t colonPos = object.find(':', keyPos + pattern.size());
        if (colonPos == std::string::npos)
        {
            return std::nullopt;
        }

        std::size_t valuePos = colonPos + 1;
        while (valuePos < object.size() && std::isspace(static_cast<unsigned char>(object[valuePos])))
        {
            ++valuePos;
        }

        std::size_t endPos = valuePos;
        while (endPos < object.size() && std::isdigit(static_cast<unsigned char>(object[endPos])))
        {
            ++endPos;
        }
        if (endPos == valuePos)
        {
            return std::nullopt;
        }

        return static_cast<std::uint32_t>(std::stoul(object.substr(valuePos, endPos - valuePos)));
    }

    static std::optional<std::string> readJsonString(const std::string& object, const char* key)
    {
        const std::string pattern = "\"" + std::string(key) + "\"";
        const std::size_t keyPos = object.find(pattern);
        if (keyPos == std::string::npos)
        {
            return std::nullopt;
        }

        const std::size_t colonPos = object.find(':', keyPos + pattern.size());
        const std::size_t quotePos = object.find('"', colonPos == std::string::npos ? keyPos : colonPos);
        if (colonPos == std::string::npos || quotePos == std::string::npos)
        {
            return std::nullopt;
        }

        const std::size_t endQuotePos = object.find('"', quotePos + 1);
        if (endQuotePos == std::string::npos)
        {
            return std::nullopt;
        }
        return object.substr(quotePos + 1, endQuotePos - quotePos - 1);
    }

    static bool readJsonBool(const std::string& object, const char* key, bool fallback)
    {
        const std::string pattern = "\"" + std::string(key) + "\"";
        const std::size_t keyPos = object.find(pattern);
        if (keyPos == std::string::npos)
        {
            return fallback;
        }

        const std::size_t colonPos = object.find(':', keyPos + pattern.size());
        if (colonPos == std::string::npos)
        {
            return fallback;
        }

        std::size_t valuePos = colonPos + 1;
        while (valuePos < object.size() && std::isspace(static_cast<unsigned char>(object[valuePos])))
        {
            ++valuePos;
        }
        return object.compare(valuePos, 4, "true") == 0;
    }

    void loadBlockDefinitions()
    {
        blockDefinitions_.clear();
        blockDefinitionIndices_.fill(-1);
        blockTextureLayerNames_.clear();

        std::ifstream configFile(sourcePath("/config/blocks.json"));
        if (configFile)
        {
            const std::string config(
                (std::istreambuf_iterator<char>(configFile)),
                std::istreambuf_iterator<char>());
            const std::size_t blocksPos = config.find("\"blocks\"");
            const std::size_t arrayBegin = config.find('[', blocksPos);
            const std::size_t arrayEnd = config.find(']', arrayBegin);
            std::size_t cursor = arrayBegin;
            while (cursor != std::string::npos && cursor < arrayEnd)
            {
                const std::size_t objectBegin = config.find('{', cursor);
                if (objectBegin == std::string::npos || objectBegin > arrayEnd)
                {
                    break;
                }
                const std::size_t objectEnd = config.find('}', objectBegin);
                if (objectEnd == std::string::npos || objectEnd > arrayEnd)
                {
                    break;
                }

                const std::string object = config.substr(objectBegin, objectEnd - objectBegin + 1);
                const auto id = readJsonUInt(object, "id");
                const auto name = readJsonString(object, "name");
                if (id && *id <= std::numeric_limits<std::uint16_t>::max() && name && !name->empty())
                {
                    addBlockDefinition(
                        static_cast<std::uint16_t>(*id),
                        *name,
                        readJsonBool(object, "solid", true));
                }

                cursor = objectEnd + 1;
            }
        }

        if (blockDefinitions_.empty())
        {
            addBlockDefinition(kRockBlockId, "rock", true);
            addBlockDefinition(kDirtBlockId, "dirt", true);
            addBlockDefinition(kGrassBlockId, "grass", true);
            addBlockDefinition(kBedrockBlockId, "bedrock", true);
        }
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
        const std::wstring selectedRelativePath =
            fileExists(sourcePathWide(specificRelativePath)) ? specificRelativePath : baseRelativePath;
        if (!fileExists(sourcePathWide(selectedRelativePath)))
        {
            throw std::runtime_error("Missing block texture.");
        }

        return addBlockTextureLayer(selectedRelativePath);
    }

    void createBlockTextureArray()
    {
        for (BlockDefinition& definition : blockDefinitions_)
        {
            if (!definition.solid || definition.id == kAirBlockId)
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

        std::vector<ImagePixels> layers;
        layers.reserve(blockTextureLayerNames_.size());
        for (const std::wstring& relativePath : blockTextureLayerNames_)
        {
            layers.push_back(loadPngWithWic(sourcePathWide(relativePath)));
        }
        if (layers.empty())
        {
            throw std::runtime_error("No block textures were loaded.");
        }

        blockTextureArray_ = createTextureArrayFromPixels(layers, VK_FORMAT_R8G8B8A8_SRGB);
    }

    int terrainHeightAt(int x, int z) const
    {
        constexpr float amplitude = static_cast<float>(kTerrainHeightRange) * 0.5f;
        const float xWave = std::sin(static_cast<float>(x) * 0.035f);
        const float zWave = std::cos(static_cast<float>(z) * 0.041f);
        const float diagonalWave = std::sin(static_cast<float>(x + z) * 0.018f) * 0.35f;
        const float normalizedWave = std::clamp((xWave + zWave + diagonalWave) / 2.35f, -1.0f, 1.0f);
        return kTerrainBaseHeight + static_cast<int>(std::lround(normalizedWave * amplitude));
    }

    void loadWorldConfig()
    {
        std::ifstream configFile(sourcePath("/config/world.json"));
        if (!configFile)
        {
            return;
        }

        const std::string config(
            (std::istreambuf_iterator<char>(configFile)),
            std::istreambuf_iterator<char>());
        auto readInt = [&](const std::string& key, int fallback) -> int
        {
            const std::string quotedKey = "\"" + key + "\"";
            const std::size_t keyPosition = config.find(quotedKey);
            if (keyPosition == std::string::npos)
            {
                return fallback;
            }

            const std::size_t colonPosition = config.find(':', keyPosition + quotedKey.size());
            if (colonPosition == std::string::npos)
            {
                return fallback;
            }

            const std::size_t valueStart = config.find_first_of("-0123456789", colonPosition + 1);
            if (valueStart == std::string::npos)
            {
                return fallback;
            }

            const std::size_t valueEnd = config.find_first_not_of("0123456789", valueStart + 1);
            try
            {
                return std::stoi(config.substr(valueStart, valueEnd - valueStart));
            }
            catch (const std::exception&)
            {
                return fallback;
            }
        };

        chunkLoadRadius_ = std::clamp(
            readInt("chunkLoadRadius", kDefaultChunkLoadRadius),
            0,
            kMaxChunkLoadRadius);
        chunkUploadsPerFrame_ = std::clamp(
            readInt("chunkUploadsPerFrame", kDefaultChunkUploadsPerFrame),
            1,
            kMaxChunkUploadsPerFrame);
        chunkBuildThreads_ = std::clamp(
            readInt("chunkBuildThreads", kDefaultChunkBuildThreads),
            1,
            kMaxChunkBuildThreads);
    }

    void createChunkMesh()
    {
        static_assert(kAirBlockId == 0 && kBlockIdCount == 65536);
        updateLoadedChunks();
    }

    void startChunkBuildWorkers()
    {
        chunkBuildWorkerRunning_ = true;
        chunkBuildWorkers_.reserve(static_cast<std::size_t>(chunkBuildThreads_));
        for (int i = 0; i < chunkBuildThreads_; ++i)
        {
            chunkBuildWorkers_.emplace_back([this]()
            {
                chunkBuildWorkerLoop();
            });
        }
    }

    void stopChunkBuildWorkers()
    {
        if (!chunkBuildWorkerRunning_ && chunkBuildWorkers_.empty())
        {
            return;
        }

        {
            std::lock_guard<std::mutex> lock(chunkBuildMutex_);
            chunkBuildWorkerRunning_ = false;
            pendingChunkBuilds_.clear();
            completedSubchunkBuilds_.clear();
        }
        chunkBuildCv_.notify_all();

        for (std::thread& worker : chunkBuildWorkers_)
        {
            if (worker.joinable())
            {
                worker.join();
            }
        }
        chunkBuildWorkers_.clear();

        {
            std::lock_guard<std::mutex> lock(chunkBuildMutex_);
            pendingChunkBuilds_.clear();
            completedSubchunkBuilds_.clear();
            completedChunkBuilds_.clear();
        }
        queuedChunks_.clear();
        desiredChunks_.clear();
        queuedChunkGenerations_.clear();
        pendingChunkMeshes_.clear();
    }

    void chunkBuildWorkerLoop()
    {
        while (true)
        {
            ChunkBuildRequest request{};
            bool hasRequest = false;
            {
                std::unique_lock<std::mutex> lock(chunkBuildMutex_);
                chunkBuildCv_.wait(lock, [this]()
                {
                    return !chunkBuildWorkerRunning_ || !pendingChunkBuilds_.empty();
                });

                while (chunkBuildWorkerRunning_ && !pendingChunkBuilds_.empty())
                {
                    std::pop_heap(
                        pendingChunkBuilds_.begin(),
                        pendingChunkBuilds_.end(),
                        ChunkBuildRequestPriority{});
                    request = pendingChunkBuilds_.back();
                    pendingChunkBuilds_.pop_back();

                    const auto generationIt = queuedChunkGenerations_.find(request.coord);
                    if (generationIt != queuedChunkGenerations_.end() &&
                        generationIt->second == request.generation)
                    {
                        const std::int64_t currentDistanceSq = chunkDistanceSq(
                            request.coord,
                            chunkBuildPriorityCenterX_,
                            chunkBuildPriorityCenterZ_);
                        if (currentDistanceSq != request.priorityDistanceSq)
                        {
                            request.priorityDistanceSq = currentDistanceSq;
                            pendingChunkBuilds_.push_back(request);
                            std::push_heap(
                                pendingChunkBuilds_.begin(),
                                pendingChunkBuilds_.end(),
                                ChunkBuildRequestPriority{});
                            continue;
                        }

                        hasRequest = true;
                        break;
                    }
                }

                if (!chunkBuildWorkerRunning_)
                {
                    return;
                }
            }

            if (!hasRequest)
            {
                continue;
            }

            SubchunkBuildResult result = buildSubchunkMeshCpu(request);

            {
                std::lock_guard<std::mutex> lock(chunkBuildMutex_);
                completedSubchunkBuilds_.push_back(std::move(result));
            }
        }
    }

    void cancelQueuedChunkBuild(ChunkCoord coord)
    {
        std::lock_guard<std::mutex> lock(chunkBuildMutex_);
        queuedChunks_.erase(coord);
        queuedChunkGenerations_.erase(coord);
        pendingChunkMeshes_.erase(coord);
    }

    bool isChunkInLoadRange(ChunkCoord coord, int centerChunkX, int centerChunkZ) const
    {
        return std::abs(coord.x - centerChunkX) <= chunkLoadRadius_ &&
               std::abs(coord.z - centerChunkZ) <= chunkLoadRadius_;
    }

    bool isChunkInCurrentLoadRange(ChunkCoord coord) const
    {
        if (loadedCenterChunkX_ == std::numeric_limits<int>::min() ||
            loadedCenterChunkZ_ == std::numeric_limits<int>::min())
        {
            return false;
        }
        return isChunkInLoadRange(coord, loadedCenterChunkX_, loadedCenterChunkZ_);
    }

    bool queueChunkBuild(
        ChunkCoord coord,
        int centerChunkX,
        int centerChunkZ)
    {
        {
            std::lock_guard<std::mutex> lock(chunkBuildMutex_);
            chunkBuildPriorityCenterX_ = centerChunkX;
            chunkBuildPriorityCenterZ_ = centerChunkZ;
            if (loadedChunks_.contains(coord) || queuedChunks_.contains(coord))
            {
                return false;
            }

            queuedChunks_.insert(coord);
            const std::uint64_t generation = ++chunkBuildGeneration_;
            queuedChunkGenerations_[coord] = generation;
            const std::int64_t priorityDistanceSq = chunkDistanceSq(coord, centerChunkX, centerChunkZ);
            for (int subchunkY = 0; subchunkY < kSubchunksPerChunk; ++subchunkY)
            {
                pendingChunkBuilds_.push_back({
                    coord,
                    subchunkY,
                    priorityDistanceSq,
                    generation,
                });
                std::push_heap(
                    pendingChunkBuilds_.begin(),
                    pendingChunkBuilds_.end(),
                    ChunkBuildRequestPriority{});
            }
        }

        return true;
    }

    void reprioritizePendingChunkBuilds(int centerChunkX, int centerChunkZ)
    {
        std::lock_guard<std::mutex> lock(chunkBuildMutex_);
        chunkBuildPriorityCenterX_ = centerChunkX;
        chunkBuildPriorityCenterZ_ = centerChunkZ;
        pendingChunkBuilds_.erase(
            std::remove_if(
                pendingChunkBuilds_.begin(),
                pendingChunkBuilds_.end(),
                [this](const ChunkBuildRequest& request)
                {
                    const auto generationIt = queuedChunkGenerations_.find(request.coord);
                    return generationIt == queuedChunkGenerations_.end() ||
                           generationIt->second != request.generation;
                }),
            pendingChunkBuilds_.end());
        for (ChunkBuildRequest& request : pendingChunkBuilds_)
        {
            request.priorityDistanceSq = chunkDistanceSq(request.coord, centerChunkX, centerChunkZ);
        }
        std::make_heap(
            pendingChunkBuilds_.begin(),
            pendingChunkBuilds_.end(),
            ChunkBuildRequestPriority{});
    }

    std::size_t cancelQueuedChunksOutsideDesired()
    {
        std::size_t canceledCount = 0;
        std::lock_guard<std::mutex> lock(chunkBuildMutex_);
        for (auto it = queuedChunkGenerations_.begin(); it != queuedChunkGenerations_.end();)
        {
            if (desiredChunks_.contains(it->first))
            {
                ++it;
                continue;
            }

            queuedChunks_.erase(it->first);
            pendingChunkMeshes_.erase(it->first);
            it = queuedChunkGenerations_.erase(it);
            ++canceledCount;
        }
        return canceledCount;
    }

    ChunkLoadUpdateStats updateLoadedChunks()
    {
        ChunkLoadUpdateStats stats{};
        const int centerChunkX = chunkCoordForWorldPosition(camera_.position.x, kChunkSizeX);
        const int centerChunkZ = chunkCoordForWorldPosition(camera_.position.z, kChunkSizeZ);
        {
            std::lock_guard<std::mutex> lock(chunkBuildMutex_);
            chunkBuildPriorityCenterX_ = centerChunkX;
            chunkBuildPriorityCenterZ_ = centerChunkZ;
        }

        if (centerChunkX != loadedCenterChunkX_ ||
            centerChunkZ != loadedCenterChunkZ_)
        {
            const std::size_t sideLength = static_cast<std::size_t>(chunkLoadRadius_ * 2 + 1);
            desiredChunks_.clear();
            desiredChunks_.reserve(sideLength * sideLength);
            for (int dz = -chunkLoadRadius_; dz <= chunkLoadRadius_; ++dz)
            {
                for (int dx = -chunkLoadRadius_; dx <= chunkLoadRadius_; ++dx)
                {
                    desiredChunks_.insert({centerChunkX + dx, centerChunkZ + dz});
                }
            }

            std::vector<ChunkCoord> chunksToUnload;
            for (ChunkCoord coord : loadedChunks_)
            {
                if (!desiredChunks_.contains(coord))
                {
                    chunksToUnload.push_back(coord);
                }
            }

            for (ChunkCoord coord : chunksToUnload)
            {
                destroyChunkMeshes(coord);
                loadedChunks_.erase(coord);
            }
            stats.unloaded = chunksToUnload.size();

            cancelQueuedChunksOutsideDesired();
            reprioritizePendingChunkBuilds(centerChunkX, centerChunkZ);
            for (ChunkCoord coord : desiredChunks_)
            {
                if (loadedChunks_.contains(coord))
                {
                    continue;
                }
                if (queueChunkBuild(coord, centerChunkX, centerChunkZ))
                {
                    ++stats.queued;
                }
            }
            if (stats.queued > 0)
            {
                chunkBuildCv_.notify_all();
            }

            loadedCenterChunkX_ = centerChunkX;
            loadedCenterChunkZ_ = centerChunkZ;
        }

        return stats;
    }

    const BlockDefinition* blockDefinitionForId(std::uint16_t blockId) const
    {
        const int index = blockDefinitionIndices_[blockId];
        if (index < 0)
        {
            return nullptr;
        }
        return &blockDefinitions_[static_cast<std::size_t>(index)];
    }

    bool isSolidBlock(std::uint16_t blockId) const
    {
        const BlockDefinition* definition = blockDefinitionForId(blockId);
        return definition != nullptr && definition->solid;
    }

    std::uint32_t textureLayerForBlockFace(std::uint16_t blockId, BlockFace face) const
    {
        const BlockDefinition* definition = blockDefinitionForId(blockId);
        if (definition == nullptr)
        {
            return 0;
        }
        return definition->textureLayers[static_cast<std::size_t>(face)];
    }

    static std::size_t chunkColumnIndex(int localX, int localZ)
    {
        return static_cast<std::size_t>(localZ * kChunkSizeX + localX);
    }

    int terrainHighestSolidYAt(int x, int z) const
    {
        const int terrainHeight = terrainHeightAt(x, z);
        return terrainHeight > 0 ? terrainHeight - 1 : -1;
    }

    ChunkColumnData generateChunkColumnData(ChunkCoord chunk) const
    {
        ChunkColumnData data{};
        const int chunkBaseX = chunk.x * kChunkSizeX;
        const int chunkBaseZ = chunk.z * kChunkSizeZ;
        for (int localZ = 0; localZ < kChunkSizeZ; ++localZ)
        {
            for (int localX = 0; localX < kChunkSizeX; ++localX)
            {
                data.highestSolidY[chunkColumnIndex(localX, localZ)] =
                    terrainHighestSolidYAt(chunkBaseX + localX, chunkBaseZ + localZ);
            }
        }
        return data;
    }

    std::uint16_t generateBaseTerrainBlock(int y, int highestSolidY) const
    {
        if (y < 0 || highestSolidY < 0 || y > highestSolidY)
        {
            return kAirBlockId;
        }
        return kRockBlockId;
    }

    std::uint16_t applyTerrainPostProcess(std::uint16_t blockId, int y, int highestSolidY) const
    {
        if (blockId == kAirBlockId)
        {
            return blockId;
        }
        if (y == 0)
        {
            return kBedrockBlockId;
        }
        if (y == highestSolidY)
        {
            return kGrassBlockId;
        }
        if (y >= highestSolidY - 4 && y < highestSolidY)
        {
            return kDirtBlockId;
        }
        return blockId;
    }

    std::uint16_t generateBlockIdFromColumn(int y, int highestSolidY) const
    {
        return applyTerrainPostProcess(
            generateBaseTerrainBlock(y, highestSolidY),
            y,
            highestSolidY);
    }

    SubchunkBuildResult buildSubchunkMeshCpu(ChunkBuildRequest request)
    {
        const ChunkCoord chunk = request.coord;
        std::array<std::vector<BlockVertex>, kSubchunkSize> verticesByLocalY;
        std::array<std::vector<std::uint32_t>, kSubchunkSize> indicesByLocalY;
        std::vector<BlockVertex> subchunkVertices;
        std::vector<std::uint32_t> subchunkIndices;
        std::vector<SubchunkDraw> subchunkDraws;
        subchunkDraws.reserve(1);

        const int chunkX = chunk.x;
        const int chunkZ = chunk.z;
        const int subchunkY = request.subchunkY;
        const int chunkBaseX = chunkX * kChunkSizeX;
        const int chunkBaseZ = chunkZ * kChunkSizeZ;
        const int subchunkMinY = subchunkY * kSubchunkSize;
        const ChunkColumnData chunkColumnData = generateChunkColumnData(chunk);

        constexpr int kPaddedSubchunkSize = kSubchunkSize + 2;
        constexpr int kPaddedSubchunkArea = kPaddedSubchunkSize * kPaddedSubchunkSize;
        constexpr int kPaddedSubchunkVolume = kPaddedSubchunkArea * kPaddedSubchunkSize;
        std::array<std::uint16_t, kPaddedSubchunkVolume> blockIds{};

        auto paddedBlockIndex = [&](int localX, int localY, int localZ) -> std::size_t
        {
            return static_cast<std::size_t>(
                (localY * kPaddedSubchunkSize + localZ) * kPaddedSubchunkSize + localX);
        };

        int nonAirBlockCount = 0;
        auto highestSolidYAt = [&](int x, int z) -> int
        {
            const int localX = x - chunkBaseX;
            const int localZ = z - chunkBaseZ;
            if (localX >= 0 && localX < kChunkSizeX &&
                localZ >= 0 && localZ < kChunkSizeZ)
            {
                return chunkColumnData.highestSolidY[chunkColumnIndex(localX, localZ)];
            }
            return terrainHighestSolidYAt(x, z);
        };

        for (int localZ = 0; localZ < kPaddedSubchunkSize; ++localZ)
        {
            const int z = chunkBaseZ + localZ - 1;
            for (int localX = 0; localX < kPaddedSubchunkSize; ++localX)
            {
                const int x = chunkBaseX + localX - 1;
                const int highestSolidY = highestSolidYAt(x, z);
                for (int localY = 0; localY < kPaddedSubchunkSize; ++localY)
                {
                    const int y = subchunkMinY + localY - 1;
                    const std::uint16_t blockId = generateBlockIdFromColumn(y, highestSolidY);
                    blockIds[paddedBlockIndex(localX, localY, localZ)] = blockId;
                    if (localX > 0 && localX <= kSubchunkSize &&
                        localY > 0 && localY <= kSubchunkSize &&
                        localZ > 0 && localZ <= kSubchunkSize &&
                        blockId != kAirBlockId)
                    {
                        ++nonAirBlockCount;
                    }
                }
            }
        }

        if (nonAirBlockCount == 0)
        {
            return {
                chunk,
                subchunkY,
                request.generation,
                {},
                {},
            };
        }

        auto isSolid = [&](int x, int y, int z) -> bool
        {
            const int localX = x - chunkBaseX + 1;
            const int localY = y - subchunkMinY + 1;
            const int localZ = z - chunkBaseZ + 1;
            if (localX < 0 || localX >= kPaddedSubchunkSize ||
                localY < 0 || localY >= kPaddedSubchunkSize ||
                localZ < 0 || localZ >= kPaddedSubchunkSize)
            {
                return false;
            }
            return isSolidBlock(blockIds[paddedBlockIndex(localX, localY, localZ)]);
        };

        auto blockIdAt = [&](int x, int y, int z) -> std::uint16_t
        {
            const int localX = x - chunkBaseX + 1;
            const int localY = y - subchunkMinY + 1;
            const int localZ = z - chunkBaseZ + 1;
            if (localX < 0 || localX >= kPaddedSubchunkSize ||
                localY < 0 || localY >= kPaddedSubchunkSize ||
                localZ < 0 || localZ >= kPaddedSubchunkSize)
            {
                return kAirBlockId;
            }
            return blockIds[paddedBlockIndex(localX, localY, localZ)];
        };

        auto emitGreedyRectangles = [](
            int width,
            int height,
            auto cellAt,
            auto emitRectangle)
        {
            std::array<bool, kSubchunkSize * kSubchunkSize> consumed{};
            auto index = [](int u, int v) -> std::size_t
            {
                return static_cast<std::size_t>(v * kSubchunkSize + u);
            };

            for (int v = 0; v < height; ++v)
            {
                for (int u = 0; u < width; ++u)
                {
                    const FaceMaskCell baseCell = cellAt(u, v);
                    if (consumed[index(u, v)] || !baseCell.filled)
                    {
                        continue;
                    }

                    int rectWidth = 1;
                    while (u + rectWidth < width &&
                           !consumed[index(u + rectWidth, v)] &&
                           cellAt(u + rectWidth, v) == baseCell)
                    {
                        ++rectWidth;
                    }

                    int rectHeight = 1;
                    bool canGrow = true;
                    while (v + rectHeight < height && canGrow)
                    {
                        for (int du = 0; du < rectWidth; ++du)
                        {
                            if (consumed[index(u + du, v + rectHeight)] ||
                                !(cellAt(u + du, v + rectHeight) == baseCell))
                            {
                                canGrow = false;
                                break;
                            }
                        }

                        if (canGrow)
                        {
                            ++rectHeight;
                        }
                    }

                    for (int dv = 0; dv < rectHeight; ++dv)
                    {
                        for (int du = 0; du < rectWidth; ++du)
                        {
                            consumed[index(u + du, v + dv)] = true;
                        }
                    }

                    emitRectangle(u, v, rectWidth, rectHeight, baseCell);
                }
            }
        };

        auto addFace = [&](
            int subchunkLocalY,
            std::array<Vec3, 4> corners,
            std::array<std::array<float, 2>, 4> uvs,
            std::uint32_t textureLayer,
            std::array<std::uint8_t, 4> ao)
        {
            std::vector<BlockVertex>& vertices = verticesByLocalY[static_cast<std::size_t>(subchunkLocalY)];
            std::vector<std::uint32_t>& indices = indicesByLocalY[static_cast<std::size_t>(subchunkLocalY)];
            const std::uint32_t baseIndex = static_cast<std::uint32_t>(vertices.size());
            const float layer = static_cast<float>(textureLayer);
            vertices.push_back({{corners[0].x, corners[0].y, corners[0].z}, {uvs[0][0], uvs[0][1]}, aoFactor(ao[0]), layer});
            vertices.push_back({{corners[1].x, corners[1].y, corners[1].z}, {uvs[1][0], uvs[1][1]}, aoFactor(ao[1]), layer});
            vertices.push_back({{corners[2].x, corners[2].y, corners[2].z}, {uvs[2][0], uvs[2][1]}, aoFactor(ao[2]), layer});
            vertices.push_back({{corners[3].x, corners[3].y, corners[3].z}, {uvs[3][0], uvs[3][1]}, aoFactor(ao[3]), layer});

            indices.push_back(baseIndex + 0);
            indices.push_back(baseIndex + 1);
            indices.push_back(baseIndex + 2);
            indices.push_back(baseIndex + 0);
            indices.push_back(baseIndex + 2);
            indices.push_back(baseIndex + 3);
        };

        auto computeAo = [&](
            int side1X, int side1Y, int side1Z,
            int side2X, int side2Y, int side2Z,
            int cornerX, int cornerY, int cornerZ) -> std::uint8_t
        {
            return vertexAo(
                isSolid(side1X, side1Y, side1Z),
                isSolid(side2X, side2Y, side2Z),
                isSolid(cornerX, cornerY, cornerZ));
        };

        auto topFaceAo = [&](int x, int z, int faceY) -> std::array<std::uint8_t, 4>
        {
            return {{
                computeAo(x - 1, faceY, z, x, faceY, z - 1, x - 1, faceY, z - 1),
                computeAo(x - 1, faceY, z, x, faceY, z + 1, x - 1, faceY, z + 1),
                computeAo(x + 1, faceY, z, x, faceY, z + 1, x + 1, faceY, z + 1),
                computeAo(x + 1, faceY, z, x, faceY, z - 1, x + 1, faceY, z - 1),
            }};
        };

        auto bottomFaceAo = [&](int x, int z, int faceY) -> std::array<std::uint8_t, 4>
        {
            const int outsideY = faceY - 1;
            return {{
                computeAo(x - 1, outsideY, z, x, outsideY, z + 1, x - 1, outsideY, z + 1),
                computeAo(x - 1, outsideY, z, x, outsideY, z - 1, x - 1, outsideY, z - 1),
                computeAo(x + 1, outsideY, z, x, outsideY, z - 1, x + 1, outsideY, z - 1),
                computeAo(x + 1, outsideY, z, x, outsideY, z + 1, x + 1, outsideY, z + 1),
            }};
        };

        auto positiveXFaceAo = [&](int x, int y, int z) -> std::array<std::uint8_t, 4>
        {
            const int outsideX = x + 1;
            return {{
                computeAo(outsideX, y - 1, z, outsideX, y, z - 1, outsideX, y - 1, z - 1),
                computeAo(outsideX, y + 1, z, outsideX, y, z - 1, outsideX, y + 1, z - 1),
                computeAo(outsideX, y + 1, z, outsideX, y, z + 1, outsideX, y + 1, z + 1),
                computeAo(outsideX, y - 1, z, outsideX, y, z + 1, outsideX, y - 1, z + 1),
            }};
        };

        auto negativeXFaceAo = [&](int x, int y, int z) -> std::array<std::uint8_t, 4>
        {
            const int outsideX = x - 1;
            return {{
                computeAo(outsideX, y - 1, z, outsideX, y, z + 1, outsideX, y - 1, z + 1),
                computeAo(outsideX, y + 1, z, outsideX, y, z + 1, outsideX, y + 1, z + 1),
                computeAo(outsideX, y + 1, z, outsideX, y, z - 1, outsideX, y + 1, z - 1),
                computeAo(outsideX, y - 1, z, outsideX, y, z - 1, outsideX, y - 1, z - 1),
            }};
        };

        auto positiveZFaceAo = [&](int x, int y, int z) -> std::array<std::uint8_t, 4>
        {
            const int outsideZ = z + 1;
            return {{
                computeAo(x, y - 1, outsideZ, x + 1, y, outsideZ, x + 1, y - 1, outsideZ),
                computeAo(x, y + 1, outsideZ, x + 1, y, outsideZ, x + 1, y + 1, outsideZ),
                computeAo(x, y + 1, outsideZ, x - 1, y, outsideZ, x - 1, y + 1, outsideZ),
                computeAo(x, y - 1, outsideZ, x - 1, y, outsideZ, x - 1, y - 1, outsideZ),
            }};
        };

        auto negativeZFaceAo = [&](int x, int y, int z) -> std::array<std::uint8_t, 4>
        {
            const int outsideZ = z - 1;
            return {{
                computeAo(x, y - 1, outsideZ, x - 1, y, outsideZ, x - 1, y - 1, outsideZ),
                computeAo(x, y + 1, outsideZ, x - 1, y, outsideZ, x - 1, y + 1, outsideZ),
                computeAo(x, y + 1, outsideZ, x + 1, y, outsideZ, x + 1, y + 1, outsideZ),
                computeAo(x, y - 1, outsideZ, x + 1, y, outsideZ, x + 1, y - 1, outsideZ),
            }};
        };

        auto addTopFace = [&](
            int subchunkLocalY,
            int faceY,
            int x,
            int z,
            int width,
            int depth,
            std::uint32_t textureLayer,
            std::array<std::uint8_t, 4> ao)
        {
            const float fx = static_cast<float>(x);
            const float fz = static_cast<float>(z);
            const float xEnd = static_cast<float>(x + width);
            const float zEnd = static_cast<float>(z + depth);
            const float y = static_cast<float>(faceY);
            addFace(
                subchunkLocalY,
                {{
                    {fx, y, fz},
                    {fx, y, zEnd},
                    {xEnd, y, zEnd},
                    {xEnd, y, fz},
                }},
                {{
                    {{0.0f, 0.0f}},
                    {{static_cast<float>(depth), 0.0f}},
                    {{static_cast<float>(depth), static_cast<float>(width)}},
                    {{0.0f, static_cast<float>(width)}},
                }},
                textureLayer,
                ao);
        };

        auto addBottomFace = [&](
            int subchunkLocalY,
            int faceY,
            int x,
            int z,
            int width,
            int depth,
            std::uint32_t textureLayer,
            std::array<std::uint8_t, 4> ao)
        {
            const float fx = static_cast<float>(x);
            const float fz = static_cast<float>(z);
            const float xEnd = static_cast<float>(x + width);
            const float zEnd = static_cast<float>(z + depth);
            const float y = static_cast<float>(faceY);
            addFace(
                subchunkLocalY,
                {{
                    {fx, y, zEnd},
                    {fx, y, fz},
                    {xEnd, y, fz},
                    {xEnd, y, zEnd},
                }},
                {{
                    {{0.0f, 0.0f}},
                    {{static_cast<float>(depth), 0.0f}},
                    {{static_cast<float>(depth), static_cast<float>(width)}},
                    {{0.0f, static_cast<float>(width)}},
                }},
                textureLayer,
                ao);
        };

        auto addSideFace = [&](
            int subchunkLocalY,
            std::array<Vec3, 4> corners,
            int width,
            int height,
            std::uint32_t textureLayer,
            std::array<std::uint8_t, 4> ao)
        {
            addFace(
                subchunkLocalY,
                corners,
                {{
                    {{static_cast<float>(width), static_cast<float>(height)}},
                    {{static_cast<float>(width), 0.0f}},
                    {{0.0f, 0.0f}},
                    {{0.0f, static_cast<float>(height)}},
                }},
                textureLayer,
                ao);
        };

        for (auto& vertices : verticesByLocalY)
        {
            vertices.clear();
        }
        for (auto& indices : indicesByLocalY)
        {
            indices.clear();
        }

            for (int localY = 0; localY < kSubchunkSize; ++localY)
            {
                const int faceY = subchunkMinY + localY + 1;
                emitGreedyRectangles(
                    kChunkSizeX,
                    kChunkSizeZ,
                    [&](int localX, int localZ) -> FaceMaskCell
                    {
                        const int x = chunkBaseX + localX;
                        const int z = chunkBaseZ + localZ;
                        if (!(isSolid(x, faceY - 1, z) && !isSolid(x, faceY, z)))
                        {
                            return {};
                        }
                        return {
                            true,
                            textureLayerForBlockFace(blockIdAt(x, faceY - 1, z), BlockFace::Top),
                            topFaceAo(x, z, faceY),
                        };
                    },
                    [&](int localX, int localZ, int width, int depth, const FaceMaskCell& cell)
                    {
                        addTopFace(
                            localY,
                            faceY,
                            chunkBaseX + localX,
                            chunkBaseZ + localZ,
                            width,
                            depth,
                            cell.textureLayer,
                            cell.ao);
                    });

                const int bottomFaceY = subchunkMinY + localY;
                emitGreedyRectangles(
                    kChunkSizeX,
                    kChunkSizeZ,
                    [&](int localX, int localZ) -> FaceMaskCell
                    {
                        const int x = chunkBaseX + localX;
                        const int z = chunkBaseZ + localZ;
                        if (!(isSolid(x, bottomFaceY, z) && !isSolid(x, bottomFaceY - 1, z)))
                        {
                            return {};
                        }
                        return {
                            true,
                            textureLayerForBlockFace(blockIdAt(x, bottomFaceY, z), BlockFace::Bottom),
                            bottomFaceAo(x, z, bottomFaceY),
                        };
                    },
                    [&](int localX, int localZ, int width, int depth, const FaceMaskCell& cell)
                    {
                        addBottomFace(
                            localY,
                            bottomFaceY,
                            chunkBaseX + localX,
                            chunkBaseZ + localZ,
                            width,
                            depth,
                            cell.textureLayer,
                            cell.ao);
                    });
            }

            for (int localX = 0; localX < kChunkSizeX; ++localX)
            {
                const int x = chunkBaseX + localX;
                const float faceX = static_cast<float>(x + 1);
                emitGreedyRectangles(
                    kChunkSizeZ,
                    kSubchunkSize,
                    [&](int localZ, int localY) -> FaceMaskCell
                    {
                        const int z = chunkBaseZ + localZ;
                        const int y = subchunkMinY + localY;
                        if (!(isSolid(x, y, z) && !isSolid(x + 1, y, z)))
                        {
                            return {};
                        }
                        return {
                            true,
                            textureLayerForBlockFace(blockIdAt(x, y, z), BlockFace::Side),
                            positiveXFaceAo(x, y, z),
                        };
                    },
                    [&](int localZ, int localY, int width, int height, const FaceMaskCell& cell)
                    {
                        const float bottom = static_cast<float>(subchunkMinY + localY);
                        const float top = static_cast<float>(subchunkMinY + localY + height);
                        const float z0 = static_cast<float>(chunkBaseZ + localZ);
                        const float z1 = static_cast<float>(chunkBaseZ + localZ + width);
                        addSideFace(
                            localY,
                            {{
                            {faceX, bottom, z0},
                            {faceX, top, z0},
                            {faceX, top, z1},
                            {faceX, bottom, z1},
                        }},
                        width,
                        height,
                        cell.textureLayer,
                        cell.ao);
                    });

                const float oppositeFaceX = static_cast<float>(x);
                emitGreedyRectangles(
                    kChunkSizeZ,
                    kSubchunkSize,
                    [&](int localZ, int localY) -> FaceMaskCell
                    {
                        const int z = chunkBaseZ + localZ;
                        const int y = subchunkMinY + localY;
                        if (!(isSolid(x, y, z) && !isSolid(x - 1, y, z)))
                        {
                            return {};
                        }
                        return {
                            true,
                            textureLayerForBlockFace(blockIdAt(x, y, z), BlockFace::Side),
                            negativeXFaceAo(x, y, z),
                        };
                    },
                    [&](int localZ, int localY, int width, int height, const FaceMaskCell& cell)
                    {
                        const float bottom = static_cast<float>(subchunkMinY + localY);
                        const float top = static_cast<float>(subchunkMinY + localY + height);
                        const float z0 = static_cast<float>(chunkBaseZ + localZ);
                        const float z1 = static_cast<float>(chunkBaseZ + localZ + width);
                        addSideFace(
                            localY,
                            {{
                                {oppositeFaceX, bottom, z1},
                                {oppositeFaceX, top, z1},
                            {oppositeFaceX, top, z0},
                            {oppositeFaceX, bottom, z0},
                        }},
                        width,
                        height,
                        cell.textureLayer,
                        cell.ao);
                    });
            }

            for (int localZ = 0; localZ < kChunkSizeZ; ++localZ)
            {
                const int z = chunkBaseZ + localZ;
                const float faceZ = static_cast<float>(z + 1);
                emitGreedyRectangles(
                    kChunkSizeX,
                    kSubchunkSize,
                    [&](int localX, int localY) -> FaceMaskCell
                    {
                        const int x = chunkBaseX + localX;
                        const int y = subchunkMinY + localY;
                        if (!(isSolid(x, y, z) && !isSolid(x, y, z + 1)))
                        {
                            return {};
                        }
                        return {
                            true,
                            textureLayerForBlockFace(blockIdAt(x, y, z), BlockFace::Side),
                            positiveZFaceAo(x, y, z),
                        };
                    },
                    [&](int localX, int localY, int width, int height, const FaceMaskCell& cell)
                    {
                        const float bottom = static_cast<float>(subchunkMinY + localY);
                        const float top = static_cast<float>(subchunkMinY + localY + height);
                        const float x0 = static_cast<float>(chunkBaseX + localX);
                        const float x1 = static_cast<float>(chunkBaseX + localX + width);
                        addSideFace(
                            localY,
                            {{
                                {x1, bottom, faceZ},
                                {x1, top, faceZ},
                            {x0, top, faceZ},
                            {x0, bottom, faceZ},
                        }},
                        width,
                        height,
                        cell.textureLayer,
                        cell.ao);
                    });

                const float oppositeFaceZ = static_cast<float>(z);
                emitGreedyRectangles(
                    kChunkSizeX,
                    kSubchunkSize,
                    [&](int localX, int localY) -> FaceMaskCell
                    {
                        const int x = chunkBaseX + localX;
                        const int y = subchunkMinY + localY;
                        if (!(isSolid(x, y, z) && !isSolid(x, y, z - 1)))
                        {
                            return {};
                        }
                        return {
                            true,
                            textureLayerForBlockFace(blockIdAt(x, y, z), BlockFace::Side),
                            negativeZFaceAo(x, y, z),
                        };
                    },
                    [&](int localX, int localY, int width, int height, const FaceMaskCell& cell)
                    {
                        const float bottom = static_cast<float>(subchunkMinY + localY);
                        const float top = static_cast<float>(subchunkMinY + localY + height);
                        const float x0 = static_cast<float>(chunkBaseX + localX);
                        const float x1 = static_cast<float>(chunkBaseX + localX + width);
                        addSideFace(
                            localY,
                            {{
                                {x0, bottom, oppositeFaceZ},
                                {x0, top, oppositeFaceZ},
                            {x1, top, oppositeFaceZ},
                            {x1, bottom, oppositeFaceZ},
                        }},
                        width,
                        height,
                        cell.textureLayer,
                        cell.ao);
                    });
            }

        appendSubchunkMesh(
            chunkX,
            chunkZ,
            subchunkY,
            verticesByLocalY,
            indicesByLocalY,
            subchunkVertices,
            subchunkIndices,
            subchunkDraws);

        return {
            chunk,
            subchunkY,
            request.generation,
            std::move(subchunkVertices),
            std::move(subchunkIndices),
        };
    }

    void appendSubchunkMesh(
        int chunkX,
        int chunkZ,
        int subchunkY,
        const std::array<std::vector<BlockVertex>, kSubchunkSize>& verticesByLocalY,
        const std::array<std::vector<std::uint32_t>, kSubchunkSize>& indicesByLocalY,
        std::vector<BlockVertex>& chunkVertices,
        std::vector<std::uint32_t>& chunkIndices,
        std::vector<SubchunkDraw>& subchunkDraws)
    {
        std::size_t vertexCount = 0;
        std::size_t indexCount = 0;
        for (int localY = 0; localY < kSubchunkSize; ++localY)
        {
            vertexCount += verticesByLocalY[static_cast<std::size_t>(localY)].size();
            indexCount += indicesByLocalY[static_cast<std::size_t>(localY)].size();
        }

        if (indexCount == 0)
        {
            return;
        }

        const std::uint32_t vertexOffset = static_cast<std::uint32_t>(chunkVertices.size());
        const std::uint32_t firstIndex = static_cast<std::uint32_t>(chunkIndices.size());
        std::uint32_t subchunkVertexCount = 0;

        for (int localY = 0; localY < kSubchunkSize; ++localY)
        {
            const auto& sourceVertices = verticesByLocalY[static_cast<std::size_t>(localY)];
            const auto& sourceIndices = indicesByLocalY[static_cast<std::size_t>(localY)];
            const std::uint32_t baseVertex = subchunkVertexCount;

            chunkVertices.insert(chunkVertices.end(), sourceVertices.begin(), sourceVertices.end());
            for (std::uint32_t index : sourceIndices)
            {
                chunkIndices.push_back(baseVertex + index);
            }
            subchunkVertexCount += static_cast<std::uint32_t>(sourceVertices.size());
        }

        SubchunkDraw draw{};
        draw.chunkX = chunkX;
        draw.chunkZ = chunkZ;
        draw.subchunkY = subchunkY;
        draw.range.vertexCount = static_cast<std::uint32_t>(vertexCount);
        draw.range.firstIndex = firstIndex;
        draw.range.indexCount = static_cast<std::uint32_t>(indexCount);
        draw.range.vertexOffset = static_cast<std::int32_t>(vertexOffset);
        subchunkDraws.push_back(draw);
    }

    void uploadChunkMesh(
        int chunkX,
        int chunkZ,
        const std::vector<BlockVertex>& vertices,
        const std::vector<std::uint32_t>& indices,
        const std::vector<SubchunkDraw>& subchunkDraws)
    {
        if (indices.empty())
        {
            return;
        }

        ChunkMesh mesh{};
        mesh.chunkX = chunkX;
        mesh.chunkZ = chunkZ;
        mesh.vertexCount = static_cast<std::uint32_t>(vertices.size());
        mesh.indexCount = static_cast<std::uint32_t>(indices.size());
        mesh.subchunks = subchunkDraws;
        meshVertexCount_ += vertices.size();
        meshIndexCount_ += indices.size();

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

        chunkMeshes_.push_back(std::move(mesh));
    }

    ChunkBuildResult assembleChunkMesh(ChunkCoord coord, PendingChunkMesh& pendingMesh)
    {
        ChunkBuildResult result{};
        result.coord = coord;

        std::size_t vertexCount = 0;
        std::size_t indexCount = 0;
        for (int subchunkY = 0; subchunkY < kSubchunksPerChunk; ++subchunkY)
        {
            vertexCount += pendingMesh.vertices[static_cast<std::size_t>(subchunkY)].size();
            indexCount += pendingMesh.indices[static_cast<std::size_t>(subchunkY)].size();
        }

        result.vertices.reserve(vertexCount);
        result.indices.reserve(indexCount);
        result.subchunks.reserve(kSubchunksPerChunk);

        for (int subchunkY = 0; subchunkY < kSubchunksPerChunk; ++subchunkY)
        {
            auto& sourceVertices = pendingMesh.vertices[static_cast<std::size_t>(subchunkY)];
            auto& sourceIndices = pendingMesh.indices[static_cast<std::size_t>(subchunkY)];
            if (sourceIndices.empty())
            {
                continue;
            }

            SubchunkDraw draw{};
            draw.chunkX = coord.x;
            draw.chunkZ = coord.z;
            draw.subchunkY = subchunkY;
            draw.range.vertexCount = static_cast<std::uint32_t>(sourceVertices.size());
            draw.range.firstIndex = static_cast<std::uint32_t>(result.indices.size());
            draw.range.indexCount = static_cast<std::uint32_t>(sourceIndices.size());
            draw.range.vertexOffset = static_cast<std::int32_t>(result.vertices.size());

            result.vertices.insert(result.vertices.end(), sourceVertices.begin(), sourceVertices.end());
            result.indices.insert(result.indices.end(), sourceIndices.begin(), sourceIndices.end());
            result.subchunks.push_back(draw);
        }

        return result;
    }

    std::size_t processCompletedSubchunkBuilds()
    {
        std::deque<SubchunkBuildResult> completedSubchunks;
        {
            std::lock_guard<std::mutex> lock(chunkBuildMutex_);
            completedSubchunks.swap(completedSubchunkBuilds_);
        }

        std::size_t processedCount = 0;
        while (!completedSubchunks.empty())
        {
            SubchunkBuildResult result = std::move(completedSubchunks.front());
            completedSubchunks.pop_front();
            ++processedCount;
            if (result.subchunkY < 0 || result.subchunkY >= kSubchunksPerChunk)
            {
                continue;
            }

            const auto generationIt = queuedChunkGenerations_.find(result.coord);
            if (generationIt == queuedChunkGenerations_.end() || generationIt->second != result.generation)
            {
                continue;
            }
            if (!isChunkInCurrentLoadRange(result.coord) || loadedChunks_.contains(result.coord))
            {
                continue;
            }

            auto [pendingIt, inserted] = pendingChunkMeshes_.try_emplace(result.coord);
            PendingChunkMesh& pendingMesh = pendingIt->second;
            if (inserted)
            {
                pendingMesh.generation = result.generation;
            }
            if (pendingMesh.generation != result.generation)
            {
                continue;
            }

            const std::size_t subchunkIndex = static_cast<std::size_t>(result.subchunkY);
            if (pendingMesh.received[subchunkIndex])
            {
                continue;
            }

            pendingMesh.vertices[subchunkIndex] = std::move(result.vertices);
            pendingMesh.indices[subchunkIndex] = std::move(result.indices);
            pendingMesh.received[subchunkIndex] = true;
            ++pendingMesh.completedSubchunks;

            if (pendingMesh.completedSubchunks == kSubchunksPerChunk)
            {
                completedChunkBuilds_.push_back(assembleChunkMesh(result.coord, pendingMesh));
                pendingChunkMeshes_.erase(pendingIt);
            }
        }
        return processedCount;
    }

    int processCompletedChunkUploads()
    {
        int uploadedCount = 0;
        for (int uploaded = 0; uploaded < chunkUploadsPerFrame_; ++uploaded)
        {
            ChunkBuildResult result{};
            if (completedChunkBuilds_.empty())
            {
                return uploadedCount;
            }

            result = std::move(completedChunkBuilds_.front());
            completedChunkBuilds_.pop_front();

            queuedChunks_.erase(result.coord);
            queuedChunkGenerations_.erase(result.coord);
            pendingChunkMeshes_.erase(result.coord);
            if (!isChunkInCurrentLoadRange(result.coord) || loadedChunks_.contains(result.coord))
            {
                continue;
            }

            uploadChunkMesh(
                result.coord.x,
                result.coord.z,
                result.vertices,
                result.indices,
                result.subchunks);
            loadedChunks_.insert(result.coord);
            ++uploadedCount;
        }
        return uploadedCount;
    }

    void processCompletedChunkBuilds()
    {
        processCompletedSubchunkBuilds();
        processCompletedChunkUploads();
    }

    std::pair<std::size_t, std::size_t> chunkBuildQueueSizes()
    {
        std::lock_guard<std::mutex> lock(chunkBuildMutex_);
        return {pendingChunkBuilds_.size(), completedSubchunkBuilds_.size() + completedChunkBuilds_.size()};
    }

    void createDebugFontAtlasTexture()
    {
        AddFontResourceExW(sourcePathWide(L"/assets/fonts/VCR_OSD_MONO.ttf").c_str(), FR_PRIVATE, nullptr);
        debugFontAtlasTexture_ = createTextureFromPixels(
            renderDebugGlyphAtlas(),
            kDebugFontAtlasWidth,
            kDebugFontAtlasHeight,
            VK_FORMAT_R8G8B8A8_UNORM);
    }

    Texture createTexture(const std::wstring& path)
    {
        ImagePixels pixels = loadPngWithWic(path);
        return createTextureFromPixels(pixels.rgba, pixels.width, pixels.height, VK_FORMAT_R8G8B8A8_SRGB);
    }

    Texture createTextureFromPixels(
        const std::vector<std::uint8_t>& pixels,
        std::uint32_t width,
        std::uint32_t height,
        VkFormat format)
    {
        const VkDeviceSize imageSize = static_cast<VkDeviceSize>(pixels.size());
        Buffer stagingBuffer = createBuffer(
            imageSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        void* data = nullptr;
        vkMapMemory(device_, stagingBuffer.memory, 0, imageSize, 0, &data);
        std::memcpy(data, pixels.data(), static_cast<std::size_t>(imageSize));
        vkUnmapMemory(device_, stagingBuffer.memory);

        Texture texture{};
        texture.format = format;
        texture.width = width;
        texture.height = height;
        createImage(
            width,
            height,
            format,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            texture.image,
            texture.memory);

        transitionImageLayout(
            texture.image,
            format,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        copyBufferToImage(stagingBuffer.buffer, texture.image, width, height);
        transitionImageLayout(
            texture.image,
            format,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        deferUploadStagingBuffer(stagingBuffer);
        texture.view = createImageView(texture.image, format);
        texture.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        return texture;
    }

    Texture createTextureArrayFromPixels(const std::vector<ImagePixels>& layers, VkFormat format)
    {
        if (layers.empty())
        {
            throw std::runtime_error("Cannot create an empty Vulkan texture array.");
        }

        const std::uint32_t width = layers.front().width;
        const std::uint32_t height = layers.front().height;
        const VkDeviceSize layerSize = static_cast<VkDeviceSize>(width) * height * 4;
        std::vector<std::uint8_t> pixels;
        pixels.reserve(static_cast<std::size_t>(layerSize) * layers.size());
        for (const ImagePixels& layer : layers)
        {
            if (layer.width != width || layer.height != height ||
                layer.rgba.size() != static_cast<std::size_t>(layerSize))
            {
                throw std::runtime_error("Block texture array layers must have the same size.");
            }
            pixels.insert(pixels.end(), layer.rgba.begin(), layer.rgba.end());
        }

        const VkDeviceSize imageSize = static_cast<VkDeviceSize>(pixels.size());
        Buffer stagingBuffer = createBuffer(
            imageSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        void* data = nullptr;
        vkMapMemory(device_, stagingBuffer.memory, 0, imageSize, 0, &data);
        std::memcpy(data, pixels.data(), static_cast<std::size_t>(imageSize));
        vkUnmapMemory(device_, stagingBuffer.memory);

        Texture texture{};
        texture.format = format;
        texture.width = width;
        texture.height = height;
        const std::uint32_t layerCount = static_cast<std::uint32_t>(layers.size());
        createImage(
            width,
            height,
            format,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            texture.image,
            texture.memory,
            layerCount);

        transitionImageLayout(
            texture.image,
            format,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            layerCount);
        copyBufferToImageArray(stagingBuffer.buffer, texture.image, width, height, layerCount);
        transitionImageLayout(
            texture.image,
            format,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            layerCount);

        deferUploadStagingBuffer(stagingBuffer);
        texture.view = createImageView(
            texture.image,
            format,
            VK_IMAGE_ASPECT_COLOR_BIT,
            VK_IMAGE_VIEW_TYPE_2D_ARRAY,
            layerCount);
        texture.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        return texture;
    }

    std::vector<std::uint8_t> renderDebugGlyphAtlas()
    {
        const int textureWidth = static_cast<int>(kDebugFontAtlasWidth);
        const int textureHeight = static_cast<int>(kDebugFontAtlasHeight);

        BITMAPINFO bitmapInfo{};
        bitmapInfo.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
        bitmapInfo.bmiHeader.biWidth = textureWidth;
        bitmapInfo.bmiHeader.biHeight = -textureHeight;
        bitmapInfo.bmiHeader.biPlanes = 1;
        bitmapInfo.bmiHeader.biBitCount = 32;
        bitmapInfo.bmiHeader.biCompression = BI_RGB;

        void* bitmapBits = nullptr;
        HDC screenDc = GetDC(nullptr);
        HDC memoryDc = CreateCompatibleDC(screenDc);
        HBITMAP bitmap = CreateDIBSection(screenDc, &bitmapInfo, DIB_RGB_COLORS, &bitmapBits, nullptr, 0);
        ReleaseDC(nullptr, screenDc);

        if (memoryDc == nullptr || bitmap == nullptr || bitmapBits == nullptr)
        {
            if (bitmap != nullptr)
            {
                DeleteObject(bitmap);
            }
            if (memoryDc != nullptr)
            {
                DeleteDC(memoryDc);
            }
            throw std::runtime_error("Failed to create debug glyph atlas bitmap.");
        }

        std::memset(bitmapBits, 0, static_cast<std::size_t>(textureWidth * textureHeight * 4));
        HGDIOBJ oldBitmap = SelectObject(memoryDc, bitmap);
        HFONT font = CreateFontW(
            -42,
            0,
            0,
            0,
            FW_NORMAL,
            FALSE,
            FALSE,
            FALSE,
            DEFAULT_CHARSET,
            OUT_DEFAULT_PRECIS,
            CLIP_DEFAULT_PRECIS,
            ANTIALIASED_QUALITY,
            DEFAULT_PITCH | FF_DONTCARE,
            L"VCR OSD Mono");
        bool ownsFont = true;
        if (font == nullptr)
        {
            font = static_cast<HFONT>(GetStockObject(DEFAULT_GUI_FONT));
            ownsFont = false;
        }
        HGDIOBJ oldFont = SelectObject(memoryDc, font);

        SetBkMode(memoryDc, TRANSPARENT);
        SetTextColor(memoryDc, RGB(255, 255, 255));
        TEXTMETRICW textMetric{};
        GetTextMetricsW(memoryDc, &textMetric);

        const auto* bgra = static_cast<const std::uint8_t*>(bitmapBits);

        for (int glyphIndex = 0; glyphIndex < kDebugGlyphCount; ++glyphIndex)
        {
            const wchar_t character = static_cast<wchar_t>(kDebugGlyphFirst + glyphIndex);
            const int column = glyphIndex % kDebugGlyphColumns;
            const int row = glyphIndex / kDebugGlyphColumns;
            const int x = column * kDebugGlyphCellWidth;
            const int y = row * kDebugGlyphCellHeight;
            SIZE glyphSize{};
            GetTextExtentPoint32W(memoryDc, &character, 1, &glyphSize);

            TextOutW(memoryDc, x + 2, y + 2, &character, 1);

            DebugGlyph& glyph = debugGlyphs_[static_cast<std::size_t>(glyphIndex)];
            const int glyphWidth = std::min(kDebugGlyphCellWidth, static_cast<int>(glyphSize.cx) + 4);
            const int glyphHeight = std::min(kDebugGlyphCellHeight, static_cast<int>(textMetric.tmHeight) + 4);
            glyph.u0 = static_cast<float>(x) / static_cast<float>(textureWidth);
            glyph.v0 = static_cast<float>(y) / static_cast<float>(textureHeight);
            glyph.u1 = static_cast<float>(x + glyphWidth) / static_cast<float>(textureWidth);
            glyph.v1 = static_cast<float>(y + glyphHeight) / static_cast<float>(textureHeight);
            glyph.advance = static_cast<float>(std::max(1L, static_cast<LONG>(glyphSize.cx)));
            glyph.width = static_cast<float>(glyphWidth);
            glyph.height = static_cast<float>(glyphHeight);
        }

        std::vector<std::uint8_t> rgba(static_cast<std::size_t>(textureWidth * textureHeight * 4));
        for (int i = 0; i < textureWidth * textureHeight; ++i)
        {
            const std::uint8_t b = bgra[i * 4 + 0];
            const std::uint8_t g = bgra[i * 4 + 1];
            const std::uint8_t r = bgra[i * 4 + 2];
            rgba[i * 4 + 0] = 255;
            rgba[i * 4 + 1] = 255;
            rgba[i * 4 + 2] = 255;
            rgba[i * 4 + 3] = std::max({r, g, b});
        }

        SelectObject(memoryDc, oldFont);
        SelectObject(memoryDc, oldBitmap);
        if (ownsFont)
        {
            DeleteObject(font);
        }
        DeleteObject(bitmap);
        DeleteDC(memoryDc);

        return rgba;
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

        if (vkCreateSampler(device_, &samplerInfo, nullptr, &textureSampler_) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create Vulkan texture sampler.");
        }
    }

    void createUniformBuffer()
    {
        uniformBuffer_ = createBuffer(
            sizeof(SkyUniform),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        vkMapMemory(device_, uniformBuffer_.memory, 0, sizeof(SkyUniform), 0, &uniformMappedMemory_);

        blockUniformBuffer_ = createBuffer(
            sizeof(BlockUniform),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        vkMapMemory(device_, blockUniformBuffer_.memory, 0, sizeof(BlockUniform), 0, &blockUniformMappedMemory_);

        debugTextVertexBuffer_ = createBuffer(
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
    }

    void createDescriptorPool()
    {
        std::array<VkDescriptorPoolSize, 2> poolSizes{};
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[0].descriptorCount = 2;
        poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes[1].descriptorCount = 4;

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<std::uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = 3;

        if (vkCreateDescriptorPool(device_, &poolInfo, nullptr, &descriptorPool_) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create Vulkan descriptor pool.");
        }
    }

    void createDescriptorSet()
    {
        VkDescriptorSetAllocateInfo allocateInfo{};
        allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocateInfo.descriptorPool = descriptorPool_;
        allocateInfo.descriptorSetCount = 1;
        allocateInfo.pSetLayouts = &descriptorSetLayout_;

        if (vkAllocateDescriptorSets(device_, &allocateInfo, &descriptorSet_) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to allocate Vulkan descriptor set.");
        }

        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = uniformBuffer_.buffer;
        bufferInfo.offset = 0;
        bufferInfo.range = sizeof(SkyUniform);

        VkDescriptorImageInfo sunImageInfo{};
        sunImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        sunImageInfo.imageView = sunTexture_.view;
        sunImageInfo.sampler = textureSampler_;

        VkDescriptorImageInfo moonImageInfo{};
        moonImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        moonImageInfo.imageView = moonTexture_.view;
        moonImageInfo.sampler = textureSampler_;

        std::array<VkWriteDescriptorSet, 3> descriptorWrites{};
        descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[0].dstSet = descriptorSet_;
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[0].pBufferInfo = &bufferInfo;

        descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[1].dstSet = descriptorSet_;
        descriptorWrites[1].dstBinding = 1;
        descriptorWrites[1].descriptorCount = 1;
        descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorWrites[1].pImageInfo = &sunImageInfo;

        descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[2].dstSet = descriptorSet_;
        descriptorWrites[2].dstBinding = 2;
        descriptorWrites[2].descriptorCount = 1;
        descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorWrites[2].pImageInfo = &moonImageInfo;

        vkUpdateDescriptorSets(
            device_,
            static_cast<std::uint32_t>(descriptorWrites.size()),
            descriptorWrites.data(),
            0,
            nullptr);
    }

    void createBlockDescriptorSet()
    {
        VkDescriptorSetAllocateInfo allocateInfo{};
        allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocateInfo.descriptorPool = descriptorPool_;
        allocateInfo.descriptorSetCount = 1;
        allocateInfo.pSetLayouts = &blockDescriptorSetLayout_;

        if (vkAllocateDescriptorSets(device_, &allocateInfo, &blockDescriptorSet_) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to allocate Vulkan block descriptor set.");
        }

        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = blockUniformBuffer_.buffer;
        bufferInfo.offset = 0;
        bufferInfo.range = sizeof(BlockUniform);

        VkDescriptorImageInfo imageInfo{};
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageInfo.imageView = blockTextureArray_.view;
        imageInfo.sampler = textureSampler_;

        std::array<VkWriteDescriptorSet, 2> descriptorWrites{};
        descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[0].dstSet = blockDescriptorSet_;
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[0].pBufferInfo = &bufferInfo;

        descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[1].dstSet = blockDescriptorSet_;
        descriptorWrites[1].dstBinding = 1;
        descriptorWrites[1].descriptorCount = 1;
        descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorWrites[1].pImageInfo = &imageInfo;

        vkUpdateDescriptorSets(
            device_,
            static_cast<std::uint32_t>(descriptorWrites.size()),
            descriptorWrites.data(),
            0,
            nullptr);
    }

    void createDebugTextDescriptorSet()
    {
        VkDescriptorSetAllocateInfo allocateInfo{};
        allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocateInfo.descriptorPool = descriptorPool_;
        allocateInfo.descriptorSetCount = 1;
        allocateInfo.pSetLayouts = &debugTextDescriptorSetLayout_;

        if (vkAllocateDescriptorSets(device_, &allocateInfo, &debugTextDescriptorSet_) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to allocate Vulkan debug text descriptor set.");
        }

        VkDescriptorImageInfo imageInfo{};
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageInfo.imageView = debugFontAtlasTexture_.view;
        imageInfo.sampler = textureSampler_;

        VkWriteDescriptorSet descriptorWrite{};
        descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrite.dstSet = debugTextDescriptorSet_;
        descriptorWrite.dstBinding = 0;
        descriptorWrite.descriptorCount = 1;
        descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorWrite.pImageInfo = &imageInfo;

        vkUpdateDescriptorSets(device_, 1, &descriptorWrite, 0, nullptr);
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

    Buffer createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties)
    {
        Buffer buffer{};

        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device_, &bufferInfo, nullptr, &buffer.buffer) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create Vulkan buffer.");
        }

        VkMemoryRequirements memoryRequirements{};
        vkGetBufferMemoryRequirements(device_, buffer.buffer, &memoryRequirements);

        VkMemoryAllocateInfo allocateInfo{};
        allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocateInfo.allocationSize = memoryRequirements.size;
        allocateInfo.memoryTypeIndex = findMemoryType(
            physicalDevice_,
            memoryRequirements.memoryTypeBits,
            properties);

        if (vkAllocateMemory(device_, &allocateInfo, nullptr, &buffer.memory) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to allocate Vulkan buffer memory.");
        }

        vkBindBufferMemory(device_, buffer.buffer, buffer.memory, 0);
        return buffer;
    }

    void createDeviceLocalBuffer(
        const void* sourceData,
        VkDeviceSize size,
        VkBufferUsageFlags usage,
        Buffer& destinationBuffer)
    {
        Buffer stagingBuffer = createBuffer(
            size,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        void* data = nullptr;
        vkMapMemory(device_, stagingBuffer.memory, 0, size, 0, &data);
        std::memcpy(data, sourceData, static_cast<std::size_t>(size));
        vkUnmapMemory(device_, stagingBuffer.memory);

        destinationBuffer = createBuffer(
            size,
            usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        copyBuffer(stagingBuffer.buffer, destinationBuffer.buffer, size);
        deferUploadStagingBuffer(stagingBuffer);
    }

    void createImage(
        std::uint32_t width,
        std::uint32_t height,
        VkFormat format,
        VkImageTiling tiling,
        VkImageUsageFlags usage,
        VkMemoryPropertyFlags properties,
        VkImage& image,
        VkDeviceMemory& imageMemory,
        std::uint32_t arrayLayers = 1)
    {
        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.extent.width = width;
        imageInfo.extent.height = height;
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = arrayLayers;
        imageInfo.format = format;
        imageInfo.tiling = tiling;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage = usage;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateImage(device_, &imageInfo, nullptr, &image) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to create Vulkan image.");
        }

        VkMemoryRequirements memoryRequirements{};
        vkGetImageMemoryRequirements(device_, image, &memoryRequirements);

        VkMemoryAllocateInfo allocateInfo{};
        allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocateInfo.allocationSize = memoryRequirements.size;
        allocateInfo.memoryTypeIndex = findMemoryType(
            physicalDevice_,
            memoryRequirements.memoryTypeBits,
            properties);

        if (vkAllocateMemory(device_, &allocateInfo, nullptr, &imageMemory) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to allocate Vulkan image memory.");
        }

        vkBindImageMemory(device_, image, imageMemory, 0);
    }

    VkCommandBuffer beginSingleTimeCommands()
    {
        VkCommandBufferAllocateInfo allocateInfo{};
        allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocateInfo.commandPool = commandPool_;
        allocateInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        vkAllocateCommandBuffers(device_, &allocateInfo, &commandBuffer);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(commandBuffer, &beginInfo);

        return commandBuffer;
    }

    void endSingleTimeCommands(VkCommandBuffer commandBuffer)
    {
        vkEndCommandBuffer(commandBuffer);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        if (vkQueueSubmit(graphicsQueue_, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to submit Vulkan upload command buffer.");
        }
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

    void copyBuffer(VkBuffer sourceBuffer, VkBuffer destinationBuffer, VkDeviceSize size)
    {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        VkBufferCopy copyRegion{};
        copyRegion.size = size;
        vkCmdCopyBuffer(commandBuffer, sourceBuffer, destinationBuffer, 1, &copyRegion);

        endSingleTimeCommands(commandBuffer);
    }

    void transitionImageLayout(
        VkImage image,
        VkFormat,
        VkImageLayout oldLayout,
        VkImageLayout newLayout,
        std::uint32_t layerCount = 1)
    {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = layerCount;

        VkPipelineStageFlags sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        VkPipelineStageFlags destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;

        if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
            newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
        {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        }
        else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
                 newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        }
        else if (oldLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL &&
                 newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
        {
            barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            sourceStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
            destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        }
        else
        {
            throw std::runtime_error("Unsupported Vulkan image layout transition.");
        }

        vkCmdPipelineBarrier(
            commandBuffer,
            sourceStage,
            destinationStage,
            0,
            0,
            nullptr,
            0,
            nullptr,
            1,
            &barrier);

        endSingleTimeCommands(commandBuffer);
    }

    void copyBufferToImage(VkBuffer buffer, VkImage image, std::uint32_t width, std::uint32_t height)
    {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = {0, 0, 0};
        region.imageExtent = {width, height, 1};

        vkCmdCopyBufferToImage(
            commandBuffer,
            buffer,
            image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &region);

        endSingleTimeCommands(commandBuffer);
    }

    void copyBufferToImageArray(
        VkBuffer buffer,
        VkImage image,
        std::uint32_t width,
        std::uint32_t height,
        std::uint32_t layerCount)
    {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = layerCount;
        region.imageOffset = {0, 0, 0};
        region.imageExtent = {width, height, 1};

        vkCmdCopyBufferToImage(
            commandBuffer,
            buffer,
            image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &region);

        endSingleTimeCommands(commandBuffer);
    }

    void uploadTexturePixels(Texture& texture, const std::vector<std::uint8_t>& pixels)
    {
        const VkDeviceSize imageSize = static_cast<VkDeviceSize>(pixels.size());
        Buffer stagingBuffer = createBuffer(
            imageSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        void* data = nullptr;
        vkMapMemory(device_, stagingBuffer.memory, 0, imageSize, 0, &data);
        std::memcpy(data, pixels.data(), static_cast<std::size_t>(imageSize));
        vkUnmapMemory(device_, stagingBuffer.memory);

        transitionImageLayout(
            texture.image,
            texture.format,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        copyBufferToImage(stagingBuffer.buffer, texture.image, texture.width, texture.height);
        transitionImageLayout(
            texture.image,
            texture.format,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        texture.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        deferUploadStagingBuffer(stagingBuffer);
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
        const float deltaSeconds = static_cast<float>(std::chrono::duration<double>(now - lastFrameTime_).count());
        lastFrameTime_ = now;

        Vec3 movement{};
        const Vec3 forward = cameraForward(camera_.yaw, 0.0f);
        const Vec3 right = cameraRight(camera_.yaw);

        if (glfwGetKey(window_, GLFW_KEY_W) == GLFW_PRESS)
        {
            movement = movement + forward;
        }
        if (glfwGetKey(window_, GLFW_KEY_S) == GLFW_PRESS)
        {
            movement = movement - forward;
        }
        if (glfwGetKey(window_, GLFW_KEY_D) == GLFW_PRESS)
        {
            movement = movement + right;
        }
        if (glfwGetKey(window_, GLFW_KEY_A) == GLFW_PRESS)
        {
            movement = movement - right;
        }
        if (glfwGetKey(window_, GLFW_KEY_SPACE) == GLFW_PRESS)
        {
            movement = movement + Vec3{0.0f, 1.0f, 0.0f};
        }
        if (glfwGetKey(window_, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
            glfwGetKey(window_, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS)
        {
            movement = movement - Vec3{0.0f, 1.0f, 0.0f};
        }

        if (dot(movement, movement) > 0.0f)
        {
            constexpr float flySpeed = 64.0f;
            camera_.position = camera_.position + normalize(movement) * (flySpeed * deltaSeconds);
        }
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

        std::wostringstream fpsText;
        fpsText << L"FPS: " << std::setw(4) << std::setfill(L'0') << fpsInteger
                << L" [" << std::setw(6) << std::setfill(L'0') << std::fixed << std::setprecision(3)
                << frameMs << L"MS]";

        std::wostringstream positionText;
        positionText << L"POS: "
                     << formatFixedWidth(camera_.position.x, 7, 2) << L" / "
                     << formatFixedWidth(camera_.position.y, 7, 2) << L" / "
                     << formatFixedWidth(camera_.position.z, 7, 2);

        const double yawDegrees = static_cast<double>(camera_.yaw) * 180.0 / static_cast<double>(kPi);
        const double pitchDegrees = static_cast<double>(camera_.pitch) * 180.0 / static_cast<double>(kPi);
        std::wostringstream cameraText;
        cameraText << std::showpos << std::fixed << std::setprecision(3)
                   << L"CAM: YAW " << std::setw(8) << yawDegrees
                   << L" PIT " << std::setw(7) << pitchDegrees;

        debugTextOverlay_.lines = {fpsText.str(), positionText.str(), cameraText.str()};

        auto msLine = [](const wchar_t* label, double value)
        {
            std::wostringstream line;
            line << label << L": " << std::fixed << std::setprecision(3)
                 << std::setw(7) << std::setfill(L'0') << value << L"MS";
            return line.str();
        };
        auto countLine = [](const wchar_t* label, std::size_t value)
        {
            std::wostringstream line;
            line << label << L": " << value;
            return line.str();
        };

        debugTextOverlay_.bottomLeftLines = {
            msLine(L"CPU FRAME", frameProfiler_.frameCpuMs),
            msLine(L"FENCE WAIT", frameProfiler_.fenceWaitMs),
            msLine(L"ACQUIRE", frameProfiler_.acquireMs),
            msLine(L"LOAD UPDATE", frameProfiler_.loadUpdateMs),
            countLine(L"LOAD Q/UNLD/UP", frameProfiler_.chunksQueued) +
                L"/" + std::to_wstring(frameProfiler_.chunksUnloaded) +
                L"/" + std::to_wstring(frameProfiler_.chunksLoaded),
            msLine(L"SUBCHUNK DONE", frameProfiler_.subchunkDoneMs),
            countLine(L"SUBCHUNK COUNT", frameProfiler_.subchunkResultsProcessed),
            msLine(L"CHUNK UPLOAD", frameProfiler_.chunkUploadMs),
            msLine(L"UNIFORM", frameProfiler_.uniformMs),
            msLine(L"DEBUG TEXT", frameProfiler_.debugTextMs),
            msLine(L"RECORD", frameProfiler_.recordMs),
            msLine(L"SUBMIT+PRESENT", frameProfiler_.submitPresentMs),
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
        std::vector<DebugTextVertex> vertices;
        vertices.reserve(6000);

        appendDebugTextBlock(vertices, debugTextOverlay_.lines, 12.0f, 10.0f, false);
        appendDebugTextBlock(
            vertices,
            debugTextOverlay_.rightLines,
            static_cast<float>(swapchainExtent_.width) - 12.0f,
            10.0f,
            true);
        const float bottomLeftY = std::max(
            10.0f,
            static_cast<float>(swapchainExtent_.height) -
                10.0f -
                57.0f * static_cast<float>(debugTextOverlay_.bottomLeftLines.size()));
        appendDebugTextBlock(
            vertices,
            debugTextOverlay_.bottomLeftLines,
            12.0f,
            bottomLeftY,
            false);

        if (vertices.size() > kMaxDebugTextVertices)
        {
            vertices.resize(kMaxDebugTextVertices);
        }

        debugTextVertexCount_ = static_cast<std::uint32_t>(vertices.size());
        if (debugTextVertexCount_ > 0)
        {
            std::memcpy(
                debugTextVertexMappedMemory_,
                vertices.data(),
                sizeof(DebugTextVertex) * vertices.size());
        }
    }

    void appendDebugTextBlock(
        std::vector<DebugTextVertex>& vertices,
        const std::vector<std::wstring>& lines,
        float x,
        float y,
        bool alignRight) const
    {
        for (const std::wstring& line : lines)
        {
            const float lineWidth = measureDebugTextLine(line);
            const float lineX = alignRight ? x - lineWidth : x;

            appendDebugTextLine(vertices, line, lineX - 1.0f, y, {0.0f, 0.0f, 0.0f, 1.0f});
            appendDebugTextLine(vertices, line, lineX + 1.0f, y, {0.0f, 0.0f, 0.0f, 1.0f});
            appendDebugTextLine(vertices, line, lineX, y - 1.0f, {0.0f, 0.0f, 0.0f, 1.0f});
            appendDebugTextLine(vertices, line, lineX, y + 1.0f, {0.0f, 0.0f, 0.0f, 1.0f});
            appendDebugTextLine(vertices, line, lineX, y, {1.0f, 1.0f, 1.0f, 1.0f});

            y += 57.0f;
        }
    }

    float measureDebugTextLine(const std::wstring& line) const
    {
        float width = 0.0f;
        for (wchar_t character : line)
        {
            width += glyphForCharacter(character).advance;
        }
        return width;
    }

    const DebugGlyph& glyphForCharacter(wchar_t character) const
    {
        if (character < kDebugGlyphFirst || character > kDebugGlyphLast)
        {
            character = L'?';
        }

        return debugGlyphs_[static_cast<std::size_t>(character - kDebugGlyphFirst)];
    }

    void appendDebugTextLine(
        std::vector<DebugTextVertex>& vertices,
        const std::wstring& line,
        float x,
        float y,
        std::array<float, 4> color) const
    {
        for (wchar_t character : line)
        {
            const DebugGlyph& glyph = glyphForCharacter(character);
            appendDebugGlyphQuad(vertices, x, y, glyph, color);
            x += glyph.advance;
        }
    }

    void appendDebugGlyphQuad(
        std::vector<DebugTextVertex>& vertices,
        float x,
        float y,
        const DebugGlyph& glyph,
        std::array<float, 4> color) const
    {
        const float x0 = -1.0f + (2.0f * x / static_cast<float>(swapchainExtent_.width));
        const float y0 = 1.0f - (2.0f * y / static_cast<float>(swapchainExtent_.height));
        const float x1 = -1.0f + (2.0f * (x + glyph.width) / static_cast<float>(swapchainExtent_.width));
        const float y1 = 1.0f - (2.0f * (y + glyph.height) / static_cast<float>(swapchainExtent_.height));

        const auto makeVertex = [&](float px, float py, float u, float v) -> DebugTextVertex
        {
            return {{px, py}, {u, v}, {color[0], color[1], color[2], color[3]}};
        };

        vertices.push_back(makeVertex(x0, y0, glyph.u0, glyph.v0));
        vertices.push_back(makeVertex(x1, y0, glyph.u1, glyph.v0));
        vertices.push_back(makeVertex(x1, y1, glyph.u1, glyph.v1));
        vertices.push_back(makeVertex(x0, y0, glyph.u0, glyph.v0));
        vertices.push_back(makeVertex(x1, y1, glyph.u1, glyph.v1));
        vertices.push_back(makeVertex(x0, y1, glyph.u0, glyph.v1));
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
            blockDrawCalls += mesh.subchunks.size();
        }
        const std::size_t debugDrawCalls = debugTextVisible_ ? 1 : 0;
        const std::size_t drawCalls = 1 + blockDrawCalls + debugDrawCalls;
        const std::size_t totalVertices = 12 + meshVertexCount_ + debugTextVertexCount_;
        const std::size_t totalIndices = meshIndexCount_;
        const std::size_t triangles = meshIndexCount_ / 3 + 4 + debugTextVertexCount_ / 3;
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
            L"CHUNKS: " + std::to_wstring(loadedChunks_.size()),
            L"LOAD RADIUS: " + std::to_wstring(chunkLoadRadius_),
            L"UPLOADS/FRAME: " + std::to_wstring(chunkUploadsPerFrame_),
            L"BUILD THREADS: " + std::to_wstring(chunkBuildThreads_),
            L"BUILD JOBS: " + std::to_wstring(pendingBuilds),
            L"BUILD DONE: " + std::to_wstring(completedBuilds),
            L"DEFERRED DESTROYS: " + std::to_wstring(deferredChunkBufferDestroys_.size()),
            L"DEFERRED UPLOADS: " + std::to_wstring(deferredUploadCleanups_.size()),
            L"VERTS: " + std::to_wstring(totalVertices),
            L"INDICES: " + std::to_wstring(totalIndices),
            L"TRIS: " + std::to_wstring(triangles),
        };
    }

    std::uint64_t getProcessRamUsageMb() const
    {
        PROCESS_MEMORY_COUNTERS_EX counters{};
        counters.cb = sizeof(counters);
        if (GetProcessMemoryInfo(
                GetCurrentProcess(),
                reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
                sizeof(counters)) == 0)
        {
            return 0;
        }

        return static_cast<std::uint64_t>(counters.WorkingSetSize / (1024ull * 1024ull));
    }

    std::uint64_t getTotalRamMb() const
    {
        MEMORYSTATUSEX memoryStatus{};
        memoryStatus.dwLength = sizeof(memoryStatus);
        if (GlobalMemoryStatusEx(&memoryStatus) == 0)
        {
            return 0;
        }

        return static_cast<std::uint64_t>(memoryStatus.ullTotalPhys / (1024ull * 1024ull));
    }

    void updateUniformBuffer()
    {
        SkyUniform uniform{};
        const float aspect = static_cast<float>(swapchainExtent_.width) / static_cast<float>(swapchainExtent_.height);
        const float fovY = 70.0f * kPi / 180.0f;

        uniform.camera[0] = camera_.yaw;
        uniform.camera[1] = camera_.pitch;
        uniform.camera[2] = aspect;
        uniform.camera[3] = std::tan(fovY * 0.5f);

        uniform.sunDirection[0] = 1.0f;
        uniform.sunDirection[1] = 0.0f;
        uniform.sunDirection[2] = 0.0f;
        uniform.sunDirection[3] = 0.0f;

        const float invLength = 1.0f / std::sqrt(
            uniform.sunDirection[0] * uniform.sunDirection[0] +
            uniform.sunDirection[1] * uniform.sunDirection[1]);
        uniform.sunDirection[0] *= invLength;
        uniform.sunDirection[1] *= invLength;

        uniform.moonDirection[0] = -uniform.sunDirection[0];
        uniform.moonDirection[1] = -uniform.sunDirection[1];
        uniform.moonDirection[2] = -uniform.sunDirection[2];
        uniform.moonDirection[3] = 0.0f;

        uniform.spriteScale[0] = 0.16f;
        uniform.spriteScale[1] = 0.14f;
        uniform.spriteScale[2] = 0.0f;
        uniform.spriteScale[3] = 0.0f;

        std::memcpy(uniformMappedMemory_, &uniform, sizeof(uniform));

        BlockUniform blockUniform{};
        const Mat4 view = makeViewMatrix(camera_.position, camera_.yaw, camera_.pitch);
        const Mat4 projection = makePerspectiveMatrix(fovY, aspect, 0.05f, 1200.0f);
        const Mat4 viewProjection = multiply(projection, view);
        std::memcpy(blockUniform.viewProjection, viewProjection.m, sizeof(blockUniform.viewProjection));
        std::memcpy(blockUniformMappedMemory_, &blockUniform, sizeof(blockUniform));
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
        clearValues[0].color = {{0.46f, 0.72f, 1.0f, 1.0f}};
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
        for (const ChunkMesh& mesh : chunkMeshes_)
        {
            VkBuffer vertexBuffers[] = {mesh.vertexBuffer.buffer};
            VkDeviceSize offsets[] = {0};
            vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
            vkCmdBindIndexBuffer(commandBuffer, mesh.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
            for (const SubchunkDraw& draw : mesh.subchunks)
            {
                vkCmdDrawIndexed(
                    commandBuffer,
                    draw.range.indexCount,
                    1,
                    draw.range.firstIndex,
                    draw.range.vertexOffset,
                    0);
            }
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
        createImageViews();
        createRenderPass();
        createDepthResources();
        createGraphicsPipeline();
        createBlockPipeline();
        createDebugTextPipeline();
        createFramebuffers();
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

        for (VkImageView imageView : swapchainImageViews_)
        {
            vkDestroyImageView(device_, imageView, nullptr);
        }
        swapchainImageViews_.clear();

        destroyTexture(depthTexture_);

        if (swapchain_ != VK_NULL_HANDLE)
        {
            vkDestroySwapchainKHR(device_, swapchain_, nullptr);
            swapchain_ = VK_NULL_HANDLE;
        }
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

    void destroyBuffer(Buffer& buffer)
    {
        if (buffer.buffer != VK_NULL_HANDLE)
        {
            vkDestroyBuffer(device_, buffer.buffer, nullptr);
            buffer.buffer = VK_NULL_HANDLE;
        }

        if (buffer.memory != VK_NULL_HANDLE)
        {
            vkFreeMemory(device_, buffer.memory, nullptr);
            buffer.memory = VK_NULL_HANDLE;
        }
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

            destroyBuffer(it->vertexBuffer);
            destroyBuffer(it->indexBuffer);
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

            destroyBuffer(it->stagingBuffer);
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
                frameCounter_ + static_cast<std::uint64_t>(kMaxFramesInFlight),
            });
            meshVertexCount_ -= std::min(meshVertexCount_, static_cast<std::size_t>(it->vertexCount));
            meshIndexCount_ -= std::min(meshIndexCount_, static_cast<std::size_t>(it->indexCount));
            it = chunkMeshes_.erase(it);
        }
    }

    void destroyAllChunkMeshes()
    {
        for (ChunkMesh& mesh : chunkMeshes_)
        {
            destroyBuffer(mesh.vertexBuffer);
            destroyBuffer(mesh.indexBuffer);
        }
        chunkMeshes_.clear();
        collectDeferredChunkBufferDestroys(true);
        loadedChunks_.clear();
        queuedChunks_.clear();
        desiredChunks_.clear();
        queuedChunkGenerations_.clear();
        pendingChunkMeshes_.clear();
        loadedCenterChunkX_ = std::numeric_limits<int>::min();
        loadedCenterChunkZ_ = std::numeric_limits<int>::min();
        meshVertexCount_ = 0;
        meshIndexCount_ = 0;
    }

    void destroyTexture(Texture& texture)
    {
        if (texture.view != VK_NULL_HANDLE)
        {
            vkDestroyImageView(device_, texture.view, nullptr);
            texture.view = VK_NULL_HANDLE;
        }

        if (texture.image != VK_NULL_HANDLE)
        {
            vkDestroyImage(device_, texture.image, nullptr);
            texture.image = VK_NULL_HANDLE;
        }

        if (texture.memory != VK_NULL_HANDLE)
        {
            vkFreeMemory(device_, texture.memory, nullptr);
            texture.memory = VK_NULL_HANDLE;
        }
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
