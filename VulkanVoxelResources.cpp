#include "BlockRegistry.h"
#include "VulkanVoxel.h"

#define NOMINMAX
#include <Windows.h>
#include <dxgi1_4.h>
#include <wingdi.h>
#include <wincodec.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#ifndef ASSET_DIR
#define ASSET_DIR "."
#endif

#ifndef CHARACTER_MESH_PATH
#define CHARACTER_MESH_PATH "./character_mesh.bin"
#endif

#ifndef CHARACTER_TEXTURE_PATH
#define CHARACTER_TEXTURE_PATH "./character_texture.bin"
#endif

namespace {

constexpr int kMaxFramesInFlight = 2;
constexpr std::size_t kMaxOverlayVertexCount = 4194304;
constexpr std::size_t kMaxCelestialVertexCount = 12;
constexpr float kWorldVerticalFovDegrees = 70.0f;
constexpr std::size_t kChunkLoadWorkerCount = 3;
constexpr std::size_t kMeshWorkerCount = 2;
constexpr std::size_t kChunkGenerationBudgetPerPass = 12;
constexpr std::size_t kMeshBuildBudgetPerPass = 12;
constexpr std::size_t kWorldRenderRemovalBudgetPerConsume = 16;
constexpr std::size_t kWorldRenderUploadBudgetPerConsume = 8;
constexpr auto kWorldMeshWorkerYieldDelay = std::chrono::milliseconds(1);

struct LoadedImage {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<std::uint8_t> pixels;
};

struct SerializedPlayerVertex {
    float position[3];
    float uv[2];
};

bool PendingChunkIdEquals(const PendingChunkId& lhs, const PendingChunkId& rhs) {
    return lhs.chunkX == rhs.chunkX && lhs.chunkZ == rhs.chunkZ;
}

void ErasePendingChunkId(
    std::vector<PendingChunkId>& chunks,
    std::unordered_set<PendingChunkId, PendingChunkIdHash>& chunkSet,
    const PendingChunkId& target
) {
    if (chunkSet.erase(target) == 0) {
        return;
    }

    chunks.erase(
        std::remove_if(
            chunks.begin(),
            chunks.end(),
            [&target](const PendingChunkId& candidate) { return PendingChunkIdEquals(candidate, target); }
        ),
        chunks.end()
    );
}

void EnqueuePendingChunkId(
    std::vector<PendingChunkId>& chunks,
    std::unordered_set<PendingChunkId, PendingChunkIdHash>& chunkSet,
    const PendingChunkId& target
) {
    if (chunkSet.insert(target).second) {
        chunks.push_back(target);
    }
}

void ErasePendingChunkUpload(
    std::vector<ChunkMeshBatchData>& uploads,
    std::unordered_set<PendingChunkId, PendingChunkIdHash>& uploadSet,
    const PendingChunkId& target
) {
    if (uploadSet.erase(target) == 0) {
        return;
    }

    uploads.erase(
        std::remove_if(
            uploads.begin(),
            uploads.end(),
            [&target](const ChunkMeshBatchData& candidate) { return PendingChunkIdEquals(candidate.id, target); }
        ),
        uploads.end()
    );
}

void EnqueuePendingChunkUpload(
    std::vector<ChunkMeshBatchData>& uploads,
    std::unordered_set<PendingChunkId, PendingChunkIdHash>& uploadSet,
    ChunkMeshBatchData&& upload
) {
    if (uploadSet.insert(upload.id).second) {
        uploads.push_back(std::move(upload));
        return;
    }

    for (ChunkMeshBatchData& existing : uploads) {
        if (PendingChunkIdEquals(existing.id, upload.id)) {
            existing = std::move(upload);
            return;
        }
    }
}

void MergeWorldRenderUpdate(WorldRenderUpdate& destination, WorldRenderUpdate&& source) {
    destination.loadedChunkCount = source.loadedChunkCount;

    auto eraseChunk = [](std::vector<PendingChunkId>& chunks, const PendingChunkId& target) {
        chunks.erase(
            std::remove_if(
                chunks.begin(),
                chunks.end(),
                [&target](const PendingChunkId& candidate) { return PendingChunkIdEquals(candidate, target); }
            ),
            chunks.end()
        );
    };

    auto eraseUpload = [](std::vector<ChunkMeshBatchData>& uploads, const PendingChunkId& target) {
        uploads.erase(
            std::remove_if(
                uploads.begin(),
                uploads.end(),
                [&target](const ChunkMeshBatchData& candidate) { return PendingChunkIdEquals(candidate.id, target); }
            ),
            uploads.end()
        );
    };

    auto containsChunk = [](const std::vector<PendingChunkId>& chunks, const PendingChunkId& target) {
        return std::any_of(
            chunks.begin(),
            chunks.end(),
            [&target](const PendingChunkId& candidate) { return PendingChunkIdEquals(candidate, target); }
        );
    };

    for (const PendingChunkId& removal : source.removals) {
        eraseUpload(destination.uploads, removal);
        if (!containsChunk(destination.removals, removal)) {
            destination.removals.push_back(removal);
        }
    }

    for (ChunkMeshBatchData& upload : source.uploads) {
        eraseChunk(destination.removals, upload.id);

        bool replaced = false;
        for (ChunkMeshBatchData& existing : destination.uploads) {
            if (PendingChunkIdEquals(existing.id, upload.id)) {
                existing = std::move(upload);
                replaced = true;
                break;
            }
        }
        if (!replaced) {
            destination.uploads.push_back(std::move(upload));
        }
    }
}

void PremultiplyAlpha(LoadedImage& image) {
    if (image.width == 0 || image.height == 0 || image.pixels.empty()) {
        return;
    }

    for (std::size_t i = 0; i < image.pixels.size(); i += 4) {
        const std::uint32_t alpha = image.pixels[i + 3];
        image.pixels[i + 0] = static_cast<std::uint8_t>((static_cast<std::uint32_t>(image.pixels[i + 0]) * alpha) / 255);
        image.pixels[i + 1] = static_cast<std::uint8_t>((static_cast<std::uint32_t>(image.pixels[i + 1]) * alpha) / 255);
        image.pixels[i + 2] = static_cast<std::uint8_t>((static_cast<std::uint32_t>(image.pixels[i + 2]) * alpha) / 255);
    }
}

struct ComScope {
    bool shouldUninitialize = false;

    ComScope() {
        const HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        if (SUCCEEDED(hr)) {
            shouldUninitialize = true;
            return;
        }

        if (hr != RPC_E_CHANGED_MODE) {
            throw std::runtime_error("Failed to initialize COM.");
        }
    }

    ~ComScope() {
        if (shouldUninitialize) {
            CoUninitialize();
        }
    }
};

void ThrowIfFailed(HRESULT hr, const std::string& message) {
    if (FAILED(hr)) {
        throw std::runtime_error(message);
    }
}

std::wstring ToWide(const std::string& text) {
    const int requiredLength = MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, nullptr, 0);
    if (requiredLength <= 0) {
        throw std::runtime_error("Failed to convert path to wide string.");
    }

    std::wstring wide(static_cast<std::size_t>(requiredLength), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, text.c_str(), -1, wide.data(), requiredLength);
    wide.pop_back();
    return wide;
}

LoadedImage LoadImageRgba(const std::string& path) {
    ComScope comScope;
    const std::wstring widePath = ToWide(path);

    Microsoft::WRL::ComPtr<IWICImagingFactory> factory;
    const HRESULT hr = CoCreateInstance(
        CLSID_WICImagingFactory,
        nullptr,
        CLSCTX_INPROC_SERVER,
        IID_PPV_ARGS(&factory)
    );
    ThrowIfFailed(hr, "Failed to create WIC factory.");

    Microsoft::WRL::ComPtr<IWICBitmapDecoder> decoder;
    ThrowIfFailed(
        factory->CreateDecoderFromFilename(
            widePath.c_str(),
            nullptr,
            GENERIC_READ,
            WICDecodeMetadataCacheOnLoad,
            &decoder
        ),
        ("Failed to open texture: " + path).c_str()
    );

    Microsoft::WRL::ComPtr<IWICBitmapFrameDecode> frame;
    ThrowIfFailed(decoder->GetFrame(0, &frame), ("Failed to read texture frame: " + path).c_str());

    Microsoft::WRL::ComPtr<IWICFormatConverter> converter;
    ThrowIfFailed(factory->CreateFormatConverter(&converter), "Failed to create WIC converter.");
    ThrowIfFailed(
        converter->Initialize(
            frame.Get(),
            GUID_WICPixelFormat32bppRGBA,
            WICBitmapDitherTypeNone,
            nullptr,
            0.0,
            WICBitmapPaletteTypeCustom
        ),
        ("Failed to convert texture to RGBA: " + path).c_str()
    );

    UINT width = 0;
    UINT height = 0;
    ThrowIfFailed(converter->GetSize(&width, &height), ("Failed to read texture size: " + path).c_str());

    LoadedImage image{};
    image.width = width;
    image.height = height;
    image.pixels.resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4);

    ThrowIfFailed(
        converter->CopyPixels(
            nullptr,
            width * 4,
            static_cast<UINT>(image.pixels.size()),
            image.pixels.data()
        ),
        ("Failed to copy texture pixels: " + path).c_str()
    );

    return image;
}

LoadedImage LoadImageRgbaFromMemory(const std::vector<std::uint8_t>& bytes) {
    ComScope comScope;

    Microsoft::WRL::ComPtr<IWICImagingFactory> factory;
    ThrowIfFailed(
        CoCreateInstance(
            CLSID_WICImagingFactory,
            nullptr,
            CLSCTX_INPROC_SERVER,
            IID_PPV_ARGS(&factory)
        ),
        "Failed to create WIC factory."
    );

    Microsoft::WRL::ComPtr<IWICStream> stream;
    ThrowIfFailed(factory->CreateStream(&stream), "Failed to create WIC stream.");
    ThrowIfFailed(
        stream->InitializeFromMemory(
            const_cast<BYTE*>(bytes.data()),
            static_cast<DWORD>(bytes.size())
        ),
        "Failed to initialize WIC stream from memory."
    );

    Microsoft::WRL::ComPtr<IWICBitmapDecoder> decoder;
    ThrowIfFailed(
        factory->CreateDecoderFromStream(
            stream.Get(),
            nullptr,
            WICDecodeMetadataCacheOnLoad,
            &decoder
        ),
        "Failed to decode image from memory."
    );

    Microsoft::WRL::ComPtr<IWICBitmapFrameDecode> frame;
    ThrowIfFailed(decoder->GetFrame(0, &frame), "Failed to read image frame.");

    Microsoft::WRL::ComPtr<IWICFormatConverter> converter;
    ThrowIfFailed(factory->CreateFormatConverter(&converter), "Failed to create WIC converter.");
    ThrowIfFailed(
        converter->Initialize(
            frame.Get(),
            GUID_WICPixelFormat32bppRGBA,
            WICBitmapDitherTypeNone,
            nullptr,
            0.0,
            WICBitmapPaletteTypeCustom
        ),
        "Failed to convert memory image to RGBA."
    );

    UINT width = 0;
    UINT height = 0;
    ThrowIfFailed(converter->GetSize(&width, &height), "Failed to read memory image size.");

    LoadedImage image{};
    image.width = width;
    image.height = height;
    image.pixels.resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4);

    ThrowIfFailed(
        converter->CopyPixels(
            nullptr,
            width * 4,
            static_cast<UINT>(image.pixels.size()),
            image.pixels.data()
        ),
        "Failed to copy memory image pixels."
    );

    return image;
}

std::wstring GuessFontFaceName(const std::filesystem::path& path) {
    std::wstring faceName = path.stem().wstring();
    for (wchar_t& c : faceName) {
        if (c == L'_' || c == L'-') {
            c = L' ';
        }
    }
    return faceName;
}

}  // namespace

void VulkanVoxelApp::CreateDepthResources() {
    depthFormat_ = FindDepthFormat();

    CreateImage(
        swapChainExtent_.width,
        swapChainExtent_.height,
        depthFormat_,
        VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        depthImage_,
        depthImageMemory_
    );

    depthImageView_ = CreateImageView(depthImage_, depthFormat_, VK_IMAGE_ASPECT_DEPTH_BIT);
    TransitionImageLayout(
        depthImage_,
        depthFormat_,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
    );
}

void VulkanVoxelApp::CreateTextureImage() {
    const std::filesystem::path textureDirectory = std::filesystem::path(ASSET_DIR) / "assets" / "textures";
    std::vector<LoadedImage> layerImages;
    layerImages.reserve(kBlockTextureLayerFiles.size());

    std::uint32_t textureWidth = 0;
    std::uint32_t textureHeight = 0;
    for (const std::string_view textureName : kBlockTextureLayerFiles) {
        const std::filesystem::path texturePath = textureDirectory / std::filesystem::path(std::string(textureName));
        if (!std::filesystem::exists(texturePath)) {
            throw std::runtime_error("Missing block texture: " + texturePath.string());
        }

        LoadedImage image = LoadImageRgba(texturePath.string());
        if (image.width == 0 || image.height == 0 || image.pixels.empty()) {
            throw std::runtime_error("Failed to load block texture: " + texturePath.string());
        }

        if (layerImages.empty()) {
            textureWidth = image.width;
            textureHeight = image.height;
        } else if (image.width != textureWidth || image.height != textureHeight) {
            throw std::runtime_error("All block textures must share the same dimensions.");
        }

        layerImages.push_back(std::move(image));
    }

    const VkDeviceSize layerSize = static_cast<VkDeviceSize>(textureWidth) * static_cast<VkDeviceSize>(textureHeight) * 4u;
    const VkDeviceSize imageSize = layerSize * static_cast<VkDeviceSize>(layerImages.size());
    textureMipLevels_ = 1;

    VkBuffer stagingBuffer = VK_NULL_HANDLE;
    VkDeviceMemory stagingBufferMemory = VK_NULL_HANDLE;
    CreateBuffer(
        imageSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        stagingBuffer,
        stagingBufferMemory
    );

    void* mappedData = nullptr;
    if (vkMapMemory(device_, stagingBufferMemory, 0, imageSize, 0, &mappedData) != VK_SUCCESS) {
        throw std::runtime_error("Failed to map texture staging buffer.");
    }

    auto* mappedBytes = static_cast<std::uint8_t*>(mappedData);
    for (std::size_t layerIndex = 0; layerIndex < layerImages.size(); ++layerIndex) {
        std::memcpy(
            mappedBytes + static_cast<std::size_t>(layerSize) * layerIndex,
            layerImages[layerIndex].pixels.data(),
            layerImages[layerIndex].pixels.size()
        );
    }
    vkUnmapMemory(device_, stagingBufferMemory);

    CreateImage(
        textureWidth,
        textureHeight,
        VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        textureImage_,
        textureImageMemory_,
        textureMipLevels_,
        static_cast<std::uint32_t>(layerImages.size())
    );

    TransitionImageLayout(
        textureImage_,
        VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        textureMipLevels_,
        static_cast<std::uint32_t>(layerImages.size())
    );
    CopyBufferToImage(
        stagingBuffer,
        textureImage_,
        textureWidth,
        textureHeight,
        static_cast<std::uint32_t>(layerImages.size())
    );
    TransitionImageLayout(
        textureImage_,
        VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        textureMipLevels_,
        static_cast<std::uint32_t>(layerImages.size())
    );

    vkDestroyBuffer(device_, stagingBuffer, nullptr);
    vkFreeMemory(device_, stagingBufferMemory, nullptr);
}

void VulkanVoxelApp::CreateTextureImageView() {
    textureImageView_ = CreateImageView(
        textureImage_,
        VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_ASPECT_COLOR_BIT,
        textureMipLevels_,
        VK_IMAGE_VIEW_TYPE_2D_ARRAY,
        static_cast<std::uint32_t>(kBlockTextureLayerFiles.size())
    );
}

void VulkanVoxelApp::CreateTextureSampler() {
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_NEAREST;
    samplerInfo.minFilter = VK_FILTER_NEAREST;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.maxAnisotropy = 1.0f;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 0.0f;

    if (vkCreateSampler(device_, &samplerInfo, nullptr, &textureSampler_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create texture sampler.");
    }
}

void VulkanVoxelApp::LoadPlayerMesh() {
    std::ifstream file(CHARACTER_MESH_PATH, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open extracted player mesh.");
    }

    char magic[4] = {};
    std::uint32_t vertexCount = 0;
    std::uint32_t indexCount = 0;
    file.read(magic, sizeof(magic));
    file.read(reinterpret_cast<char*>(&vertexCount), sizeof(vertexCount));
    file.read(reinterpret_cast<char*>(&indexCount), sizeof(indexCount));

    if (std::memcmp(magic, "PMSH", 4) != 0) {
        throw std::runtime_error("Invalid extracted player mesh format.");
    }

    std::vector<SerializedPlayerVertex> serializedVertices(vertexCount);
    playerBaseVertices_.resize(vertexCount);
    playerRenderVertices_.resize(vertexCount);
    playerIndices_.resize(indexCount);

    file.read(
        reinterpret_cast<char*>(serializedVertices.data()),
        static_cast<std::streamsize>(sizeof(SerializedPlayerVertex) * serializedVertices.size())
    );
    file.read(
        reinterpret_cast<char*>(playerIndices_.data()),
        static_cast<std::streamsize>(sizeof(std::uint32_t) * playerIndices_.size())
    );

    if (!file) {
        throw std::runtime_error("Failed to read extracted player mesh data.");
    }

    for (std::size_t i = 0; i < serializedVertices.size(); ++i) {
        const SerializedPlayerVertex& src = serializedVertices[i];
        WorldVertex& dst = playerBaseVertices_[i];
        dst.position[0] = src.position[0];
        dst.position[1] = src.position[1];
        dst.position[2] = src.position[2];
        dst.uv[0] = src.uv[0];
        dst.uv[1] = src.uv[1];
        dst.ao = 1.0f;
    }

    playerIndexCount_ = indexCount;
    playerLoaded_ = !playerBaseVertices_.empty() && !playerIndices_.empty();
    if (!playerLoaded_) {
        return;
    }

    playerModelBoundsMin_ = {
        playerBaseVertices_[0].position[0],
        playerBaseVertices_[0].position[1],
        playerBaseVertices_[0].position[2],
    };
    playerModelBoundsMax_ = playerModelBoundsMin_;

    for (const WorldVertex& vertex : playerBaseVertices_) {
        playerModelBoundsMin_.x = std::min(playerModelBoundsMin_.x, vertex.position[0]);
        playerModelBoundsMin_.y = std::min(playerModelBoundsMin_.y, vertex.position[1]);
        playerModelBoundsMin_.z = std::min(playerModelBoundsMin_.z, vertex.position[2]);
        playerModelBoundsMax_.x = std::max(playerModelBoundsMax_.x, vertex.position[0]);
        playerModelBoundsMax_.y = std::max(playerModelBoundsMax_.y, vertex.position[1]);
        playerModelBoundsMax_.z = std::max(playerModelBoundsMax_.z, vertex.position[2]);
    }
}

void VulkanVoxelApp::CreatePlayerTextureImage() {
    const std::vector<char> rawTextureBytes = ReadFile(CHARACTER_TEXTURE_PATH);
    std::vector<std::uint8_t> textureBytes(rawTextureBytes.begin(), rawTextureBytes.end());
    const LoadedImage image = LoadImageRgbaFromMemory(textureBytes);
    const VkDeviceSize imageSize = static_cast<VkDeviceSize>(image.pixels.size());

    VkBuffer stagingBuffer = VK_NULL_HANDLE;
    VkDeviceMemory stagingBufferMemory = VK_NULL_HANDLE;
    CreateBuffer(
        imageSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        stagingBuffer,
        stagingBufferMemory
    );

    void* mappedData = nullptr;
    if (vkMapMemory(device_, stagingBufferMemory, 0, imageSize, 0, &mappedData) != VK_SUCCESS) {
        throw std::runtime_error("Failed to map player texture staging buffer.");
    }

    std::memcpy(mappedData, image.pixels.data(), image.pixels.size());
    vkUnmapMemory(device_, stagingBufferMemory);

    CreateImage(
        image.width,
        image.height,
        VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        playerTextureImage_,
        playerTextureImageMemory_
    );

    TransitionImageLayout(
        playerTextureImage_,
        VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
    );
    CopyBufferToImage(stagingBuffer, playerTextureImage_, image.width, image.height);
    TransitionImageLayout(
        playerTextureImage_,
        VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    );

    vkDestroyBuffer(device_, stagingBuffer, nullptr);
    vkFreeMemory(device_, stagingBufferMemory, nullptr);
}

void VulkanVoxelApp::CreatePlayerTextureImageView() {
    playerTextureImageView_ = CreateImageView(playerTextureImage_, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);
}

void VulkanVoxelApp::CreatePlayerTextureSampler() {
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_NEAREST;
    samplerInfo.minFilter = VK_FILTER_NEAREST;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;

    if (vkCreateSampler(device_, &samplerInfo, nullptr, &playerTextureSampler_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create player texture sampler.");
    }
}

void VulkanVoxelApp::LoadOverlayFont() {
    const std::filesystem::path fontPath = std::filesystem::path(ASSET_DIR) / "assets" / "fonts" / "VCR_OSD_MONO.ttf";
    if (!std::filesystem::exists(fontPath)) {
        throw std::runtime_error("Font file not found. Expected assets/fonts/VCR_OSD_MONO.ttf");
    }

    const std::wstring widePath = ToWide(fontPath.string());
    if (AddFontResourceExW(widePath.c_str(), FR_PRIVATE, nullptr) == 0) {
        throw std::runtime_error("Failed to load overlay font file.");
    }

    const std::wstring faceName = GuessFontFaceName(fontPath);
    const int fontHeight = 20;
    const int padding = 4;

    HDC glyphDc = CreateCompatibleDC(nullptr);
    if (glyphDc == nullptr) {
        RemoveFontResourceExW(widePath.c_str(), FR_PRIVATE, nullptr);
        throw std::runtime_error("Failed to create font device context.");
    }

    HFONT font = CreateFontW(
        -fontHeight,
        0,
        0,
        0,
        FW_NORMAL,
        FALSE,
        FALSE,
        FALSE,
        DEFAULT_CHARSET,
        OUT_TT_ONLY_PRECIS,
        CLIP_DEFAULT_PRECIS,
        ANTIALIASED_QUALITY,
        FF_DONTCARE,
        faceName.c_str()
    );
    if (font == nullptr) {
        DeleteDC(glyphDc);
        RemoveFontResourceExW(widePath.c_str(), FR_PRIVATE, nullptr);
        throw std::runtime_error("Failed to create overlay font.");
    }

    const HGDIOBJ oldFont = SelectObject(glyphDc, font);
    SetTextColor(glyphDc, RGB(255, 255, 255));
    SetBkColor(glyphDc, RGB(0, 0, 0));
    SetBkMode(glyphDc, OPAQUE);

    TEXTMETRICW metrics{};
    if (GetTextMetricsW(glyphDc, &metrics) == 0) {
        SelectObject(glyphDc, oldFont);
        DeleteObject(font);
        DeleteDC(glyphDc);
        RemoveFontResourceExW(widePath.c_str(), FR_PRIVATE, nullptr);
        throw std::runtime_error("Failed to read overlay font metrics.");
    }

    overlayFontLineHeight_ = metrics.tmHeight;
    for (FontGlyphBitmap& glyph : overlayFontGlyphs_) {
        glyph = {};
    }

    for (unsigned int code = 32; code < 127; ++code) {
        const wchar_t ch = static_cast<wchar_t>(code);
        SIZE size{};
        if (GetTextExtentPoint32W(glyphDc, &ch, 1, &size) == 0) {
            continue;
        }

        FontGlyphBitmap glyph{};
        glyph.advance = std::max(size.cx, metrics.tmAveCharWidth / 2);
        glyph.valid = true;

        if (ch == L' ') {
            overlayFontGlyphs_[code] = std::move(glyph);
            continue;
        }

        const int bitmapWidth = std::max(static_cast<int>(size.cx) + padding * 2 + 8, 8);
        const int bitmapHeight = std::max(static_cast<int>(metrics.tmHeight) + padding * 2 + 8, 8);

        BITMAPINFO bitmapInfo{};
        bitmapInfo.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
        bitmapInfo.bmiHeader.biWidth = bitmapWidth;
        bitmapInfo.bmiHeader.biHeight = -bitmapHeight;
        bitmapInfo.bmiHeader.biPlanes = 1;
        bitmapInfo.bmiHeader.biBitCount = 32;
        bitmapInfo.bmiHeader.biCompression = BI_RGB;

        void* bits = nullptr;
        HBITMAP bitmap = CreateDIBSection(glyphDc, &bitmapInfo, DIB_RGB_COLORS, &bits, nullptr, 0);
        if (bitmap == nullptr || bits == nullptr) {
            continue;
        }

        const HGDIOBJ oldBitmap = SelectObject(glyphDc, bitmap);
        PatBlt(glyphDc, 0, 0, bitmapWidth, bitmapHeight, BLACKNESS);
        TextOutW(glyphDc, padding, padding, &ch, 1);

        auto* pixels = static_cast<const std::uint8_t*>(bits);
        int minX = bitmapWidth;
        int minY = bitmapHeight;
        int maxX = -1;
        int maxY = -1;

        for (int y = 0; y < bitmapHeight; ++y) {
            for (int x = 0; x < bitmapWidth; ++x) {
                const std::size_t index = static_cast<std::size_t>((y * bitmapWidth + x) * 4);
                const std::uint8_t alpha = std::max({pixels[index + 0], pixels[index + 1], pixels[index + 2]});
                if (alpha == 0) {
                    continue;
                }

                minX = std::min(minX, x);
                minY = std::min(minY, y);
                maxX = std::max(maxX, x);
                maxY = std::max(maxY, y);
            }
        }

        if (maxX >= minX && maxY >= minY) {
            glyph.width = maxX - minX + 1;
            glyph.height = maxY - minY + 1;
            glyph.offsetX = minX - padding;
            glyph.offsetY = minY - padding;
            glyph.alpha.resize(static_cast<std::size_t>(glyph.width * glyph.height));

            for (int y = 0; y < glyph.height; ++y) {
                for (int x = 0; x < glyph.width; ++x) {
                    const int sourceX = minX + x;
                    const int sourceY = minY + y;
                    const std::size_t sourceIndex = static_cast<std::size_t>((sourceY * bitmapWidth + sourceX) * 4);
                    const std::uint8_t alpha = std::max({
                        pixels[sourceIndex + 0],
                        pixels[sourceIndex + 1],
                        pixels[sourceIndex + 2]
                    });
                    glyph.alpha[static_cast<std::size_t>(y * glyph.width + x)] = alpha;
                }
            }
        }

        overlayFontGlyphs_[code] = std::move(glyph);
        SelectObject(glyphDc, oldBitmap);
        DeleteObject(bitmap);
    }

    SelectObject(glyphDc, oldFont);
    DeleteObject(font);
    DeleteDC(glyphDc);
    RemoveFontResourceExW(widePath.c_str(), FR_PRIVATE, nullptr);

    constexpr int atlasWidth = 1024;
    constexpr int atlasHeight = 1024;
    constexpr int atlasPadding = 2;
    const std::filesystem::path crosshairPath = std::filesystem::path(ASSET_DIR) / "assets" / "Crosshair.png";
    const std::filesystem::path sunPath = std::filesystem::path(ASSET_DIR) / "assets" / "Sun.png";
    const std::filesystem::path moonPath = std::filesystem::path(ASSET_DIR) / "assets" / "Moon.png";
    LoadedImage crosshairImage{};
    LoadedImage sunImage{};
    LoadedImage moonImage{};
    if (std::filesystem::exists(crosshairPath)) {
        crosshairImage = LoadImageRgba(crosshairPath.string());
        PremultiplyAlpha(crosshairImage);
    }
    if (std::filesystem::exists(sunPath)) {
        sunImage = LoadImageRgba(sunPath.string());
        PremultiplyAlpha(sunImage);
    }
    if (std::filesystem::exists(moonPath)) {
        moonImage = LoadImageRgba(moonPath.string());
        PremultiplyAlpha(moonImage);
    }

    std::vector<std::uint8_t> atlasPixels(static_cast<std::size_t>(atlasWidth * atlasHeight) * 4, 0);
    int cursorX = atlasPadding;
    int cursorY = atlasPadding;
    int rowHeight = 0;

    for (unsigned int code = 32; code < 127; ++code) {
        FontGlyphBitmap& glyph = overlayFontGlyphs_[code];
        if (!glyph.valid || glyph.width <= 0 || glyph.height <= 0) {
            continue;
        }

        if (cursorX + glyph.width + atlasPadding > atlasWidth) {
            cursorX = atlasPadding;
            cursorY += rowHeight + atlasPadding;
            rowHeight = 0;
        }

        if (cursorY + glyph.height + atlasPadding > atlasHeight) {
            throw std::runtime_error("Overlay font atlas is too small.");
        }

        for (int y = 0; y < glyph.height; ++y) {
            const std::size_t srcOffset = static_cast<std::size_t>(y * glyph.width);
            for (int x = 0; x < glyph.width; ++x) {
                const std::uint8_t alpha = glyph.alpha[srcOffset + static_cast<std::size_t>(x)];
                const std::size_t dstIndex = static_cast<std::size_t>(((cursorY + y) * atlasWidth + cursorX + x) * 4);
                atlasPixels[dstIndex + 0] = alpha;
                atlasPixels[dstIndex + 1] = alpha;
                atlasPixels[dstIndex + 2] = alpha;
                atlasPixels[dstIndex + 3] = alpha;
            }
        }

        glyph.u0 = static_cast<float>(cursorX) / static_cast<float>(atlasWidth);
        glyph.v0 = static_cast<float>(cursorY) / static_cast<float>(atlasHeight);
        glyph.u1 = static_cast<float>(cursorX + glyph.width) / static_cast<float>(atlasWidth);
        glyph.v1 = static_cast<float>(cursorY + glyph.height) / static_cast<float>(atlasHeight);
        glyph.alpha.clear();
        glyph.alpha.shrink_to_fit();

        cursorX += glyph.width + atlasPadding;
        rowHeight = std::max(rowHeight, glyph.height);
    }

    crosshairLoaded_ = false;
    if (crosshairImage.width > 0 && crosshairImage.height > 0) {
        if (cursorX + static_cast<int>(crosshairImage.width) + atlasPadding > atlasWidth) {
            cursorX = atlasPadding;
            cursorY += rowHeight + atlasPadding;
            rowHeight = 0;
        }

        if (cursorY + static_cast<int>(crosshairImage.height) + atlasPadding > atlasHeight) {
            throw std::runtime_error("Overlay atlas is too small for crosshair.");
        }

        for (std::uint32_t y = 0; y < crosshairImage.height; ++y) {
            for (std::uint32_t x = 0; x < crosshairImage.width; ++x) {
                const std::size_t srcIndex = static_cast<std::size_t>((y * crosshairImage.width + x) * 4);
                const std::size_t dstIndex = static_cast<std::size_t>(
                    ((cursorY + static_cast<int>(y)) * atlasWidth + cursorX + static_cast<int>(x)) * 4
                );
                atlasPixels[dstIndex + 0] = crosshairImage.pixels[srcIndex + 0];
                atlasPixels[dstIndex + 1] = crosshairImage.pixels[srcIndex + 1];
                atlasPixels[dstIndex + 2] = crosshairImage.pixels[srcIndex + 2];
                atlasPixels[dstIndex + 3] = crosshairImage.pixels[srcIndex + 3];
            }
        }

        crosshairU0_ = static_cast<float>(cursorX) / static_cast<float>(atlasWidth);
        crosshairV0_ = static_cast<float>(cursorY) / static_cast<float>(atlasHeight);
        crosshairU1_ = static_cast<float>(cursorX + static_cast<int>(crosshairImage.width)) / static_cast<float>(atlasWidth);
        crosshairV1_ = static_cast<float>(cursorY + static_cast<int>(crosshairImage.height)) / static_cast<float>(atlasHeight);
        crosshairWidth_ = static_cast<float>(crosshairImage.width);
        crosshairHeight_ = static_cast<float>(crosshairImage.height);
        crosshairLoaded_ = true;
        cursorX += static_cast<int>(crosshairImage.width) + atlasPadding;
        rowHeight = std::max(rowHeight, static_cast<int>(crosshairImage.height));
    }

    sunLoaded_ = sunImage.width > 0 && sunImage.height > 0;
    sunU0_ = 0.0f;
    sunV0_ = 0.0f;
    sunU1_ = 1.0f;
    sunV1_ = 1.0f;
    sunWidth_ = static_cast<float>(sunImage.width);
    sunHeight_ = static_cast<float>(sunImage.height);

    moonLoaded_ = moonImage.width > 0 && moonImage.height > 0;
    moonU0_ = 0.0f;
    moonV0_ = 0.0f;
    moonU1_ = 1.0f;
    moonV1_ = 1.0f;
    moonWidth_ = static_cast<float>(moonImage.width);
    moonHeight_ = static_cast<float>(moonImage.height);

    const VkDeviceSize atlasSize = static_cast<VkDeviceSize>(atlasPixels.size());

    VkBuffer stagingBuffer = VK_NULL_HANDLE;
    VkDeviceMemory stagingBufferMemory = VK_NULL_HANDLE;
    CreateBuffer(
        atlasSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        stagingBuffer,
        stagingBufferMemory
    );

    void* mappedData = nullptr;
    if (vkMapMemory(device_, stagingBufferMemory, 0, atlasSize, 0, &mappedData) != VK_SUCCESS) {
        throw std::runtime_error("Failed to map overlay font staging buffer.");
    }
    std::memcpy(mappedData, atlasPixels.data(), atlasPixels.size());
    vkUnmapMemory(device_, stagingBufferMemory);

    CreateImage(
        atlasWidth,
        atlasHeight,
        VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        overlayFontImage_,
        overlayFontImageMemory_
    );

    TransitionImageLayout(
        overlayFontImage_,
        VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
    );
    CopyBufferToImage(stagingBuffer, overlayFontImage_, atlasWidth, atlasHeight);
    TransitionImageLayout(
        overlayFontImage_,
        VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    );

    vkDestroyBuffer(device_, stagingBuffer, nullptr);
    vkFreeMemory(device_, stagingBufferMemory, nullptr);

    overlayFontImageView_ = CreateImageView(overlayFontImage_, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);

    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.maxAnisotropy = 1.0f;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 0.0f;

    if (vkCreateSampler(device_, &samplerInfo, nullptr, &overlayFontSampler_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create overlay font sampler.");
    }

    const auto createCelestialTexture = [&](const LoadedImage& image, VkImage& targetImage, VkDeviceMemory& targetMemory, VkImageView& targetView, VkSampler& targetSampler) {
        if (image.width == 0 || image.height == 0 || image.pixels.empty()) {
            return;
        }

        const VkDeviceSize imageSize = static_cast<VkDeviceSize>(image.pixels.size());
        VkBuffer stagingBuffer = VK_NULL_HANDLE;
        VkDeviceMemory stagingBufferMemory = VK_NULL_HANDLE;
        CreateBuffer(
            imageSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            stagingBuffer,
            stagingBufferMemory
        );

        void* mappedTextureData = nullptr;
        if (vkMapMemory(device_, stagingBufferMemory, 0, imageSize, 0, &mappedTextureData) != VK_SUCCESS) {
            throw std::runtime_error("Failed to map celestial staging buffer.");
        }
        std::memcpy(mappedTextureData, image.pixels.data(), image.pixels.size());
        vkUnmapMemory(device_, stagingBufferMemory);

        CreateImage(
            image.width,
            image.height,
            VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            targetImage,
            targetMemory
        );

        TransitionImageLayout(
            targetImage,
            VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
        );
        CopyBufferToImage(stagingBuffer, targetImage, image.width, image.height);
        TransitionImageLayout(
            targetImage,
            VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        );

        vkDestroyBuffer(device_, stagingBuffer, nullptr);
        vkFreeMemory(device_, stagingBufferMemory, nullptr);

        targetView = CreateImageView(targetImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);

        VkSamplerCreateInfo celestialSamplerInfo{};
        celestialSamplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        celestialSamplerInfo.magFilter = VK_FILTER_LINEAR;
        celestialSamplerInfo.minFilter = VK_FILTER_LINEAR;
        celestialSamplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        celestialSamplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        celestialSamplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        celestialSamplerInfo.anisotropyEnable = VK_FALSE;
        celestialSamplerInfo.maxAnisotropy = 1.0f;
        celestialSamplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        celestialSamplerInfo.unnormalizedCoordinates = VK_FALSE;
        celestialSamplerInfo.compareEnable = VK_FALSE;
        celestialSamplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        celestialSamplerInfo.minLod = 0.0f;
        celestialSamplerInfo.maxLod = 0.0f;

        if (vkCreateSampler(device_, &celestialSamplerInfo, nullptr, &targetSampler) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create celestial sampler.");
        }
    };

    createCelestialTexture(sunImage, sunImage_, sunImageMemory_, sunImageView_, sunSampler_);
    createCelestialTexture(moonImage, moonImage_, moonImageMemory_, moonImageView_, moonSampler_);
}

void VulkanVoxelApp::DestroyWorldRenderBatch(WorldRenderBatch& batch) {
    batch.poolIndex = 0;
    batch.quadOffset = 0;
    batch.quadCapacity = 0;
    batch.quadCount = 0;
    batch.vertexCount = 0;
    batch.indexCount = 0;
}

void VulkanVoxelApp::DestroyWorldQuadPool(WorldQuadPool& pool) {
    if (pool.mappedData != nullptr && pool.memory != VK_NULL_HANDLE) {
        vkUnmapMemory(device_, pool.memory);
        pool.mappedData = nullptr;
    }
    if (pool.buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device_, pool.buffer, nullptr);
        pool.buffer = VK_NULL_HANDLE;
    }
    if (pool.memory != VK_NULL_HANDLE) {
        vkFreeMemory(device_, pool.memory, nullptr);
        pool.memory = VK_NULL_HANDLE;
    }
    pool.capacityQuads = 0;
    pool.committedQuads = 0;
    pool.freeRanges.clear();
}

std::uint32_t VulkanVoxelApp::CreateWorldQuadPool(std::uint32_t capacityQuads) {
    capacityQuads = std::max<std::uint32_t>(capacityQuads, 1u);
    WorldQuadPool pool{};
    CreateBuffer(
        sizeof(WorldQuadRecord) * static_cast<VkDeviceSize>(capacityQuads),
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        pool.buffer,
        pool.memory
    );
    if (vkMapMemory(
            device_,
            pool.memory,
            0,
            sizeof(WorldQuadRecord) * static_cast<VkDeviceSize>(capacityQuads),
            0,
            &pool.mappedData) != VK_SUCCESS) {
        vkDestroyBuffer(device_, pool.buffer, nullptr);
        vkFreeMemory(device_, pool.memory, nullptr);
        throw std::runtime_error("Failed to map world quad pool.");
    }
    pool.capacityQuads = capacityQuads;
    pool.committedQuads = 0;
    pool.freeRanges.clear();
    worldQuadPools_.push_back(std::move(pool));
    return static_cast<std::uint32_t>(worldQuadPools_.size() - 1);
}

WorldQuadRange VulkanVoxelApp::AllocateWorldQuadRange(WorldQuadPool& pool, std::uint32_t quadCount) {
    if (quadCount == 0) {
        return {};
    }

    auto bestIt = pool.freeRanges.end();
    for (auto it = pool.freeRanges.begin(); it != pool.freeRanges.end(); ++it) {
        if (it->count >= quadCount &&
            (bestIt == pool.freeRanges.end() || it->count < bestIt->count)) {
            bestIt = it;
        }
    }
    if (bestIt != pool.freeRanges.end()) {
        WorldQuadRange range{bestIt->offset, quadCount};
        bestIt->offset += quadCount;
        bestIt->count -= quadCount;
        if (bestIt->count == 0) {
            pool.freeRanges.erase(bestIt);
        }
        return range;
    }

    if (pool.committedQuads + quadCount > pool.capacityQuads) {
        return {};
    }

    WorldQuadRange range{pool.committedQuads, quadCount};
    pool.committedQuads += quadCount;
    return range;
}

void VulkanVoxelApp::ReleaseWorldQuadRange(WorldQuadPool& pool, std::uint32_t offset, std::uint32_t count) {
    if (count == 0) {
        return;
    }

    pool.freeRanges.push_back({offset, count});
    std::sort(
        pool.freeRanges.begin(),
        pool.freeRanges.end(),
        [](const WorldQuadRange& a, const WorldQuadRange& b) {
            return a.offset < b.offset;
        }
    );

    std::vector<WorldQuadRange> mergedRanges;
    mergedRanges.reserve(pool.freeRanges.size());
    for (const WorldQuadRange& range : pool.freeRanges) {
        if (!mergedRanges.empty() &&
            mergedRanges.back().offset + mergedRanges.back().count == range.offset) {
            mergedRanges.back().count += range.count;
        } else {
            mergedRanges.push_back(range);
        }
    }
    pool.freeRanges = std::move(mergedRanges);
}

void VulkanVoxelApp::DestroyWorldMeshBuffers() {
    for (auto& [id, batch] : worldRenderBatches_) {
        (void)id;
        DestroyWorldRenderBatch(batch);
    }
    worldRenderBatches_.clear();
    for (WorldQuadPool& pool : worldQuadPools_) {
        DestroyWorldQuadPool(pool);
    }
    worldQuadPools_.clear();
    worldVertexCount_ = 0;
    worldIndexCount_ = 0;
}

void VulkanVoxelApp::TryCleanupRetiredWorldRenderBatches() {
    // Shared quad pools stay resident for the app lifetime. No deferred pool retirement path remains.
}

void VulkanVoxelApp::UploadWorldRenderBatch(WorldRenderBatch& batch, const ChunkMeshBatchData& batchData) {
    const auto uploadChunkStartTime = std::chrono::steady_clock::now();
    batch.id = batchData.id;
    const std::uint32_t requiredQuads = static_cast<std::uint32_t>(batchData.quads.size());
    if (requiredQuads > 0 && batch.quadCapacity < requiredQuads) {
        if (batch.quadCapacity > 0 && batch.poolIndex < worldQuadPools_.size()) {
            ReleaseWorldQuadRange(worldQuadPools_[batch.poolIndex], batch.quadOffset, batch.quadCapacity);
        }

        WorldQuadRange range{};
        std::uint32_t selectedPoolIndex = std::numeric_limits<std::uint32_t>::max();
        for (std::uint32_t poolIndex = 0; poolIndex < static_cast<std::uint32_t>(worldQuadPools_.size()); ++poolIndex) {
            range = AllocateWorldQuadRange(worldQuadPools_[poolIndex], requiredQuads);
            if (range.count >= requiredQuads) {
                selectedPoolIndex = poolIndex;
                break;
            }
        }

        if (selectedPoolIndex == std::numeric_limits<std::uint32_t>::max()) {
            const std::uint32_t previousCapacity =
                worldQuadPools_.empty() ? 0u : worldQuadPools_.back().capacityQuads;
            const std::uint32_t newPoolCapacity = std::max<std::uint32_t>(
                std::max<std::uint32_t>(requiredQuads, 1024u),
                previousCapacity > 0 ? previousCapacity * 2u : 1024u
            );
            const auto uploadAllocStartTime = std::chrono::steady_clock::now();
            selectedPoolIndex = CreateWorldQuadPool(newPoolCapacity);
            const auto uploadAllocEndTime = std::chrono::steady_clock::now();
            AccumulateRuntimeProfileSample(
                std::chrono::duration<double, std::milli>(uploadAllocEndTime - uploadAllocStartTime).count(),
                uploadAllocProfileTotalMs_,
                uploadAllocProfileMaxMs_,
                uploadAllocProfileSamples_
            );
            range = AllocateWorldQuadRange(worldQuadPools_[selectedPoolIndex], requiredQuads);
        }

        if (range.count < requiredQuads) {
            throw std::runtime_error("Failed to allocate world quad range.");
        }

        batch.poolIndex = selectedPoolIndex;
        batch.quadOffset = range.offset;
        batch.quadCapacity = range.count;
    } else if (requiredQuads == 0 && batch.quadCapacity > 0 && batch.poolIndex < worldQuadPools_.size()) {
        ReleaseWorldQuadRange(worldQuadPools_[batch.poolIndex], batch.quadOffset, batch.quadCapacity);
        batch.poolIndex = 0;
        batch.quadOffset = 0;
        batch.quadCapacity = 0;
    }

    const auto uploadCopyStartTime = std::chrono::steady_clock::now();
    if (!batchData.quads.empty()) {
        if (batch.poolIndex >= worldQuadPools_.size()) {
            throw std::runtime_error("Invalid world quad pool index.");
        }
        WorldQuadPool& pool = worldQuadPools_[batch.poolIndex];
        std::memcpy(
            static_cast<std::byte*>(pool.mappedData) +
                sizeof(WorldQuadRecord) * static_cast<std::size_t>(batch.quadOffset),
            batchData.quads.data(),
            sizeof(WorldQuadRecord) * batchData.quads.size()
        );
    }
    const auto uploadCopyEndTime = std::chrono::steady_clock::now();
    AccumulateRuntimeProfileSample(
        std::chrono::duration<double, std::milli>(uploadCopyEndTime - uploadCopyStartTime).count(),
        uploadCopyProfileTotalMs_,
        uploadCopyProfileMaxMs_,
        uploadCopyProfileSamples_
    );

    batch.quadCount = static_cast<std::uint32_t>(batchData.quads.size());
    batch.vertexCount = batch.quadCount * 4;
    batch.indexCount = batch.quadCount * 6;
    const auto uploadChunkEndTime = std::chrono::steady_clock::now();
    AccumulateRuntimeProfileSample(
        std::chrono::duration<double, std::milli>(uploadChunkEndTime - uploadChunkStartTime).count(),
        uploadChunkProfileTotalMs_,
        uploadChunkProfileMaxMs_,
        uploadChunkProfileSamples_
    );
}

void VulkanVoxelApp::DestroyPlayerBuffers() {
    for (VkBuffer& buffer : playerVertexBuffers_) {
        if (buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(device_, buffer, nullptr);
            buffer = VK_NULL_HANDLE;
        }
    }
    for (VkDeviceMemory& memory : playerVertexBuffersMemory_) {
        if (memory != VK_NULL_HANDLE) {
            vkFreeMemory(device_, memory, nullptr);
            memory = VK_NULL_HANDLE;
        }
    }
    playerVertexBuffers_.clear();
    playerVertexBuffersMemory_.clear();
    playerVertexBufferCapacities_.clear();
    if (playerIndexBuffer_ != VK_NULL_HANDLE) {
        vkDestroyBuffer(device_, playerIndexBuffer_, nullptr);
        playerIndexBuffer_ = VK_NULL_HANDLE;
    }
    if (playerIndexBufferMemory_ != VK_NULL_HANDLE) {
        vkFreeMemory(device_, playerIndexBufferMemory_, nullptr);
        playerIndexBufferMemory_ = VK_NULL_HANDLE;
    }

    playerVertexCount_ = 0;
    playerIndexCount_ = 0;
    playerIndexBufferCapacity_ = 0;
}

void VulkanVoxelApp::CreatePlayerBuffers() {
    if (!playerLoaded_) {
        return;
    }

    const VkDeviceSize vertexBufferSize =
        sizeof(WorldVertex) * static_cast<VkDeviceSize>(playerBaseVertices_.size());
    const VkDeviceSize indexBufferSize =
        sizeof(std::uint32_t) * static_cast<VkDeviceSize>(playerIndices_.size());

    if (playerVertexBuffers_.empty()) {
        playerVertexBuffers_.resize(kMaxFramesInFlight, VK_NULL_HANDLE);
        playerVertexBuffersMemory_.resize(kMaxFramesInFlight, VK_NULL_HANDLE);
        playerVertexBufferCapacities_.resize(kMaxFramesInFlight, vertexBufferSize);
        for (int i = 0; i < kMaxFramesInFlight; ++i) {
            CreateBuffer(
                playerVertexBufferCapacities_[static_cast<std::size_t>(i)],
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                playerVertexBuffers_[static_cast<std::size_t>(i)],
                playerVertexBuffersMemory_[static_cast<std::size_t>(i)]
            );
        }
    }

    if (playerIndexBuffer_ == VK_NULL_HANDLE) {
        playerIndexBufferCapacity_ = indexBufferSize;
        CreateBuffer(
            playerIndexBufferCapacity_,
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            playerIndexBuffer_,
            playerIndexBufferMemory_
        );

        void* mappedData = nullptr;
        if (vkMapMemory(device_, playerIndexBufferMemory_, 0, indexBufferSize, 0, &mappedData) != VK_SUCCESS) {
            throw std::runtime_error("Failed to map player index buffer.");
        }
        std::memcpy(mappedData, playerIndices_.data(), static_cast<std::size_t>(indexBufferSize));
        vkUnmapMemory(device_, playerIndexBufferMemory_);
    }

    playerIndexCount_ = static_cast<std::uint32_t>(playerIndices_.size());
    UpdatePlayerRenderMesh();
}

void VulkanVoxelApp::UpdatePlayerRenderMesh() {
    if (!playerLoaded_ || currentFrame_ >= playerVertexBuffers_.size() ||
        playerVertexBuffers_[currentFrame_] == VK_NULL_HANDLE) {
        playerVertexCount_ = 0;
        return;
    }

    if (!IsThirdPersonView()) {
        playerVertexCount_ = 0;
        return;
    }

    const float modelHeight = std::max(playerModelBoundsMax_.y - playerModelBoundsMin_.y, 0.0001f);
    const float scale = 1.75f / modelHeight;
    const float centerX = (playerModelBoundsMin_.x + playerModelBoundsMax_.x) * 0.5f;
    const float centerZ = (playerModelBoundsMin_.z + playerModelBoundsMax_.z) * 0.5f;
    const float yawRadians = (cameraYaw_ + 90.0f) * 3.14159265358979323846f / 180.0f;
    const float cosYaw = std::cos(yawRadians);
    const float sinYaw = std::sin(yawRadians);
    const Vec3 playerFeet = GetInterpolatedPlayerFeetPosition();

    playerRenderVertices_.resize(playerBaseVertices_.size());
    for (std::size_t i = 0; i < playerBaseVertices_.size(); ++i) {
        const WorldVertex& src = playerBaseVertices_[i];
        WorldVertex& dst = playerRenderVertices_[i];

        const float localX = (src.position[0] - centerX) * scale;
        const float localY = (src.position[1] - playerModelBoundsMin_.y) * scale;
        const float localZ = (src.position[2] - centerZ) * scale;

        const float rotatedX = localX * cosYaw - localZ * sinYaw;
        const float rotatedZ = localX * sinYaw + localZ * cosYaw;

        dst.position[0] = playerFeet.x + rotatedX;
        dst.position[1] = playerFeet.y + localY;
        dst.position[2] = playerFeet.z + rotatedZ;
        dst.uv[0] = src.uv[0];
        dst.uv[1] = src.uv[1];
        dst.ao = src.ao;
    }

    const VkDeviceSize vertexBufferSize =
        sizeof(WorldVertex) * static_cast<VkDeviceSize>(playerRenderVertices_.size());
    void* mappedData = nullptr;
    if (vkMapMemory(
            device_,
            playerVertexBuffersMemory_[currentFrame_],
            0,
            vertexBufferSize,
            0,
            &mappedData) != VK_SUCCESS) {
        throw std::runtime_error("Failed to map player vertex buffer.");
    }
    std::memcpy(mappedData, playerRenderVertices_.data(), static_cast<std::size_t>(vertexBufferSize));
    vkUnmapMemory(device_, playerVertexBuffersMemory_[currentFrame_]);

    playerVertexCount_ = static_cast<std::uint32_t>(playerRenderVertices_.size());
}

void VulkanVoxelApp::UploadWorldRenderUpdate(const WorldRenderUpdate& update) {
    const auto uploadStartTime = std::chrono::steady_clock::now();
    if (device_ == VK_NULL_HANDLE) {
        return;
    }

    TryCleanupRetiredWorldRenderBatches();
    loadedChunkCount_ = update.loadedChunkCount;
    std::uint32_t uploadedChunkCount = 0;

    const auto uploadRemoveStartTime = std::chrono::steady_clock::now();
    for (const PendingChunkId& batchId : update.removals) {
        auto batchIt = worldRenderBatches_.find(batchId);
        if (batchIt == worldRenderBatches_.end()) {
            continue;
        }

        worldVertexCount_ -= batchIt->second.vertexCount;
        worldIndexCount_ -= batchIt->second.indexCount;
        if (batchIt->second.quadCapacity > 0 && batchIt->second.poolIndex < worldQuadPools_.size()) {
            ReleaseWorldQuadRange(
                worldQuadPools_[batchIt->second.poolIndex],
                batchIt->second.quadOffset,
                batchIt->second.quadCapacity
            );
        }
        worldRenderBatches_.erase(batchIt);
    }
    const auto uploadRemoveEndTime = std::chrono::steady_clock::now();
    AccumulateRuntimeProfileSample(
        std::chrono::duration<double, std::milli>(uploadRemoveEndTime - uploadRemoveStartTime).count(),
        uploadRemoveProfileTotalMs_,
        uploadRemoveProfileMaxMs_,
        uploadRemoveProfileSamples_
    );

    for (const ChunkMeshBatchData& batchData : update.uploads) {
        const auto uploadInsertStartTime = std::chrono::steady_clock::now();
        auto existingIt = worldRenderBatches_.find(batchData.id);
        if (existingIt != worldRenderBatches_.end()) {
            worldVertexCount_ -= existingIt->second.vertexCount;
            worldIndexCount_ -= existingIt->second.indexCount;
            if (existingIt->second.quadCapacity > 0 &&
                existingIt->second.poolIndex < worldQuadPools_.size()) {
                ReleaseWorldQuadRange(
                    worldQuadPools_[existingIt->second.poolIndex],
                    existingIt->second.quadOffset,
                    existingIt->second.quadCapacity
                );
            }
            worldRenderBatches_.erase(existingIt);
        }

        WorldRenderBatch batch{};
        UploadWorldRenderBatch(batch, batchData);
        worldVertexCount_ += batch.vertexCount;
        worldIndexCount_ += batch.indexCount;
        worldRenderBatches_.emplace(batch.id, std::move(batch));
        ++uploadedChunkCount;
        const auto uploadInsertEndTime = std::chrono::steady_clock::now();
        AccumulateRuntimeProfileSample(
            std::chrono::duration<double, std::milli>(uploadInsertEndTime - uploadInsertStartTime).count(),
            uploadInsertProfileTotalMs_,
            uploadInsertProfileMaxMs_,
            uploadInsertProfileSamples_
        );
    }

    const auto uploadEndTime = std::chrono::steady_clock::now();
    AccumulateRuntimeProfileSample(
        std::chrono::duration<double, std::milli>(uploadEndTime - uploadStartTime).count(),
        uploadProfileTotalMs_,
        uploadProfileMaxMs_,
        uploadProfileSamples_
    );
    if (uploadedChunkCount > 0) {
        AccumulateRuntimeProfileSample(
            static_cast<double>(uploadedChunkCount),
            uploadCountProfileTotalValue_,
            uploadCountProfileMaxValue_,
            uploadCountProfileSamples_
        );
    }
}

void VulkanVoxelApp::BuildWorldMesh() {
    const int centerChunkX = static_cast<int>(std::floor(cameraPosition_.x / static_cast<float>(kChunkSizeX)));
    const int centerChunkZ = static_cast<int>(std::floor(cameraPosition_.z / static_cast<float>(kChunkSizeZ)));

    std::vector<PendingChunkId> chunkRequests;
    std::vector<MeshBuildInput> meshRequests;
    WorldRenderUpdate renderUpdate{};
    {
        std::unique_lock lock(worldMutex_);
        world_.UpdateStreamingTargets(centerChunkX, centerChunkZ, worldSettings_.chunkRadius);
        chunkRequests = world_.AcquireChunkLoadRequests(kChunkGenerationBudgetPerPass);
        meshRequests = world_.AcquireDirtyMeshRequests(kMeshBuildBudgetPerPass, centerChunkX, centerChunkZ, worldSettings_.chunkRadius);
    }

    std::vector<PreparedChunkColumn> preparedChunks;
    preparedChunks.reserve(chunkRequests.size());
    for (const PendingChunkId& chunkRequest : chunkRequests) {
        preparedChunks.push_back(world_.PrepareChunkColumn(chunkRequest.chunkX, chunkRequest.chunkZ));
    }

    std::vector<PreparedSubChunkMesh> preparedMeshes;
    preparedMeshes.reserve(meshRequests.size());
    for (const MeshBuildInput& meshRequest : meshRequests) {
        preparedMeshes.push_back(world_.PrepareSubChunkMesh(meshRequest));
    }

    {
        std::unique_lock lock(worldMutex_);
        world_.UpdateStreamingTargets(centerChunkX, centerChunkZ, worldSettings_.chunkRadius);
        for (PreparedChunkColumn& preparedChunk : preparedChunks) {
            world_.CommitPreparedChunkColumn(std::move(preparedChunk));
        }
        for (PreparedSubChunkMesh& preparedMesh : preparedMeshes) {
            world_.CommitPreparedSubChunkMesh(std::move(preparedMesh));
        }
        world_.FinalizeStreamingWindow(
            centerChunkX,
            centerChunkZ,
            worldSettings_.chunkRadius,
            kChunkGenerationBudgetPerPass
        );
        renderUpdate = world_.DrainRenderUpdates();
    }

    UploadWorldRenderUpdate(renderUpdate);
}

void VulkanVoxelApp::RequestWorldMeshBuild() {
    if (!worldMeshWorkerRunning_) {
        return;
    }

    WorldMeshBuildRequest request{};
    request.centerChunkX = static_cast<int>(std::floor(cameraPosition_.x / static_cast<float>(kChunkSizeX)));
    request.centerChunkZ = static_cast<int>(std::floor(cameraPosition_.z / static_cast<float>(kChunkSizeZ)));

    {
        std::lock_guard lock(worldMeshWorkerMutex_);
        request.serial = ++nextWorldMeshRequestSerial_;
        pendingWorldMeshRequest_ = request;
        worldMeshTargetAvailable_ = true;
        worldMeshRequestPending_ = true;
    }

    worldMeshWorkerCv_.notify_all();
}

void VulkanVoxelApp::StartWorldMeshWorker() {
    if (worldMeshWorkerRunning_) {
        return;
    }

    worldMeshWorkerRunning_ = true;
    chunkLoadWorkerThreads_.clear();
    chunkLoadWorkerThreads_.reserve(kChunkLoadWorkerCount);
    meshWorkerThreads_.clear();
    meshWorkerThreads_.reserve(kMeshWorkerCount);

    for (std::size_t workerIndex = 0; workerIndex < kChunkLoadWorkerCount; ++workerIndex) {
        chunkLoadWorkerThreads_.emplace_back([this]() {
            try {
                for (;;) {
                    PendingChunkId chunkRequest{};
                    bool hasChunkRequest = false;
                    {
                        std::unique_lock lock(worldMeshWorkerMutex_);
                        worldMeshWorkerCv_.wait(lock, [this]() {
                            return !worldMeshWorkerRunning_ || !pendingStorageChunkRequests_.empty() || worldMeshTargetAvailable_;
                        });

                        if (!worldMeshWorkerRunning_) {
                            return;
                        }

                        if (!pendingStorageChunkRequests_.empty()) {
                            chunkRequest = pendingStorageChunkRequests_.front();
                            pendingStorageChunkRequests_.pop_front();
                            pendingStorageChunkRequestSet_.erase(chunkRequest);
                            hasChunkRequest = true;
                        }
                    }

                    if (!hasChunkRequest) {
                        std::this_thread::sleep_for(kWorldMeshWorkerYieldDelay);
                        continue;
                    }

                    PreparedChunkColumn preparedChunk = world_.PrepareChunkColumn(chunkRequest.chunkX, chunkRequest.chunkZ);
                    {
                        std::lock_guard lock(worldMeshWorkerMutex_);
                        completedPreparedChunkColumns_.push_back(std::move(preparedChunk));
                        worldMeshTargetAvailable_ = true;
                    }
                    worldMeshWorkerCv_.notify_all();
                }
            } catch (const std::exception& e) {
                OutputDebugStringA(e.what());
                OutputDebugStringA("\n");
            }
        });
    }

    for (std::size_t workerIndex = 0; workerIndex < kMeshWorkerCount; ++workerIndex) {
        meshWorkerThreads_.emplace_back([this]() {
            try {
                for (;;) {
                    WorldMeshBuildRequest request{};
                    {
                        std::unique_lock lock(worldMeshWorkerMutex_);
                        worldMeshWorkerCv_.wait(lock, [this]() {
                            return !worldMeshWorkerRunning_ || worldMeshTargetAvailable_;
                        });

                        if (!worldMeshWorkerRunning_) {
                            return;
                        }

                        request = pendingWorldMeshRequest_;
                    }

                    std::vector<MeshBuildInput> meshRequests;
                    {
                        std::unique_lock lock(worldMutex_);
                        world_.UpdateStreamingTargets(request.centerChunkX, request.centerChunkZ, worldSettings_.chunkRadius);
                        meshRequests = world_.AcquireDirtyMeshRequests(1, request.centerChunkX, request.centerChunkZ, worldSettings_.chunkRadius);
                    }

                    if (meshRequests.empty()) {
                        std::this_thread::sleep_for(kWorldMeshWorkerYieldDelay);
                        continue;
                    }

                    PreparedSubChunkMesh preparedMesh = world_.PrepareSubChunkMesh(meshRequests.front());
                    bool hasPendingRenderUpdates = false;
                    {
                        std::unique_lock lock(worldMutex_);
                        world_.CommitPreparedSubChunkMesh(std::move(preparedMesh));
                        hasPendingRenderUpdates = world_.HasPendingRenderUpdates();
                    }

                    if (hasPendingRenderUpdates) {
                        {
                            std::lock_guard lock(worldMeshWorkerMutex_);
                            worldMeshTargetAvailable_ = true;
                        }
                        worldMeshWorkerCv_.notify_all();
                    }
                }
            } catch (const std::exception& e) {
                OutputDebugStringA(e.what());
                OutputDebugStringA("\n");
            }
        });
    }

    worldSaveWorkerThread_ = std::thread([this]() {
        for (;;) {
            RegionSaveTask saveTask{};
            bool hasSaveTask = false;
            {
                std::unique_lock lock(worldMeshWorkerMutex_);
                worldMeshWorkerCv_.wait(lock, [this]() {
                    return !worldMeshWorkerRunning_ || !pendingWorldSaveTasks_.empty();
                });

                if (!worldMeshWorkerRunning_ && pendingWorldSaveTasks_.empty()) {
                    return;
                }

                if (!pendingWorldSaveTasks_.empty()) {
                    saveTask = std::move(pendingWorldSaveTasks_.front());
                    pendingWorldSaveTasks_.pop_front();
                    hasSaveTask = true;
                }
            }

            if (!hasSaveTask) {
                continue;
            }

            try {
                world_.ExecuteSaveTask(saveTask);
                {
                    std::unique_lock lock(worldMutex_);
                    world_.CompletePendingSaveTask(saveTask);
                }
            } catch (const std::exception& e) {
                {
                    std::lock_guard lock(worldMeshWorkerMutex_);
                    pendingWorldSaveTasks_.push_back(std::move(saveTask));
                }
                OutputDebugStringA(e.what());
                OutputDebugStringA("\n");
                std::this_thread::sleep_for(kWorldMeshWorkerYieldDelay);
            }
        }
    });

    worldMeshWorkerThread_ = std::thread([this]() {
        try {
        for (;;) {
            WorldMeshBuildRequest request{};
            {
                std::unique_lock lock(worldMeshWorkerMutex_);
                worldMeshWorkerCv_.wait(lock, [this]() {
                    return !worldMeshWorkerRunning_ || worldMeshTargetAvailable_;
                });

                if (!worldMeshWorkerRunning_) {
                    return;
                }

                request = pendingWorldMeshRequest_;
            }

            std::size_t remainingStreamingWork = 0;
            std::optional<WorldRenderUpdate> renderUpdate;
            std::vector<PendingChunkId> chunkRequests;
            std::vector<PreparedChunkColumn> preparedChunks;
            std::vector<RegionSaveTask> saveTasks;
            bool hasPendingPreparedChunks = false;
            {
                std::lock_guard lock(worldMeshWorkerMutex_);
                preparedChunks.reserve(std::min<std::size_t>(completedPreparedChunkColumns_.size(), kChunkGenerationBudgetPerPass));
                while (!completedPreparedChunkColumns_.empty() &&
                       preparedChunks.size() < kChunkGenerationBudgetPerPass) {
                    preparedChunks.push_back(std::move(completedPreparedChunkColumns_.front()));
                    completedPreparedChunkColumns_.pop_front();
                }
            }
            {
                std::unique_lock lock(worldMutex_);
                world_.UpdateStreamingTargets(request.centerChunkX, request.centerChunkZ, worldSettings_.chunkRadius);
                chunkRequests = world_.AcquireChunkLoadRequests(kChunkGenerationBudgetPerPass);
                for (PreparedChunkColumn& preparedChunk : preparedChunks) {
                    world_.CommitPreparedChunkColumn(std::move(preparedChunk));
                }
                saveTasks = world_.DrainPendingSaveTasks(std::numeric_limits<std::size_t>::max());
                remainingStreamingWork = world_.FinalizeStreamingWindow(
                    request.centerChunkX,
                    request.centerChunkZ,
                    worldSettings_.chunkRadius,
                    kChunkGenerationBudgetPerPass
                );
                if (world_.HasPendingRenderUpdates()) {
                    renderUpdate = world_.DrainRenderUpdates();
                }
            }

            bool hasNewTarget = false;
            {
                std::lock_guard lock(worldMeshWorkerMutex_);
                for (const PendingChunkId& chunkRequest : chunkRequests) {
                    if (pendingStorageChunkRequestSet_.insert(chunkRequest).second) {
                        pendingStorageChunkRequests_.push_back(chunkRequest);
                    }
                }
                for (RegionSaveTask& saveTask : saveTasks) {
                    pendingWorldSaveTasks_.push_back(std::move(saveTask));
                }
                if (renderUpdate.has_value()) {
                    if (completedWorldRenderUpdate_.has_value()) {
                        MergeWorldRenderUpdate(*completedWorldRenderUpdate_, std::move(*renderUpdate));
                    } else {
                        completedWorldRenderUpdate_ = std::move(renderUpdate);
                    }
                    completedWorldMeshSerial_ = ++nextCompletedWorldMeshSerial_;
                }

                hasNewTarget = worldMeshRequestPending_;
                worldMeshRequestPending_ = false;
                hasPendingPreparedChunks = !completedPreparedChunkColumns_.empty();
            }

            if (!chunkRequests.empty() || !saveTasks.empty()) {
                worldMeshWorkerCv_.notify_all();
            }

            if (hasNewTarget || hasPendingPreparedChunks || remainingStreamingWork > 0 || renderUpdate.has_value() || !saveTasks.empty()) {
                std::this_thread::sleep_for(kWorldMeshWorkerYieldDelay);
                continue;
            }

            {
                std::lock_guard lock(worldMeshWorkerMutex_);
                if (!worldMeshRequestPending_ && completedPreparedChunkColumns_.empty()) {
                    worldMeshTargetAvailable_ = false;
                }
            }
        }
        } catch (const std::exception& e) {
            OutputDebugStringA(e.what());
            OutputDebugStringA("\n");
        }
    });
}

void VulkanVoxelApp::StopWorldMeshWorker() {
    {
        std::lock_guard lock(worldMeshWorkerMutex_);
        worldMeshWorkerRunning_ = false;
        worldMeshTargetAvailable_ = false;
        worldMeshRequestPending_ = false;
        pendingStorageChunkRequests_.clear();
        pendingStorageChunkRequestSet_.clear();
        completedPreparedChunkColumns_.clear();
    }

    worldMeshWorkerCv_.notify_all();

    for (std::thread& workerThread : chunkLoadWorkerThreads_) {
        if (workerThread.joinable()) {
            workerThread.join();
        }
    }
    chunkLoadWorkerThreads_.clear();

    for (std::thread& workerThread : meshWorkerThreads_) {
        if (workerThread.joinable()) {
            workerThread.join();
        }
    }
    meshWorkerThreads_.clear();

    if (worldMeshWorkerThread_.joinable()) {
        worldMeshWorkerThread_.join();
    }

    if (worldSaveWorkerThread_.joinable()) {
        worldSaveWorkerThread_.join();
    }
}

void VulkanVoxelApp::ConsumeCompletedWorldMesh() {
    std::optional<WorldRenderUpdate> renderUpdate;
    std::uint64_t serial = 0;

    {
        std::lock_guard lock(worldMeshWorkerMutex_);
        if (completedWorldRenderUpdate_.has_value() && completedWorldMeshSerial_ > uploadedWorldMeshSerial_) {
            serial = completedWorldMeshSerial_;
            renderUpdate = std::move(completedWorldRenderUpdate_);
            completedWorldRenderUpdate_.reset();
        }
    }

    if (renderUpdate.has_value()) {
        pendingWorldRenderLoadedChunkCount_ = renderUpdate->loadedChunkCount;

        for (const PendingChunkId& removal : renderUpdate->removals) {
            ErasePendingChunkUpload(pendingWorldRenderUploads_, pendingWorldRenderUploadSet_, removal);
            EnqueuePendingChunkId(pendingWorldRenderRemovals_, pendingWorldRenderRemovalSet_, removal);
        }

        for (ChunkMeshBatchData& upload : renderUpdate->uploads) {
            ErasePendingChunkId(pendingWorldRenderRemovals_, pendingWorldRenderRemovalSet_, upload.id);
            EnqueuePendingChunkUpload(pendingWorldRenderUploads_, pendingWorldRenderUploadSet_, std::move(upload));
        }

        uploadedWorldMeshSerial_ = serial;
    }

    if (pendingWorldRenderRemovals_.empty() && pendingWorldRenderUploads_.empty()) {
        return;
    }

    WorldRenderUpdate budgetedUpdate{};
    budgetedUpdate.loadedChunkCount = pendingWorldRenderLoadedChunkCount_;

    const std::size_t removalCount =
        std::min<std::size_t>(pendingWorldRenderRemovals_.size(), kWorldRenderRemovalBudgetPerConsume);
    for (std::size_t i = 0; i < removalCount; ++i) {
        const PendingChunkId batchId = pendingWorldRenderRemovals_[i];
        budgetedUpdate.removals.push_back(batchId);
        pendingWorldRenderRemovalSet_.erase(batchId);
    }
    pendingWorldRenderRemovals_.erase(
        pendingWorldRenderRemovals_.begin(),
        pendingWorldRenderRemovals_.begin() + static_cast<std::ptrdiff_t>(removalCount)
    );

    const std::size_t uploadCount =
        std::min<std::size_t>(pendingWorldRenderUploads_.size(), kWorldRenderUploadBudgetPerConsume);
    for (std::size_t i = 0; i < uploadCount; ++i) {
        pendingWorldRenderUploadSet_.erase(pendingWorldRenderUploads_[i].id);
        budgetedUpdate.uploads.push_back(std::move(pendingWorldRenderUploads_[i]));
    }
    pendingWorldRenderUploads_.erase(
        pendingWorldRenderUploads_.begin(),
        pendingWorldRenderUploads_.begin() + static_cast<std::ptrdiff_t>(uploadCount)
    );

    if (budgetedUpdate.removals.empty() && budgetedUpdate.uploads.empty()) {
        return;
    }

    UploadWorldRenderUpdate(budgetedUpdate);
}

void VulkanVoxelApp::CreateOverlayBuffer() {
    CreateBuffer(
        sizeof(OverlayVertex) * kMaxOverlayVertexCount,
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        overlayVertexBuffer_,
        overlayVertexBufferMemory_
    );

    celestialVertexBuffers_.resize(kMaxFramesInFlight, VK_NULL_HANDLE);
    celestialVertexBuffersMemory_.resize(kMaxFramesInFlight, VK_NULL_HANDLE);
    for (int i = 0; i < kMaxFramesInFlight; ++i) {
        CreateBuffer(
            sizeof(OverlayVertex) * kMaxCelestialVertexCount,
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            celestialVertexBuffers_[i],
            celestialVertexBuffersMemory_[i]
        );
    }

    RebuildOverlayVertices();
    UploadOverlayVertices();
}

void VulkanVoxelApp::CreateSelectionBuffer() {
    CreateBuffer(
        sizeof(SelectionVertex) * 24,
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        selectionVertexBuffer_,
        selectionVertexBufferMemory_
    );

    UpdateSelectionBuffer();
}

void VulkanVoxelApp::CreateEntityColliderBuffer() {
    CreateBuffer(
        sizeof(SelectionVertex) * 24,
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        entityColliderVertexBuffer_,
        entityColliderVertexBufferMemory_
    );

    UpdateEntityColliderBuffer();
}

void VulkanVoxelApp::CreateUniformBuffers() {
    uniformBuffers_.resize(kMaxFramesInFlight);
    uniformBuffersMemory_.resize(kMaxFramesInFlight);

    for (int i = 0; i < kMaxFramesInFlight; ++i) {
        CreateBuffer(
            sizeof(UniformBufferObject),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            uniformBuffers_[i],
            uniformBuffersMemory_[i]
        );
    }
}

void VulkanVoxelApp::CreateDescriptorPool() {
    const std::array<VkDescriptorPoolSize, 2> poolSizes = {{
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kMaxFramesInFlight * 5},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kMaxFramesInFlight * 5},
    }};

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<std::uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = kMaxFramesInFlight * 5;

    if (vkCreateDescriptorPool(device_, &poolInfo, nullptr, &descriptorPool_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor pool.");
    }
}

void VulkanVoxelApp::CreateDescriptorSets() {
    std::vector<VkDescriptorSetLayout> layouts(kMaxFramesInFlight, descriptorSetLayout_);

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool_;
    allocInfo.descriptorSetCount = kMaxFramesInFlight;
    allocInfo.pSetLayouts = layouts.data();

    descriptorSets_.resize(kMaxFramesInFlight);
    if (vkAllocateDescriptorSets(device_, &allocInfo, descriptorSets_.data()) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate descriptor sets.");
    }

    playerDescriptorSets_.resize(kMaxFramesInFlight);
    if (vkAllocateDescriptorSets(device_, &allocInfo, playerDescriptorSets_.data()) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate player descriptor sets.");
    }

    overlayDescriptorSets_.resize(kMaxFramesInFlight);
    if (vkAllocateDescriptorSets(device_, &allocInfo, overlayDescriptorSets_.data()) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate overlay descriptor sets.");
    }

    sunDescriptorSets_.resize(kMaxFramesInFlight);
    if (vkAllocateDescriptorSets(device_, &allocInfo, sunDescriptorSets_.data()) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate sun descriptor sets.");
    }

    moonDescriptorSets_.resize(kMaxFramesInFlight);
    if (vkAllocateDescriptorSets(device_, &allocInfo, moonDescriptorSets_.data()) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate moon descriptor sets.");
    }

    for (std::size_t i = 0; i < descriptorSets_.size(); ++i) {
        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = uniformBuffers_[i];
        bufferInfo.offset = 0;
        bufferInfo.range = sizeof(UniformBufferObject);

        VkDescriptorImageInfo imageInfo{};
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageInfo.imageView = textureImageView_;
        imageInfo.sampler = textureSampler_;

        std::array<VkWriteDescriptorSet, 2> descriptorWrites{};

        descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[0].dstSet = descriptorSets_[i];
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].pBufferInfo = &bufferInfo;

        descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[1].dstSet = descriptorSets_[i];
        descriptorWrites[1].dstBinding = 1;
        descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorWrites[1].descriptorCount = 1;
        descriptorWrites[1].pImageInfo = &imageInfo;

        vkUpdateDescriptorSets(
            device_,
            static_cast<std::uint32_t>(descriptorWrites.size()),
            descriptorWrites.data(),
            0,
            nullptr
        );

        VkDescriptorImageInfo playerImageInfo{};
        playerImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        playerImageInfo.imageView = playerTextureImageView_;
        playerImageInfo.sampler = playerTextureSampler_;

        std::array<VkWriteDescriptorSet, 2> playerWrites{};
        playerWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        playerWrites[0].dstSet = playerDescriptorSets_[i];
        playerWrites[0].dstBinding = 0;
        playerWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        playerWrites[0].descriptorCount = 1;
        playerWrites[0].pBufferInfo = &bufferInfo;

        playerWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        playerWrites[1].dstSet = playerDescriptorSets_[i];
        playerWrites[1].dstBinding = 1;
        playerWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        playerWrites[1].descriptorCount = 1;
        playerWrites[1].pImageInfo = &playerImageInfo;

        vkUpdateDescriptorSets(
            device_,
            static_cast<std::uint32_t>(playerWrites.size()),
            playerWrites.data(),
            0,
            nullptr
        );

        VkDescriptorImageInfo overlayImageInfo{};
        overlayImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        overlayImageInfo.imageView = overlayFontImageView_;
        overlayImageInfo.sampler = overlayFontSampler_;

        std::array<VkWriteDescriptorSet, 2> overlayWrites{};
        overlayWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        overlayWrites[0].dstSet = overlayDescriptorSets_[i];
        overlayWrites[0].dstBinding = 0;
        overlayWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        overlayWrites[0].descriptorCount = 1;
        overlayWrites[0].pBufferInfo = &bufferInfo;

        overlayWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        overlayWrites[1].dstSet = overlayDescriptorSets_[i];
        overlayWrites[1].dstBinding = 1;
        overlayWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        overlayWrites[1].descriptorCount = 1;
        overlayWrites[1].pImageInfo = &overlayImageInfo;

        vkUpdateDescriptorSets(
            device_,
            static_cast<std::uint32_t>(overlayWrites.size()),
            overlayWrites.data(),
            0,
            nullptr
        );

        if (sunImageView_ != VK_NULL_HANDLE && sunSampler_ != VK_NULL_HANDLE) {
            VkDescriptorImageInfo sunImageInfo{};
            sunImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            sunImageInfo.imageView = sunImageView_;
            sunImageInfo.sampler = sunSampler_;

            std::array<VkWriteDescriptorSet, 2> sunWrites{};
            sunWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            sunWrites[0].dstSet = sunDescriptorSets_[i];
            sunWrites[0].dstBinding = 0;
            sunWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            sunWrites[0].descriptorCount = 1;
            sunWrites[0].pBufferInfo = &bufferInfo;
            sunWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            sunWrites[1].dstSet = sunDescriptorSets_[i];
            sunWrites[1].dstBinding = 1;
            sunWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            sunWrites[1].descriptorCount = 1;
            sunWrites[1].pImageInfo = &sunImageInfo;

            vkUpdateDescriptorSets(
                device_,
                static_cast<std::uint32_t>(sunWrites.size()),
                sunWrites.data(),
                0,
                nullptr
            );
        }

        if (moonImageView_ != VK_NULL_HANDLE && moonSampler_ != VK_NULL_HANDLE) {
            VkDescriptorImageInfo moonImageInfo{};
            moonImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            moonImageInfo.imageView = moonImageView_;
            moonImageInfo.sampler = moonSampler_;

            std::array<VkWriteDescriptorSet, 2> moonWrites{};
            moonWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            moonWrites[0].dstSet = moonDescriptorSets_[i];
            moonWrites[0].dstBinding = 0;
            moonWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            moonWrites[0].descriptorCount = 1;
            moonWrites[0].pBufferInfo = &bufferInfo;
            moonWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            moonWrites[1].dstSet = moonDescriptorSets_[i];
            moonWrites[1].dstBinding = 1;
            moonWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            moonWrites[1].descriptorCount = 1;
            moonWrites[1].pImageInfo = &moonImageInfo;

            vkUpdateDescriptorSets(
                device_,
                static_cast<std::uint32_t>(moonWrites.size()),
                moonWrites.data(),
                0,
                nullptr
            );
        }
    }
}

std::vector<char> VulkanVoxelApp::ReadFile(const char* path) const {
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error(std::string("Failed to open file: ") + path);
    }

    const std::size_t fileSize = static_cast<std::size_t>(file.tellg());
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));

    return buffer;
}

std::uint32_t VulkanVoxelApp::FindMemoryType(std::uint32_t typeFilter, VkMemoryPropertyFlags properties) const {
    VkPhysicalDeviceMemoryProperties memoryProperties{};
    vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &memoryProperties);

    for (std::uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
        const bool typeMatch = (typeFilter & (1u << i)) != 0;
        const bool propertyMatch = (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties;
        if (typeMatch && propertyMatch) {
            return i;
        }
    }

    throw std::runtime_error("Failed to find a suitable memory type.");
}

void VulkanVoxelApp::CreateBuffer(
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    VkMemoryPropertyFlags properties,
    VkBuffer& buffer,
    VkDeviceMemory& bufferMemory
) const {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device_, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create buffer.");
    }

    VkMemoryRequirements memoryRequirements{};
    vkGetBufferMemoryRequirements(device_, buffer, &memoryRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memoryRequirements.size;
    allocInfo.memoryTypeIndex = FindMemoryType(memoryRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device_, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate buffer memory.");
    }

    vkBindBufferMemory(device_, buffer, bufferMemory, 0);
}

void VulkanVoxelApp::CreateImage(
    std::uint32_t width,
    std::uint32_t height,
    VkFormat format,
    VkImageTiling tiling,
    VkImageUsageFlags usage,
    VkMemoryPropertyFlags properties,
    VkImage& image,
    VkDeviceMemory& imageMemory,
    std::uint32_t mipLevels,
    std::uint32_t arrayLayers
) const {
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = mipLevels;
    imageInfo.arrayLayers = arrayLayers;
    imageInfo.format = format;
    imageInfo.tiling = tiling;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateImage(device_, &imageInfo, nullptr, &image) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create image.");
    }

    VkMemoryRequirements memoryRequirements{};
    vkGetImageMemoryRequirements(device_, image, &memoryRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memoryRequirements.size;
    allocInfo.memoryTypeIndex = FindMemoryType(memoryRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device_, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate image memory.");
    }

    vkBindImageMemory(device_, image, imageMemory, 0);
}

VkImageView VulkanVoxelApp::CreateImageView(
    VkImage image,
    VkFormat format,
    VkImageAspectFlags aspectFlags,
    std::uint32_t mipLevels,
    VkImageViewType viewType,
    std::uint32_t layerCount
) const {
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;
    viewInfo.viewType = viewType;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = aspectFlags;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = mipLevels;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = layerCount;

    VkImageView imageView = VK_NULL_HANDLE;
    if (vkCreateImageView(device_, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create image view.");
    }

    return imageView;
}

VkFormat VulkanVoxelApp::FindSupportedFormat(
    const std::vector<VkFormat>& candidates,
    VkImageTiling tiling,
    VkFormatFeatureFlags features
) const {
    for (VkFormat format : candidates) {
        VkFormatProperties properties{};
        vkGetPhysicalDeviceFormatProperties(physicalDevice_, format, &properties);

        if (tiling == VK_IMAGE_TILING_LINEAR &&
            (properties.linearTilingFeatures & features) == features) {
            return format;
        }

        if (tiling == VK_IMAGE_TILING_OPTIMAL &&
            (properties.optimalTilingFeatures & features) == features) {
            return format;
        }
    }

    throw std::runtime_error("Failed to find a supported format.");
}

VkFormat VulkanVoxelApp::FindDepthFormat() const {
    return FindSupportedFormat(
        {
            VK_FORMAT_D32_SFLOAT,
            VK_FORMAT_D32_SFLOAT_S8_UINT,
            VK_FORMAT_D24_UNORM_S8_UINT,
        },
        VK_IMAGE_TILING_OPTIMAL,
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
    );
}

bool VulkanVoxelApp::HasStencilComponent(VkFormat format) const {
    return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
}

std::optional<std::uint64_t> VulkanVoxelApp::QueryVideoMemoryUsageBytes() const {
    Microsoft::WRL::ComPtr<IDXGIFactory4> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) {
        return std::nullopt;
    }

    const std::wstring targetName = ToWide(gpuName_);

    for (UINT adapterIndex = 0;; ++adapterIndex) {
        Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
        if (factory->EnumAdapters1(adapterIndex, &adapter) == DXGI_ERROR_NOT_FOUND) {
            break;
        }

        DXGI_ADAPTER_DESC1 adapterDesc{};
        if (FAILED(adapter->GetDesc1(&adapterDesc))) {
            continue;
        }

        if (targetName != adapterDesc.Description) {
            continue;
        }

        Microsoft::WRL::ComPtr<IDXGIAdapter3> adapter3;
        if (FAILED(adapter.As(&adapter3))) {
            return std::nullopt;
        }

        DXGI_QUERY_VIDEO_MEMORY_INFO memoryInfo{};
        if (FAILED(adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &memoryInfo))) {
            return std::nullopt;
        }

        return static_cast<std::uint64_t>(memoryInfo.CurrentUsage);
    }

    return std::nullopt;
}
