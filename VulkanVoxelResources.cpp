#include "VulkanVoxel.h"

#define NOMINMAX
#include <Windows.h>
#include <dxgi1_4.h>
#include <wingdi.h>
#include <wincodec.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

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
constexpr float kWorldVerticalFovDegrees = 70.0f;

struct LoadedImage {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<std::uint8_t> pixels;
};

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
    const std::filesystem::path primaryPath = std::filesystem::path(ASSET_DIR) / "assets" / "textures" / "Block.png";
    const std::filesystem::path fallbackPath = std::filesystem::path(ASSET_DIR) / "Block.png";

    std::filesystem::path texturePath = primaryPath;
    if (!std::filesystem::exists(texturePath) && std::filesystem::exists(fallbackPath)) {
        texturePath = fallbackPath;
    }
    if (!std::filesystem::exists(texturePath)) {
        throw std::runtime_error("Texture file not found. Expected assets/textures/Block.png");
    }

    const std::string path = texturePath.string();
    const LoadedImage image = LoadImageRgba(path);
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
        throw std::runtime_error("Failed to map texture staging buffer.");
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
        textureImage_,
        textureImageMemory_
    );

    TransitionImageLayout(
        textureImage_,
        VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
    );
    CopyBufferToImage(stagingBuffer, textureImage_, image.width, image.height);
    TransitionImageLayout(
        textureImage_,
        VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    );

    vkDestroyBuffer(device_, stagingBuffer, nullptr);
    vkFreeMemory(device_, stagingBufferMemory, nullptr);
}

void VulkanVoxelApp::CreateTextureImageView() {
    textureImageView_ = CreateImageView(textureImage_, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);
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

    playerBaseVertices_.resize(vertexCount);
    playerRenderVertices_.resize(vertexCount);
    playerIndices_.resize(indexCount);

    file.read(
        reinterpret_cast<char*>(playerBaseVertices_.data()),
        static_cast<std::streamsize>(sizeof(WorldVertex) * playerBaseVertices_.size())
    );
    file.read(
        reinterpret_cast<char*>(playerIndices_.data()),
        static_cast<std::streamsize>(sizeof(std::uint32_t) * playerIndices_.size())
    );

    if (!file) {
        throw std::runtime_error("Failed to read extracted player mesh data.");
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
    LoadedImage crosshairImage{};
    if (std::filesystem::exists(crosshairPath)) {
        crosshairImage = LoadImageRgba(crosshairPath.string());
    }

    std::vector<std::uint8_t> atlasPixels(static_cast<std::size_t>(atlasWidth * atlasHeight), 0);
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
            const std::size_t dstOffset = static_cast<std::size_t>((cursorY + y) * atlasWidth + cursorX);
            std::memcpy(
                atlasPixels.data() + dstOffset,
                glyph.alpha.data() + srcOffset,
                static_cast<std::size_t>(glyph.width)
            );
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
                const std::uint8_t alpha = std::max({
                    crosshairImage.pixels[srcIndex + 3],
                    crosshairImage.pixels[srcIndex + 0],
                    crosshairImage.pixels[srcIndex + 1],
                    crosshairImage.pixels[srcIndex + 2]
                });
                atlasPixels[static_cast<std::size_t>((cursorY + static_cast<int>(y)) * atlasWidth + cursorX + static_cast<int>(x))] = alpha;
            }
        }

        crosshairU0_ = static_cast<float>(cursorX) / static_cast<float>(atlasWidth);
        crosshairV0_ = static_cast<float>(cursorY) / static_cast<float>(atlasHeight);
        crosshairU1_ = static_cast<float>(cursorX + static_cast<int>(crosshairImage.width)) / static_cast<float>(atlasWidth);
        crosshairV1_ = static_cast<float>(cursorY + static_cast<int>(crosshairImage.height)) / static_cast<float>(atlasHeight);
        crosshairWidth_ = static_cast<float>(crosshairImage.width);
        crosshairHeight_ = static_cast<float>(crosshairImage.height);
        crosshairLoaded_ = true;
    }

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
        VK_FORMAT_R8_UNORM,
        VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        overlayFontImage_,
        overlayFontImageMemory_
    );

    TransitionImageLayout(
        overlayFontImage_,
        VK_FORMAT_R8_UNORM,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
    );
    CopyBufferToImage(stagingBuffer, overlayFontImage_, atlasWidth, atlasHeight);
    TransitionImageLayout(
        overlayFontImage_,
        VK_FORMAT_R8_UNORM,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    );

    vkDestroyBuffer(device_, stagingBuffer, nullptr);
    vkFreeMemory(device_, stagingBufferMemory, nullptr);

    overlayFontImageView_ = CreateImageView(overlayFontImage_, VK_FORMAT_R8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);

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
}

void VulkanVoxelApp::DestroyWorldMeshBuffers() {
    if (worldIndexBuffer_ != VK_NULL_HANDLE) {
        vkDestroyBuffer(device_, worldIndexBuffer_, nullptr);
        worldIndexBuffer_ = VK_NULL_HANDLE;
    }
    if (worldIndexBufferMemory_ != VK_NULL_HANDLE) {
        vkFreeMemory(device_, worldIndexBufferMemory_, nullptr);
        worldIndexBufferMemory_ = VK_NULL_HANDLE;
    }
    if (worldVertexBuffer_ != VK_NULL_HANDLE) {
        vkDestroyBuffer(device_, worldVertexBuffer_, nullptr);
        worldVertexBuffer_ = VK_NULL_HANDLE;
    }
    if (worldVertexBufferMemory_ != VK_NULL_HANDLE) {
        vkFreeMemory(device_, worldVertexBufferMemory_, nullptr);
        worldVertexBufferMemory_ = VK_NULL_HANDLE;
    }

    worldVertexCount_ = 0;
    worldIndexCount_ = 0;
    worldVertexBufferCapacity_ = 0;
    worldIndexBufferCapacity_ = 0;
}

void VulkanVoxelApp::DestroyPlayerBuffers() {
    if (playerVertexBuffer_ != VK_NULL_HANDLE) {
        vkDestroyBuffer(device_, playerVertexBuffer_, nullptr);
        playerVertexBuffer_ = VK_NULL_HANDLE;
    }
    if (playerVertexBufferMemory_ != VK_NULL_HANDLE) {
        vkFreeMemory(device_, playerVertexBufferMemory_, nullptr);
        playerVertexBufferMemory_ = VK_NULL_HANDLE;
    }
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
    playerVertexBufferCapacity_ = 0;
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

    if (playerVertexBuffer_ == VK_NULL_HANDLE) {
        playerVertexBufferCapacity_ = vertexBufferSize;
        CreateBuffer(
            playerVertexBufferCapacity_,
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            playerVertexBuffer_,
            playerVertexBufferMemory_
        );
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
    if (!playerLoaded_ || playerVertexBuffer_ == VK_NULL_HANDLE) {
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
    }

    const VkDeviceSize vertexBufferSize =
        sizeof(WorldVertex) * static_cast<VkDeviceSize>(playerRenderVertices_.size());
    void* mappedData = nullptr;
    if (vkMapMemory(device_, playerVertexBufferMemory_, 0, vertexBufferSize, 0, &mappedData) != VK_SUCCESS) {
        throw std::runtime_error("Failed to map player vertex buffer.");
    }
    std::memcpy(mappedData, playerRenderVertices_.data(), static_cast<std::size_t>(vertexBufferSize));
    vkUnmapMemory(device_, playerVertexBufferMemory_);

    playerVertexCount_ = static_cast<std::uint32_t>(playerRenderVertices_.size());
}

WorldMeshData VulkanVoxelApp::BuildWorldMeshData(
    int centerChunkX,
    int centerChunkZ,
    const Vec3& cameraPosition,
    const Vec3& cameraForward,
    float aspectRatio
) {
    WorldMeshData mesh;
    {
        std::shared_lock lock(worldMutex_);
        world_.BuildVisibleMesh(
            centerChunkX,
            centerChunkZ,
            std::max(kWorldChunkCountX, kWorldChunkCountZ),
            cameraPosition,
            cameraForward,
            kWorldVerticalFovDegrees,
            aspectRatio,
            mesh
        );
    }
    return mesh;
}

void VulkanVoxelApp::UploadWorldMesh(const WorldMeshData& mesh) {
    if (device_ == VK_NULL_HANDLE) {
        return;
    }

    if (!inFlightFences_.empty()) {
        vkWaitForFences(
            device_,
            static_cast<std::uint32_t>(inFlightFences_.size()),
            inFlightFences_.data(),
            VK_TRUE,
            UINT64_MAX
        );
    }

    loadedChunkCount_ = mesh.loadedChunkCount;

    if (mesh.vertices.empty() || mesh.indices.empty()) {
        worldVertexCount_ = 0;
        worldIndexCount_ = 0;
        return;
    }

    const VkDeviceSize vertexBufferSize = sizeof(WorldVertex) * static_cast<VkDeviceSize>(mesh.vertices.size());
    const VkDeviceSize indexBufferSize = sizeof(std::uint32_t) * static_cast<VkDeviceSize>(mesh.indices.size());

    void* mappedData = nullptr;

    if (worldVertexBuffer_ == VK_NULL_HANDLE || worldVertexBufferCapacity_ < vertexBufferSize) {
        if (worldVertexBuffer_ != VK_NULL_HANDLE) {
            vkDestroyBuffer(device_, worldVertexBuffer_, nullptr);
            worldVertexBuffer_ = VK_NULL_HANDLE;
        }
        if (worldVertexBufferMemory_ != VK_NULL_HANDLE) {
            vkFreeMemory(device_, worldVertexBufferMemory_, nullptr);
            worldVertexBufferMemory_ = VK_NULL_HANDLE;
        }

        worldVertexBufferCapacity_ = std::max(vertexBufferSize, worldVertexBufferCapacity_ * 2);
        CreateBuffer(
            worldVertexBufferCapacity_,
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            worldVertexBuffer_,
            worldVertexBufferMemory_
        );
    }

    if (vkMapMemory(device_, worldVertexBufferMemory_, 0, vertexBufferSize, 0, &mappedData) != VK_SUCCESS) {
        throw std::runtime_error("Failed to map world vertex buffer.");
    }
    std::memcpy(mappedData, mesh.vertices.data(), static_cast<std::size_t>(vertexBufferSize));
    vkUnmapMemory(device_, worldVertexBufferMemory_);

    if (worldIndexBuffer_ == VK_NULL_HANDLE || worldIndexBufferCapacity_ < indexBufferSize) {
        if (worldIndexBuffer_ != VK_NULL_HANDLE) {
            vkDestroyBuffer(device_, worldIndexBuffer_, nullptr);
            worldIndexBuffer_ = VK_NULL_HANDLE;
        }
        if (worldIndexBufferMemory_ != VK_NULL_HANDLE) {
            vkFreeMemory(device_, worldIndexBufferMemory_, nullptr);
            worldIndexBufferMemory_ = VK_NULL_HANDLE;
        }

        worldIndexBufferCapacity_ = std::max(indexBufferSize, worldIndexBufferCapacity_ * 2);
        CreateBuffer(
            worldIndexBufferCapacity_,
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            worldIndexBuffer_,
            worldIndexBufferMemory_
        );
    }

    if (vkMapMemory(device_, worldIndexBufferMemory_, 0, indexBufferSize, 0, &mappedData) != VK_SUCCESS) {
        throw std::runtime_error("Failed to map world index buffer.");
    }
    std::memcpy(mappedData, mesh.indices.data(), static_cast<std::size_t>(indexBufferSize));
    vkUnmapMemory(device_, worldIndexBufferMemory_);

    worldVertexCount_ = static_cast<std::uint32_t>(mesh.vertices.size());
    worldIndexCount_ = static_cast<std::uint32_t>(mesh.indices.size());
}

void VulkanVoxelApp::BuildWorldMesh() {
    const int centerChunkX = std::clamp(static_cast<int>(cameraPosition_.x) / kChunkSizeX, 0, kWorldChunkCountX - 1);
    const int centerChunkZ = std::clamp(static_cast<int>(cameraPosition_.z) / kChunkSizeZ, 0, kWorldChunkCountZ - 1);
    const float aspectRatio = swapChainExtent_.height > 0
        ? static_cast<float>(swapChainExtent_.width) / static_cast<float>(swapChainExtent_.height)
        : (16.0f / 9.0f);
    const WorldMeshData mesh = BuildWorldMeshData(
        centerChunkX,
        centerChunkZ,
        cameraPosition_,
        GetForwardVector(),
        aspectRatio
    );
    UploadWorldMesh(mesh);
}

void VulkanVoxelApp::RequestWorldMeshBuild() {
    if (!worldMeshWorkerRunning_) {
        return;
    }

    WorldMeshBuildRequest request{};
    request.centerChunkX = std::clamp(static_cast<int>(cameraPosition_.x) / kChunkSizeX, 0, kWorldChunkCountX - 1);
    request.centerChunkZ = std::clamp(static_cast<int>(cameraPosition_.z) / kChunkSizeZ, 0, kWorldChunkCountZ - 1);
    request.cameraPosition = cameraPosition_;
    request.cameraForward = GetForwardVector();
    request.aspectRatio = swapChainExtent_.height > 0
        ? static_cast<float>(swapChainExtent_.width) / static_cast<float>(swapChainExtent_.height)
        : (16.0f / 9.0f);

    {
        std::lock_guard lock(worldMeshWorkerMutex_);
        request.serial = ++nextWorldMeshRequestSerial_;
        pendingWorldMeshRequest_ = request;
        worldMeshRequestPending_ = true;
    }

    worldMeshWorkerCv_.notify_one();
}

void VulkanVoxelApp::StartWorldMeshWorker() {
    if (worldMeshWorkerRunning_) {
        return;
    }

    worldMeshWorkerRunning_ = true;
    worldMeshWorkerThread_ = std::thread([this]() {
        for (;;) {
            WorldMeshBuildRequest request{};

            {
                std::unique_lock lock(worldMeshWorkerMutex_);
                worldMeshWorkerCv_.wait(lock, [this]() {
                    return !worldMeshWorkerRunning_ || worldMeshRequestPending_;
                });

                if (!worldMeshWorkerRunning_) {
                    return;
                }

                request = pendingWorldMeshRequest_;
                worldMeshRequestPending_ = false;
            }

            WorldMeshData mesh = BuildWorldMeshData(
                request.centerChunkX,
                request.centerChunkZ,
                request.cameraPosition,
                request.cameraForward,
                request.aspectRatio
            );

            {
                std::lock_guard lock(worldMeshWorkerMutex_);
                if (request.serial < nextWorldMeshRequestSerial_) {
                    continue;
                }

                completedWorldMesh_ = std::move(mesh);
                completedWorldMeshSerial_ = request.serial;
            }
        }
    });
}

void VulkanVoxelApp::StopWorldMeshWorker() {
    {
        std::lock_guard lock(worldMeshWorkerMutex_);
        worldMeshWorkerRunning_ = false;
        worldMeshRequestPending_ = false;
    }

    worldMeshWorkerCv_.notify_all();

    if (worldMeshWorkerThread_.joinable()) {
        worldMeshWorkerThread_.join();
    }
}

void VulkanVoxelApp::ConsumeCompletedWorldMesh() {
    std::optional<WorldMeshData> mesh;
    std::uint64_t serial = 0;

    {
        std::lock_guard lock(worldMeshWorkerMutex_);
        if (!completedWorldMesh_.has_value() || completedWorldMeshSerial_ <= uploadedWorldMeshSerial_) {
            return;
        }

        serial = completedWorldMeshSerial_;
        mesh = std::move(completedWorldMesh_);
        completedWorldMesh_.reset();
    }

    UploadWorldMesh(*mesh);
    uploadedWorldMeshSerial_ = serial;
}

void VulkanVoxelApp::CreateOverlayBuffer() {
    CreateBuffer(
        sizeof(OverlayVertex) * kMaxOverlayVertexCount,
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        overlayVertexBuffer_,
        overlayVertexBufferMemory_
    );

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
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kMaxFramesInFlight * 3},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kMaxFramesInFlight * 3},
    }};

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<std::uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = kMaxFramesInFlight * 3;

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
    VkDeviceMemory& imageMemory
) const {
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
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

VkImageView VulkanVoxelApp::CreateImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags) const {
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = aspectFlags;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

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
