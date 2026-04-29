#pragma once

#include "assets/ImageLoader.h"
#include "render/VulkanResourceContext.h"

#include <cstdint>
#include <vector>
#include <string>

struct TextureUpload
{
    Texture texture{};
    Buffer stagingBuffer{};
    std::vector<VkCommandBuffer> commandBuffers;
};

class TextureManager
{
public:
    void setContext(VkDevice device, VulkanResourceContext& resourceContext);

    TextureUpload createTextureFromFile(const std::wstring& path, VkFormat format) const;
    TextureUpload createTextureArrayFromFiles(const std::vector<std::wstring>& paths, VkFormat format) const;
    TextureUpload createTextureFromPixels(
        const std::vector<std::uint8_t>& pixels,
        std::uint32_t width,
        std::uint32_t height,
        VkFormat format) const;
    TextureUpload createTextureArrayFromPixels(const std::vector<ImagePixels>& layers, VkFormat format) const;
    TextureUpload uploadTexturePixels(Texture& texture, const std::vector<std::uint8_t>& pixels) const;

private:
    VkImageView createImageView(
        VkImage image,
        VkFormat format,
        VkImageAspectFlags aspectFlags = VK_IMAGE_ASPECT_COLOR_BIT,
        VkImageViewType viewType = VK_IMAGE_VIEW_TYPE_2D,
        std::uint32_t layerCount = 1,
        std::uint32_t mipLevels = 1) const;

    VkDevice device_ = VK_NULL_HANDLE;
    VulkanResourceContext* resourceContext_ = nullptr;
};
