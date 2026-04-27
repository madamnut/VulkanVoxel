#include "render/TextureManager.h"

#include <cstddef>
#include <cstring>
#include <stdexcept>

void TextureManager::setContext(VkDevice device, VulkanResourceContext& resourceContext)
{
    device_ = device;
    resourceContext_ = &resourceContext;
}

TextureUpload TextureManager::createTextureFromFile(const std::wstring& path, VkFormat format) const
{
    ImagePixels pixels = loadPngWithWic(path);
    return createTextureFromPixels(pixels.rgba, pixels.width, pixels.height, format);
}

TextureUpload TextureManager::createTextureArrayFromFiles(const std::vector<std::wstring>& paths, VkFormat format) const
{
    std::vector<ImagePixels> layers;
    layers.reserve(paths.size());
    for (const std::wstring& path : paths)
    {
        layers.push_back(loadPngWithWic(path));
    }
    return createTextureArrayFromPixels(layers, format);
}

TextureUpload TextureManager::createTextureFromPixels(
    const std::vector<std::uint8_t>& pixels,
    std::uint32_t width,
    std::uint32_t height,
    VkFormat format) const
{
    const VkDeviceSize imageSize = static_cast<VkDeviceSize>(pixels.size());
    TextureUpload upload{};
    upload.stagingBuffer = resourceContext_->createBuffer(
        imageSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    void* data = nullptr;
    vkMapMemory(device_, upload.stagingBuffer.memory, 0, imageSize, 0, &data);
    std::memcpy(data, pixels.data(), static_cast<std::size_t>(imageSize));
    vkUnmapMemory(device_, upload.stagingBuffer.memory);

    upload.texture.format = format;
    upload.texture.width = width;
    upload.texture.height = height;
    resourceContext_->createImage(
        width,
        height,
        format,
        VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        upload.texture.image,
        upload.texture.memory);

    upload.commandBuffers.push_back(resourceContext_->transitionImageLayout(
        upload.texture.image,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL));
    upload.commandBuffers.push_back(resourceContext_->copyBufferToImage(
        upload.stagingBuffer.buffer,
        upload.texture.image,
        width,
        height));
    upload.commandBuffers.push_back(resourceContext_->transitionImageLayout(
        upload.texture.image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));

    upload.texture.view = createImageView(upload.texture.image, format);
    upload.texture.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    return upload;
}

TextureUpload TextureManager::createTextureArrayFromPixels(const std::vector<ImagePixels>& layers, VkFormat format) const
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
    TextureUpload upload{};
    upload.stagingBuffer = resourceContext_->createBuffer(
        imageSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    void* data = nullptr;
    vkMapMemory(device_, upload.stagingBuffer.memory, 0, imageSize, 0, &data);
    std::memcpy(data, pixels.data(), static_cast<std::size_t>(imageSize));
    vkUnmapMemory(device_, upload.stagingBuffer.memory);

    upload.texture.format = format;
    upload.texture.width = width;
    upload.texture.height = height;
    const std::uint32_t layerCount = static_cast<std::uint32_t>(layers.size());
    resourceContext_->createImage(
        width,
        height,
        format,
        VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        upload.texture.image,
        upload.texture.memory,
        layerCount);

    upload.commandBuffers.push_back(resourceContext_->transitionImageLayout(
        upload.texture.image,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        layerCount));
    upload.commandBuffers.push_back(resourceContext_->copyBufferToImageArray(
        upload.stagingBuffer.buffer,
        upload.texture.image,
        width,
        height,
        layerCount));
    upload.commandBuffers.push_back(resourceContext_->transitionImageLayout(
        upload.texture.image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        layerCount));

    upload.texture.view = createImageView(
        upload.texture.image,
        format,
        VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_VIEW_TYPE_2D_ARRAY,
        layerCount);
    upload.texture.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    return upload;
}

TextureUpload TextureManager::uploadTexturePixels(Texture& texture, const std::vector<std::uint8_t>& pixels) const
{
    const VkDeviceSize imageSize = static_cast<VkDeviceSize>(pixels.size());
    TextureUpload upload{};
    upload.texture = texture;
    upload.stagingBuffer = resourceContext_->createBuffer(
        imageSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    void* data = nullptr;
    vkMapMemory(device_, upload.stagingBuffer.memory, 0, imageSize, 0, &data);
    std::memcpy(data, pixels.data(), static_cast<std::size_t>(imageSize));
    vkUnmapMemory(device_, upload.stagingBuffer.memory);

    upload.commandBuffers.push_back(resourceContext_->transitionImageLayout(
        texture.image,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL));
    upload.commandBuffers.push_back(resourceContext_->copyBufferToImage(
        upload.stagingBuffer.buffer,
        texture.image,
        texture.width,
        texture.height));
    upload.commandBuffers.push_back(resourceContext_->transitionImageLayout(
        texture.image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));

    texture.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    upload.texture = texture;
    return upload;
}

VkImageView TextureManager::createImageView(
    VkImage image,
    VkFormat format,
    VkImageAspectFlags aspectFlags,
    VkImageViewType viewType,
    std::uint32_t layerCount) const
{
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;
    viewInfo.viewType = viewType;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = aspectFlags;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = layerCount;

    VkImageView imageView = VK_NULL_HANDLE;
    if (vkCreateImageView(device_, &viewInfo, nullptr, &imageView) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create Vulkan image view.");
    }

    return imageView;
}
