#pragma once

#include "render/VulkanHelpers.h"

#include <cstdint>

class VulkanResourceContext
{
public:
    void setDevice(VkPhysicalDevice physicalDevice, VkDevice device);
    void setTransferContext(VkCommandPool commandPool, VkQueue graphicsQueue);

    Buffer createBuffer(
        VkDeviceSize size,
        VkBufferUsageFlags usage,
        VkMemoryPropertyFlags properties) const;

    void createImage(
        std::uint32_t width,
        std::uint32_t height,
        VkFormat format,
        VkImageTiling tiling,
        VkImageUsageFlags usage,
        VkMemoryPropertyFlags properties,
        VkImage& image,
        VkDeviceMemory& imageMemory,
        std::uint32_t arrayLayers = 1) const;

    VkCommandBuffer copyBuffer(VkBuffer sourceBuffer, VkBuffer destinationBuffer, VkDeviceSize size) const;

    VkCommandBuffer transitionImageLayout(
        VkImage image,
        VkImageLayout oldLayout,
        VkImageLayout newLayout,
        std::uint32_t layerCount = 1) const;

    VkCommandBuffer copyBufferToImage(
        VkBuffer buffer,
        VkImage image,
        std::uint32_t width,
        std::uint32_t height) const;

    VkCommandBuffer copyBufferToImageArray(
        VkBuffer buffer,
        VkImage image,
        std::uint32_t width,
        std::uint32_t height,
        std::uint32_t layerCount) const;

    void destroyBuffer(Buffer& buffer) const;
    void destroyTexture(Texture& texture) const;

private:
    VkCommandBuffer beginSingleTimeCommands() const;
    VkCommandBuffer submitSingleTimeCommands(VkCommandBuffer commandBuffer) const;

    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkCommandPool commandPool_ = VK_NULL_HANDLE;
    VkQueue graphicsQueue_ = VK_NULL_HANDLE;
};
