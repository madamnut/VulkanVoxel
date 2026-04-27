#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>
#include <vector>

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

std::vector<char> readBinaryFile(const std::string& path);

std::uint32_t findMemoryType(
    VkPhysicalDevice physicalDevice,
    std::uint32_t typeFilter,
    VkMemoryPropertyFlags properties);

VkShaderModule createShaderModule(VkDevice device, const std::vector<char>& code);
