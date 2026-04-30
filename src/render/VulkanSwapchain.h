#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

struct SwapchainSupportDetails
{
    VkSurfaceCapabilitiesKHR capabilities{};
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

struct VulkanSwapchainCreateInfo
{
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    std::uint32_t graphicsFamily = 0;
    std::uint32_t presentFamily = 0;
    std::uint32_t framebufferWidth = 0;
    std::uint32_t framebufferHeight = 0;
};

struct VulkanSwapchainResources
{
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    VkFormat imageFormat = VK_FORMAT_UNDEFINED;
    VkExtent2D extent{};
    bool supportsTransferSrc = false;
    std::vector<VkImage> images;
    std::vector<VkImageView> imageViews;
};

SwapchainSupportDetails querySwapchainSupport(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface);

VulkanSwapchainResources createVulkanSwapchain(const VulkanSwapchainCreateInfo& createInfo);

VkImageView createVulkanImageView(
    VkDevice device,
    VkImage image,
    VkFormat format,
    VkImageAspectFlags aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
    VkImageViewType viewType = VK_IMAGE_VIEW_TYPE_2D,
    std::uint32_t layerCount = 1);

void destroyVulkanSwapchain(VkDevice device, VulkanSwapchainResources& swapchainResources);
