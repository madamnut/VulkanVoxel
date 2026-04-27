#include "render/VulkanSwapchain.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace
{
VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats)
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

VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes)
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

VkExtent2D chooseSwapExtent(
    const VkSurfaceCapabilitiesKHR& capabilities,
    std::uint32_t framebufferWidth,
    std::uint32_t framebufferHeight)
{
    if (capabilities.currentExtent.width != std::numeric_limits<std::uint32_t>::max())
    {
        return capabilities.currentExtent;
    }

    VkExtent2D actualExtent = {framebufferWidth, framebufferHeight};
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
}

SwapchainSupportDetails querySwapchainSupport(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface)
{
    SwapchainSupportDetails details{};
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &details.capabilities);

    std::uint32_t formatCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, nullptr);
    if (formatCount != 0)
    {
        details.formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, details.formats.data());
    }

    std::uint32_t presentModeCount = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, nullptr);
    if (presentModeCount != 0)
    {
        details.presentModes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(
            physicalDevice,
            surface,
            &presentModeCount,
            details.presentModes.data());
    }

    return details;
}

VulkanSwapchainResources createVulkanSwapchain(const VulkanSwapchainCreateInfo& createInfo)
{
    const SwapchainSupportDetails support = querySwapchainSupport(createInfo.physicalDevice, createInfo.surface);
    const VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(support.formats);
    const VkPresentModeKHR presentMode = chooseSwapPresentMode(support.presentModes);
    const VkExtent2D extent = chooseSwapExtent(
        support.capabilities,
        createInfo.framebufferWidth,
        createInfo.framebufferHeight);

    std::uint32_t imageCount = support.capabilities.minImageCount + 1;
    if (support.capabilities.maxImageCount > 0 && imageCount > support.capabilities.maxImageCount)
    {
        imageCount = support.capabilities.maxImageCount;
    }

    const std::array<std::uint32_t, 2> queueFamilyIndices = {
        createInfo.graphicsFamily,
        createInfo.presentFamily,
    };

    VkSwapchainCreateInfoKHR swapchainCreateInfo{};
    swapchainCreateInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    swapchainCreateInfo.surface = createInfo.surface;
    swapchainCreateInfo.minImageCount = imageCount;
    swapchainCreateInfo.imageFormat = surfaceFormat.format;
    swapchainCreateInfo.imageColorSpace = surfaceFormat.colorSpace;
    swapchainCreateInfo.imageExtent = extent;
    swapchainCreateInfo.imageArrayLayers = 1;
    swapchainCreateInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    if (createInfo.graphicsFamily != createInfo.presentFamily)
    {
        swapchainCreateInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        swapchainCreateInfo.queueFamilyIndexCount = static_cast<std::uint32_t>(queueFamilyIndices.size());
        swapchainCreateInfo.pQueueFamilyIndices = queueFamilyIndices.data();
    }
    else
    {
        swapchainCreateInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    swapchainCreateInfo.preTransform = support.capabilities.currentTransform;
    swapchainCreateInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    swapchainCreateInfo.presentMode = presentMode;
    swapchainCreateInfo.clipped = VK_TRUE;

    VulkanSwapchainResources resources{};
    if (vkCreateSwapchainKHR(
            createInfo.device,
            &swapchainCreateInfo,
            nullptr,
            &resources.swapchain) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create Vulkan swapchain.");
    }

    vkGetSwapchainImagesKHR(createInfo.device, resources.swapchain, &imageCount, nullptr);
    resources.images.resize(imageCount);
    vkGetSwapchainImagesKHR(createInfo.device, resources.swapchain, &imageCount, resources.images.data());

    resources.imageFormat = surfaceFormat.format;
    resources.extent = extent;
    resources.imageViews.resize(resources.images.size());
    for (std::size_t i = 0; i < resources.images.size(); ++i)
    {
        resources.imageViews[i] = createVulkanImageView(
            createInfo.device,
            resources.images[i],
            resources.imageFormat);
    }

    return resources;
}

VkImageView createVulkanImageView(
    VkDevice device,
    VkImage image,
    VkFormat format,
    VkImageAspectFlags aspectMask,
    VkImageViewType viewType,
    std::uint32_t layerCount)
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
    if (vkCreateImageView(device, &createInfo, nullptr, &imageView) != VK_SUCCESS)
    {
        throw std::runtime_error("Failed to create Vulkan image view.");
    }

    return imageView;
}

void destroyVulkanSwapchain(VkDevice device, VulkanSwapchainResources& swapchainResources)
{
    for (VkImageView imageView : swapchainResources.imageViews)
    {
        vkDestroyImageView(device, imageView, nullptr);
    }
    swapchainResources.imageViews.clear();
    swapchainResources.images.clear();

    if (swapchainResources.swapchain != VK_NULL_HANDLE)
    {
        vkDestroySwapchainKHR(device, swapchainResources.swapchain, nullptr);
    }
    swapchainResources = {};
}
