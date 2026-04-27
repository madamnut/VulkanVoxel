#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <span>

VkDescriptorSetLayoutBinding descriptorSetLayoutBinding(
    std::uint32_t binding,
    VkDescriptorType descriptorType,
    VkShaderStageFlags stageFlags,
    std::uint32_t descriptorCount = 1);

VkDescriptorSetLayout createVulkanDescriptorSetLayout(
    VkDevice device,
    std::span<const VkDescriptorSetLayoutBinding> bindings,
    const char* errorMessage);

VkDescriptorPool createVulkanDescriptorPool(
    VkDevice device,
    std::span<const VkDescriptorPoolSize> poolSizes,
    std::uint32_t maxSets,
    const char* errorMessage);

VkDescriptorSet allocateVulkanDescriptorSet(
    VkDevice device,
    VkDescriptorPool descriptorPool,
    VkDescriptorSetLayout descriptorSetLayout,
    const char* errorMessage);

VkDescriptorBufferInfo descriptorBufferInfo(
    VkBuffer buffer,
    VkDeviceSize range,
    VkDeviceSize offset = 0);

VkDescriptorImageInfo descriptorImageInfo(
    VkImageView imageView,
    VkSampler sampler,
    VkImageLayout imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

VkWriteDescriptorSet writeBufferDescriptor(
    VkDescriptorSet descriptorSet,
    std::uint32_t binding,
    VkDescriptorType descriptorType,
    const VkDescriptorBufferInfo& bufferInfo);

VkWriteDescriptorSet writeImageDescriptor(
    VkDescriptorSet descriptorSet,
    std::uint32_t binding,
    VkDescriptorType descriptorType,
    const VkDescriptorImageInfo& imageInfo);

void updateVulkanDescriptorSets(
    VkDevice device,
    std::span<const VkWriteDescriptorSet> descriptorWrites);
