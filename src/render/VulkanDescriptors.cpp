#include "render/VulkanDescriptors.h"

#include <stdexcept>

VkDescriptorSetLayoutBinding descriptorSetLayoutBinding(
    std::uint32_t binding,
    VkDescriptorType descriptorType,
    VkShaderStageFlags stageFlags,
    std::uint32_t descriptorCount)
{
    VkDescriptorSetLayoutBinding layoutBinding{};
    layoutBinding.binding = binding;
    layoutBinding.descriptorType = descriptorType;
    layoutBinding.descriptorCount = descriptorCount;
    layoutBinding.stageFlags = stageFlags;
    return layoutBinding;
}

VkDescriptorSetLayout createVulkanDescriptorSetLayout(
    VkDevice device,
    std::span<const VkDescriptorSetLayoutBinding> bindings,
    const char* errorMessage)
{
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<std::uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS)
    {
        throw std::runtime_error(errorMessage);
    }
    return descriptorSetLayout;
}

VkDescriptorPool createVulkanDescriptorPool(
    VkDevice device,
    std::span<const VkDescriptorPoolSize> poolSizes,
    std::uint32_t maxSets,
    const char* errorMessage)
{
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<std::uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = maxSets;

    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS)
    {
        throw std::runtime_error(errorMessage);
    }
    return descriptorPool;
}

VkDescriptorSet allocateVulkanDescriptorSet(
    VkDevice device,
    VkDescriptorPool descriptorPool,
    VkDescriptorSetLayout descriptorSetLayout,
    const char* errorMessage)
{
    VkDescriptorSetAllocateInfo allocateInfo{};
    allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocateInfo.descriptorPool = descriptorPool;
    allocateInfo.descriptorSetCount = 1;
    allocateInfo.pSetLayouts = &descriptorSetLayout;

    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    if (vkAllocateDescriptorSets(device, &allocateInfo, &descriptorSet) != VK_SUCCESS)
    {
        throw std::runtime_error(errorMessage);
    }
    return descriptorSet;
}

VkDescriptorBufferInfo descriptorBufferInfo(
    VkBuffer buffer,
    VkDeviceSize range,
    VkDeviceSize offset)
{
    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = buffer;
    bufferInfo.offset = offset;
    bufferInfo.range = range;
    return bufferInfo;
}

VkDescriptorImageInfo descriptorImageInfo(
    VkImageView imageView,
    VkSampler sampler,
    VkImageLayout imageLayout)
{
    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageLayout = imageLayout;
    imageInfo.imageView = imageView;
    imageInfo.sampler = sampler;
    return imageInfo;
}

VkWriteDescriptorSet writeBufferDescriptor(
    VkDescriptorSet descriptorSet,
    std::uint32_t binding,
    VkDescriptorType descriptorType,
    const VkDescriptorBufferInfo& bufferInfo)
{
    VkWriteDescriptorSet descriptorWrite{};
    descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrite.dstSet = descriptorSet;
    descriptorWrite.dstBinding = binding;
    descriptorWrite.descriptorCount = 1;
    descriptorWrite.descriptorType = descriptorType;
    descriptorWrite.pBufferInfo = &bufferInfo;
    return descriptorWrite;
}

VkWriteDescriptorSet writeImageDescriptor(
    VkDescriptorSet descriptorSet,
    std::uint32_t binding,
    VkDescriptorType descriptorType,
    const VkDescriptorImageInfo& imageInfo)
{
    VkWriteDescriptorSet descriptorWrite{};
    descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrite.dstSet = descriptorSet;
    descriptorWrite.dstBinding = binding;
    descriptorWrite.descriptorCount = 1;
    descriptorWrite.descriptorType = descriptorType;
    descriptorWrite.pImageInfo = &imageInfo;
    return descriptorWrite;
}

void updateVulkanDescriptorSets(
    VkDevice device,
    std::span<const VkWriteDescriptorSet> descriptorWrites)
{
    vkUpdateDescriptorSets(
        device,
        static_cast<std::uint32_t>(descriptorWrites.size()),
        descriptorWrites.data(),
        0,
        nullptr);
}
