#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>

struct GraphicsPipelineConfig
{
    std::string vertexShaderPath;
    std::string fragmentShaderPath;
    VkExtent2D extent{};
    VkRenderPass renderPass = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkCullModeFlags cullMode = VK_CULL_MODE_NONE;
    VkFrontFace frontFace = VK_FRONT_FACE_CLOCKWISE;
    bool depthTestEnable = false;
    bool depthWriteEnable = false;
    bool alphaBlendEnable = true;
    const VkVertexInputBindingDescription* vertexBindingDescription = nullptr;
    const VkVertexInputAttributeDescription* vertexAttributeDescriptions = nullptr;
    std::uint32_t vertexAttributeDescriptionCount = 0;
    const char* layoutErrorMessage = "Failed to create Vulkan pipeline layout.";
    const char* pipelineErrorMessage = "Failed to create Vulkan graphics pipeline.";
};

struct GraphicsPipelineBundle
{
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
};

GraphicsPipelineBundle createGraphicsPipelineBundle(
    VkDevice device,
    const GraphicsPipelineConfig& config);
