#include "render/VulkanPipelineBuilder.h"

#include "render/VulkanHelpers.h"

#include <array>
#include <stdexcept>
#include <vector>

namespace
{
VkPipelineShaderStageCreateInfo makeShaderStage(
    VkShaderStageFlagBits stage,
    VkShaderModule module)
{
    VkPipelineShaderStageCreateInfo shaderStageInfo{};
    shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageInfo.stage = stage;
    shaderStageInfo.module = module;
    shaderStageInfo.pName = "main";
    return shaderStageInfo;
}
}

GraphicsPipelineBundle createGraphicsPipelineBundle(
    VkDevice device,
    const GraphicsPipelineConfig& config)
{
    const std::vector<char> vertexShaderCode = readBinaryFile(config.vertexShaderPath);
    const std::vector<char> fragmentShaderCode = readBinaryFile(config.fragmentShaderPath);
    const VkShaderModule vertexShaderModule = createShaderModule(device, vertexShaderCode);
    const VkShaderModule fragmentShaderModule = createShaderModule(device, fragmentShaderCode);

    const std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages = {
        makeShaderStage(VK_SHADER_STAGE_VERTEX_BIT, vertexShaderModule),
        makeShaderStage(VK_SHADER_STAGE_FRAGMENT_BIT, fragmentShaderModule),
    };

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    if (config.vertexBindingDescription != nullptr)
    {
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.pVertexBindingDescriptions = config.vertexBindingDescription;
    }
    vertexInputInfo.vertexAttributeDescriptionCount = config.vertexAttributeDescriptionCount;
    vertexInputInfo.pVertexAttributeDescriptions = config.vertexAttributeDescriptions;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = config.topology;

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = static_cast<float>(config.extent.height);
    viewport.width = static_cast<float>(config.extent.width);
    viewport.height = -static_cast<float>(config.extent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = config.extent;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = config.cullMode;
    rasterizer.frontFace = config.frontFace;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = config.depthTestEnable ? VK_TRUE : VK_FALSE;
    depthStencil.depthWriteEnable = config.depthWriteEnable ? VK_TRUE : VK_FALSE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.stencilTestEnable = VK_FALSE;

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT |
        VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT |
        VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = config.alphaBlendEnable ? VK_TRUE : VK_FALSE;
    colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
    colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &config.descriptorSetLayout;

    GraphicsPipelineBundle result{};
    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &result.layout) != VK_SUCCESS)
    {
        vkDestroyShaderModule(device, fragmentShaderModule, nullptr);
        vkDestroyShaderModule(device, vertexShaderModule, nullptr);
        throw std::runtime_error(config.layoutErrorMessage);
    }

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = static_cast<std::uint32_t>(shaderStages.size());
    pipelineInfo.pStages = shaderStages.data();
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.layout = result.layout;
    pipelineInfo.renderPass = config.renderPass;
    pipelineInfo.subpass = 0;

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &result.pipeline) != VK_SUCCESS)
    {
        vkDestroyPipelineLayout(device, result.layout, nullptr);
        vkDestroyShaderModule(device, fragmentShaderModule, nullptr);
        vkDestroyShaderModule(device, vertexShaderModule, nullptr);
        throw std::runtime_error(config.pipelineErrorMessage);
    }

    vkDestroyShaderModule(device, fragmentShaderModule, nullptr);
    vkDestroyShaderModule(device, vertexShaderModule, nullptr);
    return result;
}
