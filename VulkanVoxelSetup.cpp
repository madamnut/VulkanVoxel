#include "VulkanVoxel.h"

#include <algorithm>
#include <array>
#include <set>
#include <stdexcept>
#include <string>

#ifndef SHADER_DIR
#define SHADER_DIR "shaders"
#endif

namespace {

constexpr int kMaxFramesInFlight = 2;
constexpr std::array<const char*, 1> kDeviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
};

}  // namespace

void VulkanVoxelApp::CreateInstance() {
    if (glfwVulkanSupported() != GLFW_TRUE) {
        throw std::runtime_error("GLFW reports that Vulkan is not supported.");
    }

    std::uint32_t extensionCount = 0;
    const char** extensions = glfwGetRequiredInstanceExtensions(&extensionCount);
    if (extensions == nullptr || extensionCount == 0) {
        throw std::runtime_error("Failed to get GLFW Vulkan extensions.");
    }

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "VulkanVoxel";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = extensionCount;
    createInfo.ppEnabledExtensionNames = extensions;

    if (vkCreateInstance(&createInfo, nullptr, &instance_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Vulkan instance.");
    }
}

void VulkanVoxelApp::CreateSurface() {
    if (glfwCreateWindowSurface(instance_, window_, nullptr, &surface_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create window surface.");
    }
}

void VulkanVoxelApp::PickPhysicalDevice() {
    std::uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance_, &deviceCount, nullptr);
    if (deviceCount == 0) {
        throw std::runtime_error("No Vulkan GPU found.");
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance_, &deviceCount, devices.data());

    for (VkPhysicalDevice device : devices) {
        if (IsDeviceSuitable(device)) {
            physicalDevice_ = device;
            break;
        }
    }

    if (physicalDevice_ == VK_NULL_HANDLE) {
        throw std::runtime_error("Failed to find a suitable GPU.");
    }

    vkGetPhysicalDeviceProperties(physicalDevice_, &physicalDeviceProperties_);
}

void VulkanVoxelApp::CreateLogicalDevice() {
    const QueueFamilyIndices indices = FindQueueFamilies(physicalDevice_);
    const std::set<std::uint32_t> uniqueQueueFamilies = {
        indices.graphicsFamily.value(),
        indices.presentFamily.value(),
    };

    float queuePriority = 1.0f;
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;

    for (std::uint32_t queueFamily : uniqueQueueFamilies) {
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = queueFamily;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(queueCreateInfo);
    }

    VkPhysicalDeviceFeatures deviceFeatures{};

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.queueCreateInfoCount = static_cast<std::uint32_t>(queueCreateInfos.size());
    createInfo.pQueueCreateInfos = queueCreateInfos.data();
    createInfo.pEnabledFeatures = &deviceFeatures;
    createInfo.enabledExtensionCount = static_cast<std::uint32_t>(kDeviceExtensions.size());
    createInfo.ppEnabledExtensionNames = kDeviceExtensions.data();

    if (vkCreateDevice(physicalDevice_, &createInfo, nullptr, &device_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create logical device.");
    }

    vkGetDeviceQueue(device_, indices.graphicsFamily.value(), 0, &graphicsQueue_);
    vkGetDeviceQueue(device_, indices.presentFamily.value(), 0, &presentQueue_);
}

void VulkanVoxelApp::CreateSwapChain() {
    const SwapChainSupportDetails support = QuerySwapChainSupport(physicalDevice_);
    const VkSurfaceFormatKHR surfaceFormat = ChooseSwapSurfaceFormat(support.formats);
    const VkPresentModeKHR presentMode = ChooseSwapPresentMode(support.presentModes);
    const VkExtent2D extent = ChooseSwapExtent(support.capabilities);

    std::uint32_t imageCount = support.capabilities.minImageCount + 1;
    if (support.capabilities.maxImageCount > 0 && imageCount > support.capabilities.maxImageCount) {
        imageCount = support.capabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = surface_;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

    const QueueFamilyIndices indices = FindQueueFamilies(physicalDevice_);
    const std::uint32_t queueFamilyIndices[] = {
        indices.graphicsFamily.value(),
        indices.presentFamily.value(),
    };

    if (indices.graphicsFamily != indices.presentFamily) {
        createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        createInfo.queueFamilyIndexCount = 2;
        createInfo.pQueueFamilyIndices = queueFamilyIndices;
    } else {
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    createInfo.preTransform = support.capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;

    if (vkCreateSwapchainKHR(device_, &createInfo, nullptr, &swapChain_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create swap chain.");
    }

    vkGetSwapchainImagesKHR(device_, swapChain_, &imageCount, nullptr);
    swapChainImages_.resize(imageCount);
    vkGetSwapchainImagesKHR(device_, swapChain_, &imageCount, swapChainImages_.data());

    swapChainImageFormat_ = surfaceFormat.format;
    swapChainExtent_ = extent;
    presentMode_ = presentMode;
}

void VulkanVoxelApp::CreateImageViews() {
    swapChainImageViews_.resize(swapChainImages_.size());

    for (std::size_t i = 0; i < swapChainImages_.size(); ++i) {
        swapChainImageViews_[i] = CreateImageView(
            swapChainImages_[i],
            swapChainImageFormat_,
            VK_IMAGE_ASPECT_COLOR_BIT
        );
    }
}

void VulkanVoxelApp::CreateCommandPool() {
    const QueueFamilyIndices queueFamilies = FindQueueFamilies(physicalDevice_);

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = queueFamilies.graphicsFamily.value();

    if (vkCreateCommandPool(device_, &poolInfo, nullptr, &commandPool_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create command pool.");
    }
}

void VulkanVoxelApp::CreateDescriptorSetLayout() {
    VkDescriptorSetLayoutBinding uniformBinding{};
    uniformBinding.binding = 0;
    uniformBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uniformBinding.descriptorCount = 1;
    uniformBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutBinding samplerBinding{};
    samplerBinding.binding = 1;
    samplerBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    samplerBinding.descriptorCount = 1;
    samplerBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    const std::array<VkDescriptorSetLayoutBinding, 2> bindings = {
        uniformBinding,
        samplerBinding,
    };

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<std::uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr, &descriptorSetLayout_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create descriptor set layout.");
    }
}

void VulkanVoxelApp::CreateRenderPass() {
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = swapChainImageFormat_;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentDescription depthAttachment{};
    depthAttachment.format = FindDepthFormat();
    depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depthAttachmentRef{};
    depthAttachmentRef.attachment = 1;
    depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;
    subpass.pDepthStencilAttachment = &depthAttachmentRef;

    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    const std::array<VkAttachmentDescription, 2> attachments = {
        colorAttachment,
        depthAttachment,
    };

    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = static_cast<std::uint32_t>(attachments.size());
    renderPassInfo.pAttachments = attachments.data();
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    if (vkCreateRenderPass(device_, &renderPassInfo, nullptr, &renderPass_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create render pass.");
    }
}

void VulkanVoxelApp::CreatePipelines() {
    gFatalStage = "CreatePipelines.ReadShaders";
    const std::vector<char> skyVertShaderCode = ReadFile(SHADER_DIR "/sky.vert.spv");
    const std::vector<char> skyFragShaderCode = ReadFile(SHADER_DIR "/sky.frag.spv");
    const std::vector<char> terrainVertShaderCode = ReadFile(SHADER_DIR "/terrain.vert.spv");
    const std::vector<char> terrainFragShaderCode = ReadFile(SHADER_DIR "/terrain.frag.spv");
    const std::vector<char> worldVertShaderCode = ReadFile(SHADER_DIR "/world.vert.spv");
    const std::vector<char> worldFragShaderCode = ReadFile(SHADER_DIR "/world.frag.spv");
    const std::vector<char> overlayVertShaderCode = ReadFile(SHADER_DIR "/overlay.vert.spv");
    const std::vector<char> overlayFragShaderCode = ReadFile(SHADER_DIR "/overlay.frag.spv");
    const std::vector<char> selectionVertShaderCode = ReadFile(SHADER_DIR "/selection.vert.spv");
    const std::vector<char> selectionFragShaderCode = ReadFile(SHADER_DIR "/selection.frag.spv");

    gFatalStage = "CreatePipelines.CreateShaderModules";
    const VkShaderModule skyVertShaderModule = CreateShaderModule(skyVertShaderCode);
    const VkShaderModule skyFragShaderModule = CreateShaderModule(skyFragShaderCode);
    const VkShaderModule terrainVertShaderModule = CreateShaderModule(terrainVertShaderCode);
    const VkShaderModule terrainFragShaderModule = CreateShaderModule(terrainFragShaderCode);
    const VkShaderModule worldVertShaderModule = CreateShaderModule(worldVertShaderCode);
    const VkShaderModule worldFragShaderModule = CreateShaderModule(worldFragShaderCode);
    const VkShaderModule overlayVertShaderModule = CreateShaderModule(overlayVertShaderCode);
    const VkShaderModule overlayFragShaderModule = CreateShaderModule(overlayFragShaderCode);
    const VkShaderModule selectionVertShaderModule = CreateShaderModule(selectionVertShaderCode);
    const VkShaderModule selectionFragShaderModule = CreateShaderModule(selectionFragShaderCode);

    VkPipelineShaderStageCreateInfo worldVertStage{};
    worldVertStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    worldVertStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    worldVertStage.module = worldVertShaderModule;
    worldVertStage.pName = "main";

    VkPipelineShaderStageCreateInfo worldFragStage{};
    worldFragStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    worldFragStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    worldFragStage.module = worldFragShaderModule;
    worldFragStage.pName = "main";

    const std::array<VkPipelineShaderStageCreateInfo, 2> worldStages = {
        worldVertStage,
        worldFragStage,
    };

    VkPipelineShaderStageCreateInfo terrainVertStage{};
    terrainVertStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    terrainVertStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    terrainVertStage.module = terrainVertShaderModule;
    terrainVertStage.pName = "main";

    VkPipelineShaderStageCreateInfo terrainFragStage{};
    terrainFragStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    terrainFragStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    terrainFragStage.module = terrainFragShaderModule;
    terrainFragStage.pName = "main";

    const std::array<VkPipelineShaderStageCreateInfo, 2> terrainStages = {
        terrainVertStage,
        terrainFragStage,
    };

    VkVertexInputBindingDescription worldBindingDescription{};
    worldBindingDescription.binding = 0;
    worldBindingDescription.stride = sizeof(WorldVertex);
    worldBindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    std::array<VkVertexInputAttributeDescription, 3> worldAttributes{};
    worldAttributes[0].binding = 0;
    worldAttributes[0].location = 0;
    worldAttributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    worldAttributes[0].offset = offsetof(WorldVertex, position);
    worldAttributes[1].binding = 0;
    worldAttributes[1].location = 1;
    worldAttributes[1].format = VK_FORMAT_R32G32_SFLOAT;
    worldAttributes[1].offset = offsetof(WorldVertex, uv);
    worldAttributes[2].binding = 0;
    worldAttributes[2].location = 2;
    worldAttributes[2].format = VK_FORMAT_R32_SFLOAT;
    worldAttributes[2].offset = offsetof(WorldVertex, ao);

    VkPipelineVertexInputStateCreateInfo worldVertexInput{};
    worldVertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    worldVertexInput.vertexBindingDescriptionCount = 1;
    worldVertexInput.pVertexBindingDescriptions = &worldBindingDescription;
    worldVertexInput.vertexAttributeDescriptionCount = static_cast<std::uint32_t>(worldAttributes.size());
    worldVertexInput.pVertexAttributeDescriptions = worldAttributes.data();

    VkVertexInputBindingDescription terrainBindingDescription{};
    terrainBindingDescription.binding = 0;
    terrainBindingDescription.stride = sizeof(WorldQuadRecord);
    terrainBindingDescription.inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;

    std::array<VkVertexInputAttributeDescription, 2> terrainAttributes{};
    terrainAttributes[0].binding = 0;
    terrainAttributes[0].location = 0;
    terrainAttributes[0].format = VK_FORMAT_R32_UINT;
    terrainAttributes[0].offset = offsetof(WorldQuadRecord, packed0);
    terrainAttributes[1].binding = 0;
    terrainAttributes[1].location = 1;
    terrainAttributes[1].format = VK_FORMAT_R32_UINT;
    terrainAttributes[1].offset = offsetof(WorldQuadRecord, packed1);

    VkPipelineVertexInputStateCreateInfo terrainVertexInput{};
    terrainVertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    terrainVertexInput.vertexBindingDescriptionCount = 1;
    terrainVertexInput.pVertexBindingDescriptions = &terrainBindingDescription;
    terrainVertexInput.vertexAttributeDescriptionCount = static_cast<std::uint32_t>(terrainAttributes.size());
    terrainVertexInput.pVertexAttributeDescriptions = terrainAttributes.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(swapChainExtent_.width);
    viewport.height = static_cast<float>(swapChainExtent_.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = swapChainExtent_;

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
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo worldDepthStencil{};
    worldDepthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    worldDepthStencil.depthTestEnable = VK_TRUE;
    worldDepthStencil.depthWriteEnable = VK_TRUE;
    worldDepthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
    worldDepthStencil.depthBoundsTestEnable = VK_FALSE;
    worldDepthStencil.stencilTestEnable = VK_FALSE;

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
                                          VK_COLOR_COMPONENT_G_BIT |
                                          VK_COLOR_COMPONENT_B_BIT |
                                          VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;

    VkPipelineColorBlendAttachmentState overlayBlendAttachment = colorBlendAttachment;
    overlayBlendAttachment.blendEnable = VK_TRUE;
    overlayBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    overlayBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    overlayBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
    overlayBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    overlayBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    overlayBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

    VkPipelineColorBlendStateCreateInfo overlayColorBlending = colorBlending;
    overlayColorBlending.pAttachments = &overlayBlendAttachment;

    VkPipelineShaderStageCreateInfo skyVertStage{};
    skyVertStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    skyVertStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    skyVertStage.module = skyVertShaderModule;
    skyVertStage.pName = "main";

    VkPipelineShaderStageCreateInfo skyFragStage{};
    skyFragStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    skyFragStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    skyFragStage.module = skyFragShaderModule;
    skyFragStage.pName = "main";

    const std::array<VkPipelineShaderStageCreateInfo, 2> skyStages = {
        skyVertStage,
        skyFragStage,
    };

    VkPipelineVertexInputStateCreateInfo skyVertexInput{};
    skyVertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineRasterizationStateCreateInfo skyRasterizer = rasterizer;
    skyRasterizer.cullMode = VK_CULL_MODE_NONE;

    VkPipelineDepthStencilStateCreateInfo skyDepthStencil{};
    skyDepthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    skyDepthStencil.depthTestEnable = VK_FALSE;
    skyDepthStencil.depthWriteEnable = VK_FALSE;
    skyDepthStencil.depthBoundsTestEnable = VK_FALSE;
    skyDepthStencil.stencilTestEnable = VK_FALSE;

    VkPipelineLayoutCreateInfo skyLayoutInfo{};
    skyLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    skyLayoutInfo.setLayoutCount = 1;
    skyLayoutInfo.pSetLayouts = &descriptorSetLayout_;

    gFatalStage = "CreatePipelines.Sky";
    if (vkCreatePipelineLayout(device_, &skyLayoutInfo, nullptr, &skyPipelineLayout_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create sky pipeline layout.");
    }

    VkGraphicsPipelineCreateInfo skyPipelineInfo{};
    skyPipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    skyPipelineInfo.stageCount = static_cast<std::uint32_t>(skyStages.size());
    skyPipelineInfo.pStages = skyStages.data();
    skyPipelineInfo.pVertexInputState = &skyVertexInput;
    skyPipelineInfo.pInputAssemblyState = &inputAssembly;
    skyPipelineInfo.pViewportState = &viewportState;
    skyPipelineInfo.pRasterizationState = &skyRasterizer;
    skyPipelineInfo.pMultisampleState = &multisampling;
    skyPipelineInfo.pDepthStencilState = &skyDepthStencil;
    skyPipelineInfo.pColorBlendState = &colorBlending;
    skyPipelineInfo.layout = skyPipelineLayout_;
    skyPipelineInfo.renderPass = renderPass_;
    skyPipelineInfo.subpass = 0;

    if (vkCreateGraphicsPipelines(
            device_,
            VK_NULL_HANDLE,
            1,
            &skyPipelineInfo,
            nullptr,
            &skyPipeline_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create sky pipeline.");
    }

    VkPipelineLayoutCreateInfo worldLayoutInfo{};
    worldLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    worldLayoutInfo.setLayoutCount = 1;
    worldLayoutInfo.pSetLayouts = &descriptorSetLayout_;

    gFatalStage = "CreatePipelines.WorldLayout";
    if (vkCreatePipelineLayout(device_, &worldLayoutInfo, nullptr, &worldPipelineLayout_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create world pipeline layout.");
    }

    VkPushConstantRange terrainPushConstantRange{};
    terrainPushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    terrainPushConstantRange.offset = 0;
    terrainPushConstantRange.size = sizeof(TerrainPushConstants);

    VkPipelineLayoutCreateInfo terrainLayoutInfo = worldLayoutInfo;
    terrainLayoutInfo.pushConstantRangeCount = 1;
    terrainLayoutInfo.pPushConstantRanges = &terrainPushConstantRange;

    gFatalStage = "CreatePipelines.TerrainLayout";
    if (vkCreatePipelineLayout(device_, &terrainLayoutInfo, nullptr, &terrainPipelineLayout_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create terrain pipeline layout.");
    }

    VkGraphicsPipelineCreateInfo terrainPipelineInfo{};
    terrainPipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    terrainPipelineInfo.stageCount = static_cast<std::uint32_t>(terrainStages.size());
    terrainPipelineInfo.pStages = terrainStages.data();
    terrainPipelineInfo.pVertexInputState = &terrainVertexInput;
    terrainPipelineInfo.pInputAssemblyState = &inputAssembly;
    terrainPipelineInfo.pViewportState = &viewportState;
    terrainPipelineInfo.pRasterizationState = &rasterizer;
    terrainPipelineInfo.pMultisampleState = &multisampling;
    terrainPipelineInfo.pDepthStencilState = &worldDepthStencil;
    terrainPipelineInfo.pColorBlendState = &colorBlending;
    terrainPipelineInfo.layout = terrainPipelineLayout_;
    terrainPipelineInfo.renderPass = renderPass_;
    terrainPipelineInfo.subpass = 0;

    gFatalStage = "CreatePipelines.TerrainPipeline";
    if (vkCreateGraphicsPipelines(
            device_,
            VK_NULL_HANDLE,
            1,
            &terrainPipelineInfo,
            nullptr,
            &terrainPipeline_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create terrain pipeline.");
    }

    VkGraphicsPipelineCreateInfo worldPipelineInfo{};
    worldPipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    worldPipelineInfo.stageCount = static_cast<std::uint32_t>(worldStages.size());
    worldPipelineInfo.pStages = worldStages.data();
    worldPipelineInfo.pVertexInputState = &worldVertexInput;
    worldPipelineInfo.pInputAssemblyState = &inputAssembly;
    worldPipelineInfo.pViewportState = &viewportState;
    worldPipelineInfo.pRasterizationState = &rasterizer;
    worldPipelineInfo.pMultisampleState = &multisampling;
    worldPipelineInfo.pDepthStencilState = &worldDepthStencil;
    worldPipelineInfo.pColorBlendState = &colorBlending;
    worldPipelineInfo.layout = worldPipelineLayout_;
    worldPipelineInfo.renderPass = renderPass_;
    worldPipelineInfo.subpass = 0;

    gFatalStage = "CreatePipelines.WorldPipeline";
    if (vkCreateGraphicsPipelines(
            device_,
            VK_NULL_HANDLE,
            1,
            &worldPipelineInfo,
            nullptr,
            &worldPipeline_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create world pipeline.");
    }

    VkPipelineShaderStageCreateInfo overlayVertStage{};
    overlayVertStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    overlayVertStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    overlayVertStage.module = overlayVertShaderModule;
    overlayVertStage.pName = "main";

    VkPipelineShaderStageCreateInfo overlayFragStage{};
    overlayFragStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    overlayFragStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    overlayFragStage.module = overlayFragShaderModule;
    overlayFragStage.pName = "main";

    const std::array<VkPipelineShaderStageCreateInfo, 2> overlayStages = {
        overlayVertStage,
        overlayFragStage,
    };

    VkVertexInputBindingDescription overlayBindingDescription{};
    overlayBindingDescription.binding = 0;
    overlayBindingDescription.stride = sizeof(OverlayVertex);
    overlayBindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    std::array<VkVertexInputAttributeDescription, 3> overlayAttributes{};
    overlayAttributes[0].binding = 0;
    overlayAttributes[0].location = 0;
    overlayAttributes[0].format = VK_FORMAT_R32G32_SFLOAT;
    overlayAttributes[0].offset = offsetof(OverlayVertex, position);
    overlayAttributes[1].binding = 0;
    overlayAttributes[1].location = 1;
    overlayAttributes[1].format = VK_FORMAT_R32G32_SFLOAT;
    overlayAttributes[1].offset = offsetof(OverlayVertex, uv);
    overlayAttributes[2].binding = 0;
    overlayAttributes[2].location = 2;
    overlayAttributes[2].format = VK_FORMAT_R32G32B32_SFLOAT;
    overlayAttributes[2].offset = offsetof(OverlayVertex, color);

    VkPipelineVertexInputStateCreateInfo overlayVertexInput{};
    overlayVertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    overlayVertexInput.vertexBindingDescriptionCount = 1;
    overlayVertexInput.pVertexBindingDescriptions = &overlayBindingDescription;
    overlayVertexInput.vertexAttributeDescriptionCount = static_cast<std::uint32_t>(overlayAttributes.size());
    overlayVertexInput.pVertexAttributeDescriptions = overlayAttributes.data();

    VkPipelineRasterizationStateCreateInfo overlayRasterizer = rasterizer;
    overlayRasterizer.cullMode = VK_CULL_MODE_NONE;

    VkPipelineDepthStencilStateCreateInfo overlayDepthStencil{};
    overlayDepthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    overlayDepthStencil.depthTestEnable = VK_FALSE;
    overlayDepthStencil.depthWriteEnable = VK_FALSE;
    overlayDepthStencil.depthBoundsTestEnable = VK_FALSE;
    overlayDepthStencil.stencilTestEnable = VK_FALSE;

    VkPipelineLayoutCreateInfo overlayLayoutInfo{};
    overlayLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    overlayLayoutInfo.setLayoutCount = 1;
    overlayLayoutInfo.pSetLayouts = &descriptorSetLayout_;

    gFatalStage = "CreatePipelines.Overlay";
    if (vkCreatePipelineLayout(device_, &overlayLayoutInfo, nullptr, &overlayPipelineLayout_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create overlay pipeline layout.");
    }

    VkGraphicsPipelineCreateInfo overlayPipelineInfo{};
    overlayPipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    overlayPipelineInfo.stageCount = static_cast<std::uint32_t>(overlayStages.size());
    overlayPipelineInfo.pStages = overlayStages.data();
    overlayPipelineInfo.pVertexInputState = &overlayVertexInput;
    overlayPipelineInfo.pInputAssemblyState = &inputAssembly;
    overlayPipelineInfo.pViewportState = &viewportState;
    overlayPipelineInfo.pRasterizationState = &overlayRasterizer;
    overlayPipelineInfo.pMultisampleState = &multisampling;
    overlayPipelineInfo.pDepthStencilState = &overlayDepthStencil;
    overlayPipelineInfo.pColorBlendState = &overlayColorBlending;
    overlayPipelineInfo.layout = overlayPipelineLayout_;
    overlayPipelineInfo.renderPass = renderPass_;
    overlayPipelineInfo.subpass = 0;

    if (vkCreateGraphicsPipelines(
            device_,
            VK_NULL_HANDLE,
            1,
            &overlayPipelineInfo,
            nullptr,
            &overlayPipeline_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create overlay pipeline.");
    }

    VkPipelineShaderStageCreateInfo selectionVertStage{};
    selectionVertStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    selectionVertStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    selectionVertStage.module = selectionVertShaderModule;
    selectionVertStage.pName = "main";

    VkPipelineShaderStageCreateInfo selectionFragStage{};
    selectionFragStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    selectionFragStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    selectionFragStage.module = selectionFragShaderModule;
    selectionFragStage.pName = "main";

    const std::array<VkPipelineShaderStageCreateInfo, 2> selectionStages = {
        selectionVertStage,
        selectionFragStage,
    };

    VkVertexInputBindingDescription selectionBindingDescription{};
    selectionBindingDescription.binding = 0;
    selectionBindingDescription.stride = sizeof(SelectionVertex);
    selectionBindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription selectionAttribute{};
    selectionAttribute.binding = 0;
    selectionAttribute.location = 0;
    selectionAttribute.format = VK_FORMAT_R32G32B32_SFLOAT;
    selectionAttribute.offset = offsetof(SelectionVertex, position);

    VkPipelineVertexInputStateCreateInfo selectionVertexInput{};
    selectionVertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    selectionVertexInput.vertexBindingDescriptionCount = 1;
    selectionVertexInput.pVertexBindingDescriptions = &selectionBindingDescription;
    selectionVertexInput.vertexAttributeDescriptionCount = 1;
    selectionVertexInput.pVertexAttributeDescriptions = &selectionAttribute;

    VkPipelineInputAssemblyStateCreateInfo selectionInputAssembly = inputAssembly;
    selectionInputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;

    VkPipelineRasterizationStateCreateInfo selectionRasterizer = rasterizer;
    selectionRasterizer.cullMode = VK_CULL_MODE_NONE;

    VkPipelineDepthStencilStateCreateInfo selectionDepthStencil = worldDepthStencil;
    selectionDepthStencil.depthWriteEnable = VK_FALSE;

    VkPipelineLayoutCreateInfo selectionLayoutInfo{};
    selectionLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    selectionLayoutInfo.setLayoutCount = 1;
    selectionLayoutInfo.pSetLayouts = &descriptorSetLayout_;

    gFatalStage = "CreatePipelines.Selection";
    if (vkCreatePipelineLayout(device_, &selectionLayoutInfo, nullptr, &selectionPipelineLayout_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create selection pipeline layout.");
    }

    VkGraphicsPipelineCreateInfo selectionPipelineInfo{};
    selectionPipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    selectionPipelineInfo.stageCount = static_cast<std::uint32_t>(selectionStages.size());
    selectionPipelineInfo.pStages = selectionStages.data();
    selectionPipelineInfo.pVertexInputState = &selectionVertexInput;
    selectionPipelineInfo.pInputAssemblyState = &selectionInputAssembly;
    selectionPipelineInfo.pViewportState = &viewportState;
    selectionPipelineInfo.pRasterizationState = &selectionRasterizer;
    selectionPipelineInfo.pMultisampleState = &multisampling;
    selectionPipelineInfo.pDepthStencilState = &selectionDepthStencil;
    selectionPipelineInfo.pColorBlendState = &colorBlending;
    selectionPipelineInfo.layout = selectionPipelineLayout_;
    selectionPipelineInfo.renderPass = renderPass_;
    selectionPipelineInfo.subpass = 0;

    if (vkCreateGraphicsPipelines(
            device_,
            VK_NULL_HANDLE,
            1,
            &selectionPipelineInfo,
            nullptr,
            &selectionPipeline_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create selection pipeline.");
    }

    vkDestroyShaderModule(device_, skyFragShaderModule, nullptr);
    vkDestroyShaderModule(device_, skyVertShaderModule, nullptr);
    vkDestroyShaderModule(device_, selectionFragShaderModule, nullptr);
    vkDestroyShaderModule(device_, selectionVertShaderModule, nullptr);
    vkDestroyShaderModule(device_, overlayFragShaderModule, nullptr);
    vkDestroyShaderModule(device_, overlayVertShaderModule, nullptr);
    vkDestroyShaderModule(device_, worldFragShaderModule, nullptr);
    vkDestroyShaderModule(device_, terrainFragShaderModule, nullptr);
    vkDestroyShaderModule(device_, terrainVertShaderModule, nullptr);
    vkDestroyShaderModule(device_, worldVertShaderModule, nullptr);
    gFatalStage = "CreatePipelines.Done";
}

void VulkanVoxelApp::CreateFramebuffers() {
    swapChainFramebuffers_.resize(swapChainImageViews_.size());

    for (std::size_t i = 0; i < swapChainImageViews_.size(); ++i) {
        const std::array<VkImageView, 2> attachments = {
            swapChainImageViews_[i],
            depthImageView_,
        };

        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = renderPass_;
        framebufferInfo.attachmentCount = static_cast<std::uint32_t>(attachments.size());
        framebufferInfo.pAttachments = attachments.data();
        framebufferInfo.width = swapChainExtent_.width;
        framebufferInfo.height = swapChainExtent_.height;
        framebufferInfo.layers = 1;

        if (vkCreateFramebuffer(device_, &framebufferInfo, nullptr, &swapChainFramebuffers_[i]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create framebuffer.");
        }
    }
}

void VulkanVoxelApp::CreateCommandBuffers() {
    commandBuffers_.resize(swapChainFramebuffers_.size());

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool_;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = static_cast<std::uint32_t>(commandBuffers_.size());

    if (vkAllocateCommandBuffers(device_, &allocInfo, commandBuffers_.data()) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate command buffers.");
    }
}

void VulkanVoxelApp::CreateSyncObjects() {
    imageAvailableSemaphores_.resize(kMaxFramesInFlight);
    renderFinishedSemaphores_.resize(kMaxFramesInFlight);
    inFlightFences_.resize(kMaxFramesInFlight);
    imagesInFlight_.resize(swapChainImages_.size(), VK_NULL_HANDLE);

    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (int i = 0; i < kMaxFramesInFlight; ++i) {
        if (vkCreateSemaphore(device_, &semaphoreInfo, nullptr, &imageAvailableSemaphores_[i]) != VK_SUCCESS ||
            vkCreateSemaphore(device_, &semaphoreInfo, nullptr, &renderFinishedSemaphores_[i]) != VK_SUCCESS ||
            vkCreateFence(device_, &fenceInfo, nullptr, &inFlightFences_[i]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create sync objects.");
        }
    }
}

QueueFamilyIndices VulkanVoxelApp::FindQueueFamilies(VkPhysicalDevice device) const {
    QueueFamilyIndices indices;

    std::uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

    for (std::uint32_t i = 0; i < queueFamilyCount; ++i) {
        if ((queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0) {
            indices.graphicsFamily = i;
        }

        VkBool32 presentSupport = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface_, &presentSupport);
        if (presentSupport == VK_TRUE) {
            indices.presentFamily = i;
        }

        if (indices.IsComplete()) {
            break;
        }
    }

    return indices;
}

bool VulkanVoxelApp::IsDeviceSuitable(VkPhysicalDevice device) const {
    const QueueFamilyIndices indices = FindQueueFamilies(device);
    const bool extensionsSupported = CheckDeviceExtensionSupport(device);

    bool swapChainAdequate = false;
    if (extensionsSupported) {
        const SwapChainSupportDetails support = QuerySwapChainSupport(device);
        swapChainAdequate = !support.formats.empty() && !support.presentModes.empty();
    }

    return indices.IsComplete() && extensionsSupported && swapChainAdequate;
}

bool VulkanVoxelApp::CheckDeviceExtensionSupport(VkPhysicalDevice device) const {
    std::uint32_t extensionCount = 0;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

    std::set<std::string> requiredExtensions(kDeviceExtensions.begin(), kDeviceExtensions.end());
    for (const VkExtensionProperties& extension : availableExtensions) {
        requiredExtensions.erase(extension.extensionName);
    }

    return requiredExtensions.empty();
}

SwapChainSupportDetails VulkanVoxelApp::QuerySwapChainSupport(VkPhysicalDevice device) const {
    SwapChainSupportDetails support;

    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface_, &support.capabilities);

    std::uint32_t formatCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface_, &formatCount, nullptr);
    if (formatCount > 0) {
        support.formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface_, &formatCount, support.formats.data());
    }

    std::uint32_t presentModeCount = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface_, &presentModeCount, nullptr);
    if (presentModeCount > 0) {
        support.presentModes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(
            device,
            surface_,
            &presentModeCount,
            support.presentModes.data()
        );
    }

    return support;
}

VkSurfaceFormatKHR VulkanVoxelApp::ChooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& formats) const {
    for (const VkSurfaceFormatKHR& format : formats) {
        if (format.format == VK_FORMAT_B8G8R8A8_SRGB &&
            format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return format;
        }
    }

    return formats.front();
}

VkPresentModeKHR VulkanVoxelApp::ChooseSwapPresentMode(const std::vector<VkPresentModeKHR>& presentModes) const {
    (void)presentModes;
    return VK_PRESENT_MODE_IMMEDIATE_KHR;
}

VkExtent2D VulkanVoxelApp::ChooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) const {
    if (capabilities.currentExtent.width != UINT32_MAX) {
        return capabilities.currentExtent;
    }

    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window_, &width, &height);

    VkExtent2D actualExtent = {
        static_cast<std::uint32_t>(width),
        static_cast<std::uint32_t>(height),
    };

    actualExtent.width = std::clamp(
        actualExtent.width,
        capabilities.minImageExtent.width,
        capabilities.maxImageExtent.width
    );
    actualExtent.height = std::clamp(
        actualExtent.height,
        capabilities.minImageExtent.height,
        capabilities.maxImageExtent.height
    );

    return actualExtent;
}

VkShaderModule VulkanVoxelApp::CreateShaderModule(const std::vector<char>& code) const {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const std::uint32_t*>(code.data());

    VkShaderModule shaderModule = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device_, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create shader module.");
    }

    return shaderModule;
}
