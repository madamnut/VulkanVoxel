#include "VulkanVoxel.h"

#define NOMINMAX
#include <Windows.h>
#include <wincodec.h>
#include <wrl/client.h>

#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {

constexpr int kMaxFramesInFlight = 2;
constexpr float kPi = 3.14159265358979323846f;

struct ScreenshotComScope {
    bool shouldUninitialize = false;

    ScreenshotComScope() {
        const HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        if (SUCCEEDED(hr)) {
            shouldUninitialize = true;
            return;
        }

        if (hr != RPC_E_CHANGED_MODE) {
            throw std::runtime_error("Failed to initialize COM for screenshot.");
        }
    }

    ~ScreenshotComScope() {
        if (shouldUninitialize) {
            CoUninitialize();
        }
    }
};

}  // namespace

void VulkanVoxelApp::DrawFrame() {
    vkWaitForFences(device_, 1, &inFlightFences_[currentFrame_], VK_TRUE, UINT64_MAX);
    TryCleanupRetiredWorldRenderBatches();

    if (overlayDirty_) {
        vkWaitForFences(
            device_,
            static_cast<std::uint32_t>(inFlightFences_.size()),
            inFlightFences_.data(),
            VK_TRUE,
            UINT64_MAX
        );
        UploadOverlayVertices();
        overlayDirty_ = false;
    }

    UpdateCelestialVertices(static_cast<std::uint32_t>(currentFrame_));
    UpdatePlayerRenderMesh();
    UpdateUniformBuffer(static_cast<std::uint32_t>(currentFrame_));

    std::uint32_t imageIndex = 0;
    const VkResult acquireResult = vkAcquireNextImageKHR(
        device_,
        swapChain_,
        UINT64_MAX,
        imageAvailableSemaphores_[currentFrame_],
        VK_NULL_HANDLE,
        &imageIndex
    );

    if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR) {
        RecreateSwapChain();
        return;
    }
    if (acquireResult != VK_SUCCESS && acquireResult != VK_SUBOPTIMAL_KHR) {
        throw std::runtime_error("Failed to acquire swap chain image.");
    }

    if (imagesInFlight_[imageIndex] != VK_NULL_HANDLE) {
        vkWaitForFences(device_, 1, &imagesInFlight_[imageIndex], VK_TRUE, UINT64_MAX);
    }
    imagesInFlight_[imageIndex] = inFlightFences_[currentFrame_];

    if (screenshotRequested_) {
        PrepareScreenshotCapture();
    }

    vkResetFences(device_, 1, &inFlightFences_[currentFrame_]);
    vkResetCommandBuffer(commandBuffers_[imageIndex], 0);
    RecordCommandBuffer(commandBuffers_[imageIndex], imageIndex);

    const VkSemaphore waitSemaphores[] = {imageAvailableSemaphores_[currentFrame_]};
    const VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    const VkSemaphore signalSemaphores[] = {renderFinishedSemaphores_[currentFrame_]};

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffers_[imageIndex];
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    if (vkQueueSubmit(graphicsQueue_, 1, &submitInfo, inFlightFences_[currentFrame_]) != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit draw commands.");
    }

    const VkSwapchainKHR swapChains[] = {swapChain_};
    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapChains;
    presentInfo.pImageIndices = &imageIndex;

    const VkResult presentResult = vkQueuePresentKHR(presentQueue_, &presentInfo);
    if (presentResult == VK_ERROR_OUT_OF_DATE_KHR || presentResult == VK_SUBOPTIMAL_KHR) {
        RecreateSwapChain();
    } else if (presentResult != VK_SUCCESS) {
        throw std::runtime_error("Failed to present swap chain image.");
    }

    if (screenshotCapturePending_) {
        vkWaitForFences(device_, 1, &inFlightFences_[currentFrame_], VK_TRUE, UINT64_MAX);
        FinalizeScreenshotCapture();
    }

    currentFrame_ = (currentFrame_ + 1) % kMaxFramesInFlight;
}

void VulkanVoxelApp::PrepareScreenshotCapture() {
    if (screenshotStagingBuffer_ != VK_NULL_HANDLE) {
        vkDestroyBuffer(device_, screenshotStagingBuffer_, nullptr);
        screenshotStagingBuffer_ = VK_NULL_HANDLE;
    }
    if (screenshotStagingBufferMemory_ != VK_NULL_HANDLE) {
        vkFreeMemory(device_, screenshotStagingBufferMemory_, nullptr);
        screenshotStagingBufferMemory_ = VK_NULL_HANDLE;
    }

    screenshotWidth_ = swapChainExtent_.width;
    screenshotHeight_ = swapChainExtent_.height;
    screenshotStagingBufferSize_ =
        static_cast<VkDeviceSize>(screenshotWidth_) *
        static_cast<VkDeviceSize>(screenshotHeight_) * 4;

    CreateBuffer(
        screenshotStagingBufferSize_,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        screenshotStagingBuffer_,
        screenshotStagingBufferMemory_
    );

    screenshotCapturePending_ = true;
}

void VulkanVoxelApp::RecordScreenshotCopyCommands(VkCommandBuffer commandBuffer, std::uint32_t imageIndex) {
    if (!screenshotCapturePending_) {
        return;
    }

    VkImageMemoryBarrier toTransferBarrier{};
    toTransferBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toTransferBarrier.oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    toTransferBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    toTransferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransferBarrier.image = swapChainImages_[imageIndex];
    toTransferBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    toTransferBarrier.subresourceRange.baseMipLevel = 0;
    toTransferBarrier.subresourceRange.levelCount = 1;
    toTransferBarrier.subresourceRange.baseArrayLayer = 0;
    toTransferBarrier.subresourceRange.layerCount = 1;
    toTransferBarrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    toTransferBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        0,
        nullptr,
        0,
        nullptr,
        1,
        &toTransferBarrier
    );

    VkBufferImageCopy copyRegion{};
    copyRegion.bufferOffset = 0;
    copyRegion.bufferRowLength = 0;
    copyRegion.bufferImageHeight = 0;
    copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.imageSubresource.mipLevel = 0;
    copyRegion.imageSubresource.baseArrayLayer = 0;
    copyRegion.imageSubresource.layerCount = 1;
    copyRegion.imageOffset = {0, 0, 0};
    copyRegion.imageExtent = {screenshotWidth_, screenshotHeight_, 1};

    vkCmdCopyImageToBuffer(
        commandBuffer,
        swapChainImages_[imageIndex],
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        screenshotStagingBuffer_,
        1,
        &copyRegion
    );

    VkImageMemoryBarrier toPresentBarrier = toTransferBarrier;
    toPresentBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    toPresentBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    toPresentBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    toPresentBarrier.dstAccessMask = 0;

    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        0,
        0,
        nullptr,
        0,
        nullptr,
        1,
        &toPresentBarrier
    );
}

void VulkanVoxelApp::FinalizeScreenshotCapture() {
    if (!screenshotCapturePending_ || screenshotStagingBufferMemory_ == VK_NULL_HANDLE) {
        return;
    }

    std::vector<std::uint8_t> rgbaPixels(static_cast<std::size_t>(screenshotStagingBufferSize_));
    void* mappedData = nullptr;
    if (vkMapMemory(device_, screenshotStagingBufferMemory_, 0, screenshotStagingBufferSize_, 0, &mappedData) != VK_SUCCESS) {
        throw std::runtime_error("Failed to map screenshot staging buffer.");
    }

    const auto* sourcePixels = static_cast<const std::uint8_t*>(mappedData);
    const bool isBgraFormat =
        swapChainImageFormat_ == VK_FORMAT_B8G8R8A8_UNORM ||
        swapChainImageFormat_ == VK_FORMAT_B8G8R8A8_SRGB;

    for (std::size_t i = 0; i < rgbaPixels.size(); i += 4) {
        if (isBgraFormat) {
            rgbaPixels[i + 0] = sourcePixels[i + 2];
            rgbaPixels[i + 1] = sourcePixels[i + 1];
            rgbaPixels[i + 2] = sourcePixels[i + 0];
            rgbaPixels[i + 3] = sourcePixels[i + 3];
        } else {
            rgbaPixels[i + 0] = sourcePixels[i + 0];
            rgbaPixels[i + 1] = sourcePixels[i + 1];
            rgbaPixels[i + 2] = sourcePixels[i + 2];
            rgbaPixels[i + 3] = sourcePixels[i + 3];
        }
    }

    vkUnmapMemory(device_, screenshotStagingBufferMemory_);
    SaveScreenshotPng(rgbaPixels, screenshotWidth_, screenshotHeight_);

    vkDestroyBuffer(device_, screenshotStagingBuffer_, nullptr);
    vkFreeMemory(device_, screenshotStagingBufferMemory_, nullptr);
    screenshotStagingBuffer_ = VK_NULL_HANDLE;
    screenshotStagingBufferMemory_ = VK_NULL_HANDLE;
    screenshotStagingBufferSize_ = 0;
    screenshotCapturePending_ = false;
    screenshotRequested_ = false;
}

void VulkanVoxelApp::SaveScreenshotPng(
    const std::vector<std::uint8_t>& rgbaPixels,
    std::uint32_t width,
    std::uint32_t height
) const {
    ScreenshotComScope comScope;

    namespace fs = std::filesystem;
    const fs::path screenshotDirectory = fs::path(ASSET_DIR) / "screenshots";
    fs::create_directories(screenshotDirectory);

    const auto now = std::chrono::system_clock::now();
    const std::time_t nowTime = std::chrono::system_clock::to_time_t(now);
    std::tm localTime{};
    localtime_s(&localTime, &nowTime);

    std::ostringstream nameStream;
    nameStream << "screenshot_" << std::put_time(&localTime, "%Y-%m-%d_%H-%M-%S") << ".png";
    const fs::path screenshotPath = screenshotDirectory / nameStream.str();

    Microsoft::WRL::ComPtr<IWICImagingFactory> factory;
    if (FAILED(CoCreateInstance(
            CLSID_WICImagingFactory,
            nullptr,
            CLSCTX_INPROC_SERVER,
            IID_PPV_ARGS(&factory)))) {
        throw std::runtime_error("Failed to create WIC factory for screenshot.");
    }

    Microsoft::WRL::ComPtr<IWICStream> stream;
    if (FAILED(factory->CreateStream(&stream))) {
        throw std::runtime_error("Failed to create screenshot stream.");
    }
    if (FAILED(stream->InitializeFromFilename(screenshotPath.c_str(), GENERIC_WRITE))) {
        throw std::runtime_error("Failed to open screenshot file for writing.");
    }

    Microsoft::WRL::ComPtr<IWICBitmapEncoder> encoder;
    if (FAILED(factory->CreateEncoder(GUID_ContainerFormatPng, nullptr, &encoder))) {
        throw std::runtime_error("Failed to create PNG encoder.");
    }
    if (FAILED(encoder->Initialize(stream.Get(), WICBitmapEncoderNoCache))) {
        throw std::runtime_error("Failed to initialize PNG encoder.");
    }

    Microsoft::WRL::ComPtr<IWICBitmapFrameEncode> frame;
    Microsoft::WRL::ComPtr<IPropertyBag2> propertyBag;
    if (FAILED(encoder->CreateNewFrame(&frame, &propertyBag))) {
        throw std::runtime_error("Failed to create PNG frame.");
    }
    if (FAILED(frame->Initialize(propertyBag.Get()))) {
        throw std::runtime_error("Failed to initialize PNG frame.");
    }
    if (FAILED(frame->SetSize(width, height))) {
        throw std::runtime_error("Failed to set PNG frame size.");
    }

    WICPixelFormatGUID pixelFormat = GUID_WICPixelFormat32bppRGBA;
    if (FAILED(frame->SetPixelFormat(&pixelFormat))) {
        throw std::runtime_error("Failed to set PNG pixel format.");
    }

    const UINT stride = width * 4;
    const UINT imageSize = static_cast<UINT>(rgbaPixels.size());
    if (FAILED(frame->WritePixels(height, stride, imageSize, const_cast<BYTE*>(rgbaPixels.data())))) {
        throw std::runtime_error("Failed to write screenshot pixels.");
    }
    if (FAILED(frame->Commit())) {
        throw std::runtime_error("Failed to finalize PNG frame.");
    }
    if (FAILED(encoder->Commit())) {
        throw std::runtime_error("Failed to finalize PNG file.");
    }
}

void VulkanVoxelApp::UpdateUniformBuffer(std::uint32_t frameIndex) {
    UniformBufferObject ubo{};

    const float aspect = static_cast<float>(swapChainExtent_.width) /
                         static_cast<float>(swapChainExtent_.height);
    const Vec3 renderCameraPosition = GetRenderCameraPosition();
    const Vec3 renderCameraForward = GetRenderCameraForward();
    const float fovYRadians = 70.0f * kPi / 180.0f;
    const float tanHalfFovY = std::tan(fovYRadians * 0.5f);
    const float tanHalfFovX = tanHalfFovY * aspect;
    const Vec3 worldUp{0.0f, 1.0f, 0.0f};
    Vec3 renderCameraRight = Normalize(Cross(renderCameraForward, worldUp));
    if (Length(renderCameraRight) <= 0.00001f) {
        renderCameraRight = {1.0f, 0.0f, 0.0f};
    }
    const Vec3 renderCameraUp = Normalize(Cross(renderCameraRight, renderCameraForward));

    const Mat4 projection = Perspective(fovYRadians, aspect, 0.1f, 2048.0f);
    const Mat4 view = LookAt(
        renderCameraPosition,
        renderCameraPosition + renderCameraForward,
        worldUp
    );
    ubo.viewProj = Multiply(projection, view);
    ubo.cameraRight[0] = renderCameraRight.x;
    ubo.cameraRight[1] = renderCameraRight.y;
    ubo.cameraRight[2] = renderCameraRight.z;
    ubo.cameraRight[3] = 0.0f;
    ubo.cameraUp[0] = renderCameraUp.x;
    ubo.cameraUp[1] = renderCameraUp.y;
    ubo.cameraUp[2] = renderCameraUp.z;
    ubo.cameraUp[3] = 0.0f;
    ubo.cameraForward[0] = renderCameraForward.x;
    ubo.cameraForward[1] = renderCameraForward.y;
    ubo.cameraForward[2] = renderCameraForward.z;
    ubo.cameraForward[3] = 0.0f;
    ubo.projectionParams[0] = tanHalfFovX;
    ubo.projectionParams[1] = tanHalfFovY;
    ubo.projectionParams[2] = 0.0f;
    ubo.projectionParams[3] = 0.0f;

    void* mappedData = nullptr;
    if (vkMapMemory(
            device_,
            uniformBuffersMemory_[frameIndex],
            0,
            sizeof(ubo),
            0,
            &mappedData) != VK_SUCCESS) {
        throw std::runtime_error("Failed to map uniform buffer.");
    }

    std::memcpy(mappedData, &ubo, sizeof(ubo));
    vkUnmapMemory(device_, uniformBuffersMemory_[frameIndex]);
}

void VulkanVoxelApp::RecordCommandBuffer(VkCommandBuffer commandBuffer, std::uint32_t imageIndex) {
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
        throw std::runtime_error("Failed to begin command buffer.");
    }

    std::array<VkClearValue, 2> clearValues{};
    clearValues[0].color.float32[0] = 0.0f;
    clearValues[0].color.float32[1] = 0.0f;
    clearValues[0].color.float32[2] = 0.0f;
    clearValues[0].color.float32[3] = 1.0f;
    clearValues[1].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = renderPass_;
    renderPassInfo.framebuffer = swapChainFramebuffers_[imageIndex];
    renderPassInfo.renderArea.offset = {0, 0};
    renderPassInfo.renderArea.extent = swapChainExtent_;
    renderPassInfo.clearValueCount = static_cast<std::uint32_t>(clearValues.size());
    renderPassInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

    drawCallCount_ = 0;

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, skyPipeline_);
    vkCmdBindDescriptorSets(
        commandBuffer,
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        skyPipelineLayout_,
        0,
        1,
        &descriptorSets_[currentFrame_],
        0,
        nullptr
    );
    vkCmdDraw(commandBuffer, 3, 1, 0, 0);
    ++drawCallCount_;

    if (sunVertexCount_ > 0) {
        const VkBuffer overlayVertexBuffers[] = {celestialVertexBuffers_[currentFrame_]};
        const VkDeviceSize overlayOffsets[] = {0};

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, overlayPipeline_);
        vkCmdBindDescriptorSets(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            overlayPipelineLayout_,
            0,
            1,
            &sunDescriptorSets_[currentFrame_],
            0,
            nullptr
        );
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, overlayVertexBuffers, overlayOffsets);
        vkCmdDraw(commandBuffer, sunVertexCount_, 1, 0, 0);
        ++drawCallCount_;
    }

    if (moonVertexCount_ > 0) {
        const VkBuffer overlayVertexBuffers[] = {celestialVertexBuffers_[currentFrame_]};
        const VkDeviceSize overlayOffsets[] = {0};

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, overlayPipeline_);
        vkCmdBindDescriptorSets(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            overlayPipelineLayout_,
            0,
            1,
            &moonDescriptorSets_[currentFrame_],
            0,
            nullptr
        );
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, overlayVertexBuffers, overlayOffsets);
        vkCmdDraw(commandBuffer, moonVertexCount_, 1, sunVertexCount_, 0);
        ++drawCallCount_;
    }

    if (!worldRenderBatches_.empty()) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, worldPipeline_);
        vkCmdBindDescriptorSets(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            worldPipelineLayout_,
            0,
            1,
            &descriptorSets_[currentFrame_],
            0,
            nullptr
        );

        for (const auto& [id, batch] : worldRenderBatches_) {
            (void)id;
            if (batch.vertexCount == 0 || batch.indexCount == 0) {
                continue;
            }

            const VkBuffer worldVertexBuffers[] = {batch.vertexBuffer};
            const VkDeviceSize worldOffsets[] = {0};
            vkCmdBindVertexBuffers(commandBuffer, 0, 1, worldVertexBuffers, worldOffsets);
            vkCmdBindIndexBuffer(commandBuffer, batch.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(commandBuffer, batch.indexCount, 1, 0, 0, 0);
            ++drawCallCount_;
        }
    }

    if (playerVertexCount_ > 0 && playerIndexCount_ > 0) {
        const VkBuffer playerVertexBuffers[] = {playerVertexBuffers_[currentFrame_]};
        const VkDeviceSize playerOffsets[] = {0};

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, worldPipeline_);
        vkCmdBindDescriptorSets(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            worldPipelineLayout_,
            0,
            1,
            &playerDescriptorSets_[currentFrame_],
            0,
            nullptr
        );
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, playerVertexBuffers, playerOffsets);
        vkCmdBindIndexBuffer(commandBuffer, playerIndexBuffer_, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(commandBuffer, playerIndexCount_, 1, 0, 0, 0);
        ++drawCallCount_;
    }

    if (selectionVertexCount_ > 0) {
        const VkBuffer selectionVertexBuffers[] = {selectionVertexBuffer_};
        const VkDeviceSize selectionOffsets[] = {0};

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, selectionPipeline_);
        vkCmdBindDescriptorSets(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            selectionPipelineLayout_,
            0,
            1,
            &descriptorSets_[currentFrame_],
            0,
            nullptr
        );
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, selectionVertexBuffers, selectionOffsets);
        vkCmdDraw(commandBuffer, selectionVertexCount_, 1, 0, 0);
        ++drawCallCount_;
    }

    if (entityColliderVertexCount_ > 0) {
        const VkBuffer entityColliderVertexBuffers[] = {entityColliderVertexBuffer_};
        const VkDeviceSize entityColliderOffsets[] = {0};

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, selectionPipeline_);
        vkCmdBindDescriptorSets(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            selectionPipelineLayout_,
            0,
            1,
            &descriptorSets_[currentFrame_],
            0,
            nullptr
        );
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, entityColliderVertexBuffers, entityColliderOffsets);
        vkCmdDraw(commandBuffer, entityColliderVertexCount_, 1, 0, 0);
        ++drawCallCount_;
    }

    if (overlayVertexCount_ > 0) {
        const VkBuffer overlayVertexBuffers[] = {overlayVertexBuffer_};
        const VkDeviceSize overlayOffsets[] = {0};

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, overlayPipeline_);
        vkCmdBindDescriptorSets(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            overlayPipelineLayout_,
            0,
            1,
            &overlayDescriptorSets_[currentFrame_],
            0,
            nullptr
        );
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, overlayVertexBuffers, overlayOffsets);
        vkCmdDraw(commandBuffer, overlayVertexCount_, 1, 0, 0);
        ++drawCallCount_;
    }

    vkCmdEndRenderPass(commandBuffer);

    RecordScreenshotCopyCommands(commandBuffer, imageIndex);

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to end command buffer.");
    }
}

void VulkanVoxelApp::CopyBuffer(VkBuffer sourceBuffer, VkBuffer destinationBuffer, VkDeviceSize size) const {
    const VkCommandBuffer commandBuffer = BeginSingleTimeCommands();

    VkBufferCopy copyRegion{};
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, sourceBuffer, destinationBuffer, 1, &copyRegion);

    EndSingleTimeCommands(commandBuffer);
}

void VulkanVoxelApp::TransitionImageLayout(
    VkImage image,
    VkFormat format,
    VkImageLayout oldLayout,
    VkImageLayout newLayout,
    std::uint32_t mipLevels
) const {
    const VkCommandBuffer commandBuffer = BeginSingleTimeCommands();

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = mipLevels;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    if (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        if (HasStencilComponent(format)) {
            barrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
        }
    } else {
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    }

    VkPipelineStageFlags sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    VkPipelineStageFlags destinationStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
        newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
               newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
               newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    } else {
        throw std::runtime_error("Unsupported image layout transition.");
    }

    vkCmdPipelineBarrier(
        commandBuffer,
        sourceStage,
        destinationStage,
        0,
        0,
        nullptr,
        0,
        nullptr,
        1,
        &barrier
    );

    EndSingleTimeCommands(commandBuffer);
}

void VulkanVoxelApp::CopyBufferToImage(
    VkBuffer buffer,
    VkImage image,
    std::uint32_t width,
    std::uint32_t height
) const {
    const VkCommandBuffer commandBuffer = BeginSingleTimeCommands();

    VkBufferImageCopy region{};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = {width, height, 1};

    vkCmdCopyBufferToImage(
        commandBuffer,
        buffer,
        image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &region
    );

    EndSingleTimeCommands(commandBuffer);
}

void VulkanVoxelApp::GenerateMipmaps(
    VkImage image,
    VkFormat imageFormat,
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t mipLevels
) const {
    VkFormatProperties formatProperties{};
    vkGetPhysicalDeviceFormatProperties(physicalDevice_, imageFormat, &formatProperties);
    if ((formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) == 0) {
        throw std::runtime_error("Texture format does not support linear blitting for mipmaps.");
    }

    const VkCommandBuffer commandBuffer = BeginSingleTimeCommands();

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.image = image;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.subresourceRange.levelCount = 1;

    std::int32_t mipWidth = static_cast<std::int32_t>(width);
    std::int32_t mipHeight = static_cast<std::int32_t>(height);

    for (std::uint32_t i = 1; i < mipLevels; ++i) {
        barrier.subresourceRange.baseMipLevel = i - 1;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

        vkCmdPipelineBarrier(
            commandBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0,
            0,
            nullptr,
            0,
            nullptr,
            1,
            &barrier
        );

        VkImageBlit blit{};
        blit.srcOffsets[0] = {0, 0, 0};
        blit.srcOffsets[1] = {mipWidth, mipHeight, 1};
        blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.srcSubresource.mipLevel = i - 1;
        blit.srcSubresource.baseArrayLayer = 0;
        blit.srcSubresource.layerCount = 1;
        blit.dstOffsets[0] = {0, 0, 0};
        blit.dstOffsets[1] = {
            mipWidth > 1 ? mipWidth / 2 : 1,
            mipHeight > 1 ? mipHeight / 2 : 1,
            1
        };
        blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.dstSubresource.mipLevel = i;
        blit.dstSubresource.baseArrayLayer = 0;
        blit.dstSubresource.layerCount = 1;

        vkCmdBlitImage(
            commandBuffer,
            image,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &blit,
            VK_FILTER_LINEAR
        );

        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(
            commandBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0,
            0,
            nullptr,
            0,
            nullptr,
            1,
            &barrier
        );

        if (mipWidth > 1) {
            mipWidth /= 2;
        }
        if (mipHeight > 1) {
            mipHeight /= 2;
        }
    }

    barrier.subresourceRange.baseMipLevel = mipLevels - 1;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0,
        0,
        nullptr,
        0,
        nullptr,
        1,
        &barrier
    );

    EndSingleTimeCommands(commandBuffer);
}

VkCommandBuffer VulkanVoxelApp::BeginSingleTimeCommands() const {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool_;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(device_, &allocInfo, &commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate single-use command buffer.");
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
        throw std::runtime_error("Failed to begin single-use command buffer.");
    }

    return commandBuffer;
}

void VulkanVoxelApp::EndSingleTimeCommands(VkCommandBuffer commandBuffer) const {
    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to end single-use command buffer.");
    }

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    if (vkQueueSubmit(graphicsQueue_, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
        throw std::runtime_error("Failed to submit single-use command buffer.");
    }

    vkQueueWaitIdle(graphicsQueue_);
    vkFreeCommandBuffers(device_, commandPool_, 1, &commandBuffer);
}
