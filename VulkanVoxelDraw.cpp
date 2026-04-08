#include "VulkanVoxel.h"

#include <array>
#include <cstring>
#include <stdexcept>

namespace {

constexpr int kMaxFramesInFlight = 2;
constexpr float kPi = 3.14159265358979323846f;

}  // namespace

void VulkanVoxelApp::DrawFrame() {
    vkWaitForFences(device_, 1, &inFlightFences_[currentFrame_], VK_TRUE, UINT64_MAX);

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

    vkWaitForFences(
        device_,
        static_cast<std::uint32_t>(inFlightFences_.size()),
        inFlightFences_.data(),
        VK_TRUE,
        UINT64_MAX
    );
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

    currentFrame_ = (currentFrame_ + 1) % kMaxFramesInFlight;
}

void VulkanVoxelApp::UpdateUniformBuffer(std::uint32_t frameIndex) {
    UniformBufferObject ubo{};

    const float aspect = static_cast<float>(swapChainExtent_.width) /
                         static_cast<float>(swapChainExtent_.height);
    const Vec3 renderCameraPosition = GetRenderCameraPosition();
    const Vec3 renderCameraForward = GetRenderCameraForward();
    const Mat4 projection = Perspective(70.0f * kPi / 180.0f, aspect, 0.1f, 2048.0f);
    const Mat4 view = LookAt(
        renderCameraPosition,
        renderCameraPosition + renderCameraForward,
        {0.0f, 1.0f, 0.0f}
    );
    ubo.viewProj = Multiply(projection, view);

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
    clearValues[0].color.float32[0] = 0.53f;
    clearValues[0].color.float32[1] = 0.81f;
    clearValues[0].color.float32[2] = 0.92f;
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

    if (worldVertexCount_ > 0 && worldIndexCount_ > 0) {
        const VkBuffer worldVertexBuffers[] = {worldVertexBuffer_};
        const VkDeviceSize worldOffsets[] = {0};

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
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, worldVertexBuffers, worldOffsets);
        vkCmdBindIndexBuffer(commandBuffer, worldIndexBuffer_, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(commandBuffer, worldIndexCount_, 1, 0, 0, 0);
        ++drawCallCount_;
    }

    if (playerVertexCount_ > 0 && playerIndexCount_ > 0) {
        const VkBuffer playerVertexBuffers[] = {playerVertexBuffer_};
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
    VkImageLayout newLayout
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
    barrier.subresourceRange.levelCount = 1;
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
