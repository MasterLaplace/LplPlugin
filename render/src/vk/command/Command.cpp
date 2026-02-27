#include "command/Command.hpp"
#include <lpl/core/Log.hpp>
#include <cstdlib>

namespace lpl::render::vk {

void Command::Create(const VkDevice &device, const VkPhysicalDevice &physicalDevice, const VkSurfaceKHR &surface)
{
    _queueFamilies.FindQueueFamilies(physicalDevice, surface);

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = _queueFamilies.GetIndices().graphicsFamily.value();

    if (vkCreateCommandPool(device, &poolInfo, nullptr, &_commandPool) != VK_SUCCESS)
        { ::lpl::core::Log::fatal("failed to create command pool!"); std::abort(); }
}

void Command::CreateCommandBuffers(const VkDevice &device,
                                   [[maybe_unused]] const std::vector<VkFramebuffer> &swapChainFramebuffers)
{
    _commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = _commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = static_cast<uint32_t>(_commandBuffers.size());

    if (vkAllocateCommandBuffers(device, &allocInfo, _commandBuffers.data()) != VK_SUCCESS)
        { ::lpl::core::Log::fatal("failed to allocate command buffers!"); std::abort(); }
}

void Command::Destroy(const VkDevice &device)
{
    vkFreeCommandBuffers(device, _commandPool, static_cast<uint32_t>(_commandBuffers.size()), _commandBuffers.data());
    vkDestroyCommandPool(device, _commandPool, nullptr);
}

void Command::RecordBuffer(const RecordInfo &info)
{
    auto commandBuffer = _commandBuffers[info.currentFrame];

    vkResetCommandBuffer(commandBuffer, 0);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS)
        { ::lpl::core::Log::fatal("failed to begin recording command buffer!"); std::abort(); }

    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = info.renderPass;
    renderPassInfo.framebuffer = info.swapChainFramebuffers[info.imageIndex];
    renderPassInfo.renderArea.offset = {0, 0};
    renderPassInfo.renderArea.extent = info.swapChainExtent;

    std::array<VkClearValue, 2> clearValues{};
    clearValues[0].color = {_clearColor.r, _clearColor.g, _clearColor.b, _clearColor.a};
    clearValues[1].depthStencil = {1.0f, 0};
    renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
    renderPassInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

    if (info.graphicsPipeline != VK_NULL_HANDLE)
    {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, info.graphicsPipeline);

        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = (float) info.swapChainExtent.width;
        viewport.height = (float) info.swapChainExtent.height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.offset = {0, 0};
        scissor.extent = info.swapChainExtent;
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        VkDeviceSize offsets{};
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, &info.vertexBuffer, &offsets);

        if (info.indexBuffer != VK_NULL_HANDLE)
            vkCmdBindIndexBuffer(commandBuffer, info.indexBuffer, 0, VK_INDEX_TYPE_UINT32);

        if (info.pipelineLayout != VK_NULL_HANDLE)
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, info.pipelineLayout, 0, 1,
                                    &info.descriptorSet, 0, nullptr);

        if (info.indexCount > 0)
            vkCmdDrawIndexed(commandBuffer, info.indexCount, 1, 0, 0, 0);
    }

    if (_isGuiEnabled)
        GUI::GetInstance().Render(_clearColor, commandBuffer);

    vkCmdEndRenderPass(commandBuffer);

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
        { ::lpl::core::Log::fatal("failed to record command buffer!"); std::abort(); }
}

} // namespace lpl::render::vk
