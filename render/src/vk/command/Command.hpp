/**************************************************************************
 * VkWrapper v0.0.4
 *
 * VkWrapper is a software package, part of the Engine².
 *
 * This file is part of the VkWrapper project that is under GPL-3.0 License.
 * Copyright © 2024 by @EngineSquared, All rights reserved.
 *
 * VkWrapper is a free software: you can redistribute it and/or modify
 * it under the terms of the GPL-3.0 License as published by the
 * Free Software Foundation. See the GPL-3.0 License for more details.
 *
 * @file Command.hpp
 * @brief Command class declaration.
 *
 *
 * @author @MasterLaplace
 * @version 0.0.4
 * @date 2024-10-24
 **************************************************************************/

#ifndef LPL_RENDER_VK_COMMAND_HPP_
    #define LPL_RENDER_VK_COMMAND_HPP_

#include "gui/GUI.hpp"

namespace lpl::render::vk {

/**
 * @brief Command class.
 *
 * @example
 * @code
 * Command command;
 * Command::CreateInfo info = {
 *    .physicalDevice = physicalDevice.Get(),
 *    .surface = surface.Get(),
 *    .swapChainExtent = swapChain.GetExtent(),
 *    .renderPass = renderPass.Get(),
 *    .swapChainFramebuffers = framebuffer.GetSwapChainFramebuffers(),
 *    .graphicsPipeline = graphicsPipeline.Get(),
 * };
 * command.Create(device, info);
 * command.Destroy(device);
 * @endcode
 */
class Command {
  public:
    /**
     * @brief RecordInfo struct.
     *
     * This struct holds the information required to record a command buffer.
     *
     * @var uint32_t RecordInfo::currentFrame  The current frame.
     * @var uint32_t RecordInfo::imageIndex  The image index.
     * @var VkRenderPass RecordInfo::renderPass  The render pass.
     * @var VkExtent2D RecordInfo::swapChainExtent  The swap chain extent.
     * @var std::vector<VkFramebuffer> RecordInfo::swapChainFramebuffers  The swap chain framebuffers.
     * @var VkPipeline RecordInfo::graphicsPipeline  The graphics pipeline.
     * @var VkPipelineLayout RecordInfo::pipelineLayout  The pipeline layout.
     * @var VkDescriptorSet RecordInfo::descriptorSet  The descriptor set.
     * @var VkBuffer RecordInfo::vertexBuffer  The vertex buffer.
     * @var VkBuffer RecordInfo::indexBuffer  The index buffer.
     */
    struct RecordInfo {
        uint32_t currentFrame;
        uint32_t imageIndex;
        VkRenderPass renderPass;
        VkExtent2D swapChainExtent;
        std::vector<VkFramebuffer> swapChainFramebuffers;
        VkPipeline graphicsPipeline;
        VkPipelineLayout pipelineLayout;
        VkDescriptorSet descriptorSet;
        VkBuffer vertexBuffer;
        VkBuffer indexBuffer;
        uint32_t vertexCount;
        uint32_t indexCount;
    };

  public:
    /**
     * @brief Creates a command pool and command buffers.
     *
     * This function creates a command pool from the device and the queue families.
     * It also creates command buffers using the command pool.
     *
     * @param device  The Vulkan device.
     * @param queueFamilies  The queue families.
     */
    void Create(const VkDevice &device, const VkPhysicalDevice &physicalDevice, const VkSurfaceKHR &surface);

    /**
     * @brief Creates command buffers separately.
     *
     * This function creates command buffers using the command pool.
     * It creates a command buffer for each image in the swap chain.
     *
     * @param device  The Vulkan device.
     * @param swapChainFramebuffers  The swap chain framebuffers.
     */
    void CreateCommandBuffers(const VkDevice &device, const std::vector<VkFramebuffer> &swapChainFramebuffers);

    /**
     * @brief Destroys the command pool.
     *
     * This function destroys the command pool.
     *
     * @param device  The Vulkan device.
     */
    void Destroy(const VkDevice &device);

    /**
     * @brief Records a command buffer.
     *
     * This function records a command buffer from the record info.
     *
     * @param info  The record info.
     */
    void RecordBuffer(const RecordInfo &info);

    /**
     * @brief Gets the command pool.
     *
     * This function returns the command pool.
     *
     * @return The command pool.
     */
    [[nodiscard]] const VkCommandPool &GetCommandPool() { return _commandPool; }

    /**
     * @brief Gets the command buffer.
     *
     * This function returns the command buffer.
     *
     * @param imageIndex  The image index.
     * @return The command buffer.
     */
    [[nodiscard]] const VkCommandBuffer &GetCommandBuffer(const uint32_t imageIndex)
    {
        return _commandBuffers[imageIndex];
    }

    /**
     * @brief Sets the clear color.
     *
     * This function sets the clear color.
     *
     * @param color  The clear color.
     */
    inline void SetClearColor(const glm::vec4 &color) { _clearColor = color; }

    /**
     * @brief Sets the GUI.
     *
     * This function sets the GUI.
     *
     * @param isGui  The GUI.
     */
    inline void SetGui(bool isGui) { _isGuiEnabled = isGui; }

  private:
    VkCommandPool _commandPool;
    QueueFamilies _queueFamilies;
    std::vector<VkCommandBuffer> _commandBuffers;
    glm::vec4 _clearColor = {0.0f, 0.0f, 0.0f, 1.0f};
    bool _isGuiEnabled = false;
};

} // namespace lpl::render::vk

#endif /* !COMMAND_HPP_ */
