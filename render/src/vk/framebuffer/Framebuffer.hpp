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
 * @file Framebuffer.hpp
 * @brief Framebuffer class declaration.
 *
 *
 * @author @MasterLaplace
 * @version 0.0.4
 * @date 2024-10-24
 **************************************************************************/

#ifndef LPL_RENDER_VK_FRAMEBUFFER_HPP_
    #define LPL_RENDER_VK_FRAMEBUFFER_HPP_

#include "debugMessenger/DebugMessenger.hpp"

namespace lpl::render::vk {

/**
 * @brief Framebuffer class.
 *
 * @example
 * @code
 * Framebuffer framebuffer;
 * framebuffer.Create(device, info);
 * framebuffer.Destroy(device);
 * @endcode
 */
class Framebuffer {
  public:
    struct CreateInfo {
        VkExtent2D swapChainExtent;
        VkRenderPass renderPass;
        std::vector<VkImageView> swapChainImageViews;
        VkImageView depthImageView;
        VkImageView colorImageView;
    };

  public:
    /**
     * @brief Creates a framebuffer.
     *
     * This function creates a framebuffer from the device and the create info.
     *
     * @param device  The Vulkan device.
     * @param info  The create info.
     */
    void Create(const VkDevice &device, const CreateInfo &info);

    /**
     * @brief Destroys the framebuffer.
     *
     * This function destroys the framebuffer.
     *
     * @param device  The Vulkan device.
     */
    void Destroy(const VkDevice &device) const;

    /**
     * @brief Gets the swap chain framebuffers.
     *
     * This function returns the swap chain framebuffers.
     *
     * @return The swap chain framebuffers.
     */
    [[nodiscard]] const std::vector<VkFramebuffer> &GetSwapChainFramebuffers() const { return _swapChainFramebuffers; }

  private:
    std::vector<VkFramebuffer> _swapChainFramebuffers;
};

} // namespace lpl::render::vk

#endif /* !FRAMEBUFFER_HPP_ */
