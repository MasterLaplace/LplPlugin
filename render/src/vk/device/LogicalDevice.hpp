/**************************************************************************
 * VkWrapper v0.0.4
 *
 * VkWrapper is a software package, part of LplPlugin.
 *
 * This file is part of the VkWrapper project that is under the MIT License.
 * https://opensource.org/license/mit
 * Copyright © 2024 by @MasterLaplace, All rights reserved.
 *
 * VkWrapper is free software: you can use, copy, modify, merge, publish and
 * distribute it under the terms of the MIT License, provided this copyright
 * notice and the permission notice are kept. See the LICENSE file.
 *
 * @file LogicalDevice.hpp
 * @brief LogicalDevice class declaration.
 *
 *
 * @author @MasterLaplace
 * @version 0.0.4
 * @date 2024-10-22
 **************************************************************************/

#ifndef LPL_RENDER_VK_LOGICALDEVICE_HPP_
#define LPL_RENDER_VK_LOGICALDEVICE_HPP_

#include "queueFamilies/QueueFamilies.hpp"

#include <set>

namespace lpl::render::vk {

/**
 * @brief LogicalDevice class.
 *
 * @example
 * @code
 * LogicalDevice device;
 * device.Create(physicalDevice);
 * device.Destroy();
 * @endcode
 */
class LogicalDevice {
public:
    /**
     * @brief Creates a logical device from the selected physical device.
     *
     * This function creates a logical device, which is an abstraction
     * representing the GPU. It enables communication with the physical
     * device and allows the application to execute Vulkan commands.
     * The logical device is configured with specific features and
     * extensions required by the application.
     *
     * @param physicalDevice  The selected physical device.
     * @param surface  The Vulkan surface.
     */
    void Create(const VkPhysicalDevice &physicalDevice, const VkSurfaceKHR &surface);

    /**
     * @brief Destroys the logical device.
     *
     * This function destroys the logical device.
     */
    void Destroy() { vkDestroyDevice(_device, nullptr); }

    /**
     * @brief Gets the logical device.
     *
     * This function returns the logical device.
     *
     * @return The logical device.
     */
    [[nodiscard]] const VkDevice &Get() const { return _device; }

    /**
     * @brief Gets the present queue.
     *
     * This function returns the present queue.
     *
     * @return The present queue.
     */
    [[nodiscard]] const VkQueue &GetPresentQueue() { return _presentQueue; }

    /**
     * @brief Gets the graphics queue.
     *
     * This function returns the graphics queue.
     *
     * @return The graphics queue.
     */
    [[nodiscard]] const VkQueue &GetGraphicsQueue() { return _graphicsQueue; }

private:
    VkDevice _device = VK_NULL_HANDLE;
    QueueFamilies _queueFamilies;
    VkQueue _graphicsQueue;
    VkQueue _presentQueue;
};

} // namespace lpl::render::vk

#endif /* !LOGICALDEVICE_HPP_ */
