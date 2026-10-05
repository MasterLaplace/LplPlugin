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
 * @file Surface.hpp
 * @brief Surface class declaration.
 *
 * The surface class is used to create a surface for the Vulkan API.
 * WSI (Window System Integration) is used to render graphics to a window or other surface.
 *
 * @author @MasterLaplace
 * @version 0.0.4
 * @date 2024-10-22
 **************************************************************************/

#ifndef LPL_RENDER_VK_SURFACE_HPP_
#define LPL_RENDER_VK_SURFACE_HPP_

#include "queueFamilies/QueueFamilies.hpp"

namespace lpl::render::vk {

/**
 * @brief Surface class.
 *
 * This class is used to create a surface for the Vulkan API. It is used to
 * render graphics to a window or other surface. The surface is created from
 * a GLFW window and the Vulkan instance.
 *
 * @example
 * @code
 * Surface surface;
 * surface.Create(window, instance);
 * surface.Destroy(instance);
 * @endcode
 */
class Surface {
public:
    /**
     * @brief Creates a surface for the Vulkan API.
     *
     * This function creates a surface for the Vulkan API using a GLFW window
     * and the Vulkan instance. The surface is used to render graphics to the
     * window or other surface.
     *
     * @param window  The GLFW window.
     * @param instance  The Vulkan instance.
     * @param allocator  The Vulkan allocation callbacks.
     */
    void Create(GLFWwindow *window, const VkInstance &instance, VkAllocationCallbacks *allocator);

    /**
     * @brief Destroys the surface.
     *
     * This function destroys the surface created for the Vulkan API.
     *
     * @param instance  The Vulkan instance.
     */
    void Destroy(const VkInstance &instance);

    /**
     * @brief Returns the surface.
     *
     * This function returns the surface created for the Vulkan API.
     *
     * @return The surface.
     */
    [[nodiscard]] const VkSurfaceKHR &Get() const { return _surface; }

private:
    VkSurfaceKHR _surface;
};

} // namespace lpl::render::vk

#endif /* !SURFACE_HPP_ */
