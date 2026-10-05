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
 * @file UniformObject.hpp
 * @brief UniformObject class declaration.
 *
 *
 * @author @MasterLaplace
 * @version 0.0.4
 * @date 2024-11-03
 **************************************************************************/

#ifndef LPL_RENDER_VK_UNIFORMOBJECT_HPP_
#define LPL_RENDER_VK_UNIFORMOBJECT_HPP_

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace lpl::render::vk {

/**
 * @brief UniformObject class.
 *
 * This class is used to represent a uniform object in the Vulkan API.
 * It contains the model, view, and projection matrices. The matrices are used
 * to transform the vertices in the vertex shader.
 */
struct UniformBufferObject {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
};

} // namespace lpl::render::vk

#endif /* !UNIFORMOBJECT_HPP_ */
