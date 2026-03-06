/**
 * @file Model.hpp
 * @brief Internal Vulkan model descriptor (mesh + texture bindings).
 *
 * A @c Model bundles a CPU-side @c Mesh together with the list of texture IDs
 * that are bound to it before the graphics pipeline is created.  It is stored
 * inside @c Wrapper's model map and consumed by @c Buffers::Create() when
 * uploading vertex/index data to the GPU.
 *
 * @note This file is internal to the @c lpl-render module and is NOT part of
 *       the public API.  Consumers should use @c VulkanRenderer and @c IRenderer
 *       instead.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */
#pragma once
#ifndef LPL_RENDER_VK_MODEL_HPP
#    define LPL_RENDER_VK_MODEL_HPP

#    include <cstdint>
#    include <lpl/render/Mesh.hpp>
#    include <vector>

namespace lpl::render::vk {
/**
 * @brief A structure holding a CPU mesh and its associated texture IDs.
 */
struct Model {
    ::lpl::render::Mesh mesh;
    std::vector<uint32_t> textures;
};
} // namespace lpl::render::vk

#endif // LPL_RENDER_VK_MODEL_HPP
