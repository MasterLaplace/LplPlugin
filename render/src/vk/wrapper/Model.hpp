#pragma once
#ifndef LPL_RENDER_VK_MODEL_HPP
#define LPL_RENDER_VK_MODEL_HPP

#include <lpl/render/Mesh.hpp>
#include <vector>
#include <cstdint>

namespace lpl::render::vk {
    /**
     * @brief A structure holding a CPU mesh and its associated texture IDs.
     */
    struct Model {
        ::lpl::render::Mesh mesh;
        std::vector<uint32_t> textures;
    };
}

#endif // LPL_RENDER_VK_MODEL_HPP
