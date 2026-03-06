/**
 * @file Material.hpp
 * @brief Material parameters for the renderer.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-03-05
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_RENDER_MATERIAL_HPP
#    define LPL_RENDER_MATERIAL_HPP

#    include <lpl/math/Vec3.hpp>

namespace lpl::render {

/**
 * @struct Material
 * @brief Simple PBR material descriptor.
 */
struct Material {
    math::Vec3<float> albedo{1.0f, 1.0f, 1.0f};
    float roughness{0.5f};
    float metallic{0.0f};
};

} // namespace lpl::render

#endif // LPL_RENDER_MATERIAL_HPP
