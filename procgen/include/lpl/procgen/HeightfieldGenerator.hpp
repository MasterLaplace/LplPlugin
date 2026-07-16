/**
 * @file HeightfieldGenerator.hpp
 * @brief Populates an ECS world with a noise-driven grid of cube entities.
 *
 * A first `procgen` generator: sample fBm value noise over a cols×rows grid and
 * place one cube entity per cell at (x, height, z), height following the noise.
 * The whole world is authored in Fixed32 (deterministic), so re-running with the
 * same seed reproduces it bit-for-bit — and it can be serialized to `.lplscene`,
 * stepped by the physics backend, or rendered like the CubePile sample. This is
 * the miniature of the Caine vision: the AI (or a tool) picks the seed and the
 * high-level params, and the deterministic C++ engine materialises the world.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-16
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_PROCGEN_HEIGHTFIELDGENERATOR_HPP
#    define LPL_PROCGEN_HEIGHTFIELDGENERATOR_HPP

#    include <lpl/core/Types.hpp>

namespace lpl::ecs {
class Registry;
}

namespace lpl::procgen {

/**
 * @struct HeightfieldParams
 * @brief High-level knobs an editor / AI would set (kept small — token-cheap).
 */
struct HeightfieldParams {
    core::u32 seed{1337u};       ///< World seed (determinism anchor).
    core::u32 cols{32u};         ///< Grid columns (X).
    core::u32 rows{32u};         ///< Grid rows (Z).
    core::f32 spacing{0.5f};     ///< World units between cells.
    core::f32 noiseScale{0.15f}; ///< Sampling frequency (smaller = smoother).
    core::f32 amplitude{4.0f};   ///< Vertical relief scale.
    core::u32 octaves{4u};       ///< fBm octaves.
    core::f32 cubeHalf{0.2f};    ///< Cube half-extent (AABB).
};

/**
 * @brief Generates a heightfield of cube entities into @p registry.
 * @param registry Destination world (Position + AABB per entity, Fixed32).
 * @param params   Generation parameters.
 * @return Number of entities created (cols × rows).
 */
core::u32 generateHeightfield(ecs::Registry &registry, const HeightfieldParams &params);

} // namespace lpl::procgen

#endif // LPL_PROCGEN_HEIGHTFIELDGENERATOR_HPP
