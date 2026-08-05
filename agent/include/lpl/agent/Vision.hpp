/**
 * @file Vision.hpp
 * @brief The vision-language-action seam.
 *
 * Move a camera, render, hand back a frame the model can look at. Looking has
 * already found four deterministic bugs that no fold caught; this is that method,
 * made repeatable without a human operator.
 *
 * Assembly, not invention. Every piece existed: @c render::OrbitCamera for the
 * pose, @c render::fillPolygonClipped for the geometry, @c render::foldTarget for
 * the signature, @c image::writePpm for the bytes. What was missing was a path
 * from an @c ecs::Registry to a file, and that is all this adds.
 *
 * A note on what the fold means. @ref Screenshot::fold is @c render::foldTarget —
 * NOT a second implementation — and it is a PERCEPTUAL signature: the render path
 * is float, hence non-authoritative by the determinism contract. It answers "did
 * the picture change", which is exactly what a correction loop needs. It does not
 * answer "do two machines agree", and no gate should ever ask it to.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_AGENT_VISION_HPP
#    define LPL_LPL_AGENT_VISION_HPP

#    include <lpl/core/Expected.hpp>
#    include <lpl/core/Types.hpp>
#    include <lpl/image/Image.hpp>

#    include <string>
#    include <string_view>

namespace lpl::ecs {
class Registry;
}

namespace lpl::agent {

/**
 * @struct CameraPose
 * @brief Where to look from.
 *
 * Angles in degrees because this is the surface a model writes to, and a model
 * that has to reason in radians spends tokens on trigonometry instead of on the
 * world. The conversion happens once, here.
 */
struct CameraPose {
    core::f32 yawDeg{45.0f};
    /// Chosen by looking, not by taste: at 30 degrees a wide terrain converges to
    /// a point and reads as a spike; at 65 the relief flattens out and a ridge is
    /// indistinguishable from a plain. Around 45 both survive.
    core::f32 pitchDeg{45.0f};
    /// Orbit radius; zero means "frame whatever is there", computed from the bounds.
    core::f32 distance{0.0f};
};

/**
 * @struct Screenshot
 * @brief What a capture produced, besides the pixels.
 */
struct Screenshot {
    core::u32 width{0u};
    core::u32 height{0u};
    core::u32 fold{0u};          ///< render::foldTarget over the colour buffer.
    core::u32 entitiesDrawn{0u}; ///< Entities that had a position to draw at.
    core::u32 triangles{0u};
};

/**
 * @brief Renders every positioned entity of @p registry into @p out.
 *
 * Entities are drawn as boxes sized by their @c AABB half-extents, or by a unit
 * default when they carry none — a world of invisible points would defeat the
 * purpose. Colour is derived from the archetype so that different KINDS of thing
 * read differently; it is cosmetic and never flows back into world state.
 */
[[nodiscard]] Screenshot renderWorld(const ecs::Registry &registry, core::u32 width, core::u32 height,
                                     const CameraPose &pose, image::Image &out);

/**
 * @brief @copybrief renderWorld — and writes it to @p path as a binary PPM.
 *
 * PPM because it is the one format this project already encodes and decodes, on
 * both the host and the kernel, with a byte layout that has no compression to
 * disagree about.
 */
[[nodiscard]] core::Expected<Screenshot> captureToFile(const ecs::Registry &registry, std::string_view path,
                                                       core::u32 width, core::u32 height, const CameraPose &pose);

} // namespace lpl::agent

#endif // LPL_LPL_AGENT_VISION_HPP
