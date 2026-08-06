/**
 * @file VerticalSpan.hpp
 * @brief What a column of the world offers a body: a floor, and a ceiling.
 *
 * Its own header, and a deliberately tiny one, because of who has to include it.
 * @ref lpl::engine::CharacterController is written to know nothing about terrain —
 * that is the sentence at the top of its file and the reason it can be tested against
 * a flat plane — but the question it asks the world had to grow a second half the day
 * the world grew a roof. Pulling @ref CaveWarren.hpp in to get one three-field struct
 * would have dragged the cave generator, the landmark lattice and the chunk scheme
 * into a header that needs none of them, and the honest way to avoid that is to put
 * the concept where it is cheap rather than to duplicate it.
 *
 * It lives in procgen because it is a statement about the SHAPE OF THE WORLD, and
 * that is what procgen is. Nothing here generates anything.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-08-06
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_PROCGEN_VERTICAL_SPAN_HPP
#    define LPL_PROCGEN_VERTICAL_SPAN_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/math/FixedPoint.hpp>

namespace lpl::procgen {

/**
 * @brief A height nothing in this world stands above.
 *
 * Q16.16 saturates a little past 32768, and terrain here is tens of metres, so 4096
 * is "no ceiling" with two orders of magnitude of headroom and no risk of overflow in
 * a caller that subtracts a body height from it. A saturating maximum would have been
 * the tempting choice and is the wrong one for exactly that reason.
 */
[[nodiscard]] constexpr math::Fixed32 openSky() noexcept { return math::Fixed32{4096 << 16}; }

/**
 * @struct VerticalSpan
 * @brief The open gap a body occupies at one column: what is under it, what is over it.
 *
 * A heightfield answers "how high is the ground", and that is a complete answer only
 * while the world has one surface per column. The moment there is a gallery under a
 * hill there are two, and a body is in one of them — so the question a collider has to
 * ask is not how high the ground is but WHICH gap it is standing in.
 *
 * Authoritative, therefore Fixed32: this decides where a body may be.
 */
struct VerticalSpan {
    math::Fixed32 floor{};            ///< Top of the solid under the body.
    math::Fixed32 ceiling{openSky()}; ///< Underside of the solid over it.
    bool enclosed{false};             ///< Whether there is rock overhead at all.

    /// @return How much room there is between the two.
    [[nodiscard]] constexpr math::Fixed32 headroom() const noexcept { return ceiling - floor; }
};

/**
 * @brief The span of a world that is only a surface: this ground, and open sky.
 *
 * The adaptor that lets a heightfield — a bounded grid, a test's flat plane, a
 * streamed field with no caves in it — answer the richer question without every one
 * of them having to know the richer question exists.
 *
 * @param ground Height of the ground at the column.
 * @return A span from that ground to the sky.
 */
[[nodiscard]] constexpr VerticalSpan surfaceSpan(math::Fixed32 ground) noexcept
{
    return VerticalSpan{ground, openSky(), false};
}

} // namespace lpl::procgen

#endif // LPL_PROCGEN_VERTICAL_SPAN_HPP
