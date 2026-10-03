/**
 * @file Residency.hpp
 * @brief Choosing which bricks deserve to be in memory, before anybody goes and fetches them.
 *
 * @warning **Choosing is arithmetic and goes everywhere; fetching needs a filesystem or a socket
 * and stays with whoever has one.** That split is why this header allocates nothing, opens
 * nothing, and writes its answer into a buffer the caller owns.
 *
 * @warning **Requests come out nearest-first, and the order is the contract.** A host whose budget
 * is smaller than the plan truncates the tail, so the brick it loses must be the furthest one. Emit
 * the rings outward-in and a machine with a small budget loses the ground under its own feet
 * instead of the horizon -- which is the failure that looks like a bug in everything except the
 * planner.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_VOXEL_RESIDENCY_HPP
#    define LPL_VOXEL_RESIDENCY_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/voxel/Brick.hpp>
#    include <lpl/voxel/Volume.hpp>

namespace lpl::voxel {

/**
 * @struct ResidencyParams
 * @brief How much detail, how far out, and how much of it may be held.
 */
struct ResidencyParams final {
    core::u32 finestLevel{0u};   ///< Best level anybody may ask for.
    core::u32 coarsestLevel{5u}; ///< Level that covers the whole subject; always resident.
    core::i32 ringRadius{2};     ///< Bricks around the eye, per level, in each direction.
    core::u32 budget{256u};      ///< Hard cap on emitted requests.
};

/**
 * @brief Fills @p out with the bricks that should be resident for an eye at a level-0 sample.
 *
 * A pyramid rather than a window: every level contributes a small ring around the eye, so the
 * finest levels cover the near field and the coarsest one still covers the whole subject. That is
 * what makes the far field cost almost nothing -- a coarse brick spans 2^level times more of the
 * world for the same two mebibytes.
 *
 * @param geometry  The subject; bricks outside it are never requested.
 * @param params    Budget and reach.
 * @param eye       Eye position in level-0 samples, (z, y, x).
 * @param out       Destination.
 * @param capacity  Entries available in @p out.
 * @return Requests written, always at most min(@p capacity, params.budget).
 */
[[nodiscard]] core::u32 planResidency(const VolumeGeometry &geometry, const ResidencyParams &params,
                                      const core::i64 eye[3], BrickKey *out, core::u32 capacity) noexcept;

} // namespace lpl::voxel

#endif // LPL_VOXEL_RESIDENCY_HPP
