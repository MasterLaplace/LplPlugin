/**
 * @file Botany.hpp
 * @brief Trees grown from a grammar: a 3D turtle over an L-system string.
 *
 * @ref LSystem.hpp has had a branching grammar since the day it was written, and
 * for just as long it had no way to make a tree: its turtle walks a 2D grid and
 * paints cells, which draws a road network or a river delta seen from above but
 * cannot express a trunk. This is the missing half — the same rewrite rules, read
 * by a turtle that carries an orientation in space.
 *
 * The output is a SKELETON, not a mesh: tapered segments and leaf positions, in
 * world units, relative to the foot of the trunk. Turning that into triangles is
 * the renderer's business, and keeping the split means one grown tree can be
 * drawn a thousand times at a thousand transforms — a tree is data, an instance
 * is a transform.
 *
 * Fixed32 and CORDIC throughout, no libm: this module is linked into ring 0 and
 * a shape that differed between the host and the kernel would be a shape that
 * cannot be folded into a parity signature.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-28
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_PROCGEN_BOTANY_HPP
#    define LPL_PROCGEN_BOTANY_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/math/FixedPoint.hpp>
#    include <lpl/procgen/LSystem.hpp>
#    include <lpl/std/vector.hpp>

namespace lpl::procgen {

/**
 * @enum TreeSpecies
 * @brief Which grammar and proportions to grow with.
 *
 * Not a cosmetic label: the species picks the rewrite rules, so a conifer is
 * narrow because its grammar branches shallowly and often, not because a width
 * was multiplied at the end.
 */
enum class TreeSpecies : core::u8 {
    Conifer = 0, ///< A dominant stem with short side whorls: spruce, pine.
    Broadleaf,   ///< A stem that forks and forks again: oak, beech.
    Shrub,       ///< No dominant stem: many short stems from the base.
    Count
};

/**
 * @struct TreeParams
 * @brief Proportions of one grown tree.
 */
struct TreeParams {
    TreeSpecies species{TreeSpecies::Conifer};
    core::u32 seed{0x7A3Eu};                                 ///< Determinism anchor.
    core::u32 iterations{4u};                                 ///< Rewrite rounds.
    core::u32 maxSegments{192u};                              ///< Hard cap: a kernel heap is 4 MiB.
    math::Fixed32 segmentLength{math::Fixed32::fromFloat(0.9f)};  ///< Length of a depth-0 segment.
    math::Fixed32 lengthDecay{math::Fixed32::fromFloat(0.78f)};   ///< Length multiplier per branch depth.
    math::Fixed32 radius{math::Fixed32::fromFloat(0.13f)};        ///< Trunk radius at the foot.
    math::Fixed32 radiusDecay{math::Fixed32::fromFloat(0.68f)};   ///< Radius multiplier per branch depth.
    math::Fixed32 branchAngle{math::Fixed32::fromFloat(0.42f)};   ///< Pitch away from the parent, radians.
    math::Fixed32 rollAngle{math::Fixed32::fromFloat(2.39996f)};  ///< Roll between successive branches (golden angle).
    math::Fixed32 leafSize{math::Fixed32::fromFloat(0.26f)};      ///< Half-extent of a leaf sprite.
};

/**
 * @struct TreeBranch
 * @brief One tapered segment, from parent end to its own end.
 */
struct TreeBranch {
    math::Fixed32 x0{}, y0{}, z0{};
    math::Fixed32 x1{}, y1{}, z1{};
    math::Fixed32 radius0{}; ///< Radius at the start.
    math::Fixed32 radius1{}; ///< Radius at the end: a branch tapers.
    core::u8 depth{0u};      ///< Branch depth, 0 at the trunk.
};

/**
 * @struct TreeLeaf
 * @brief A cluster of foliage at a branch tip.
 */
struct TreeLeaf {
    math::Fixed32 x{}, y{}, z{};
    math::Fixed32 size{};
    core::u8 depth{0u};
};

/**
 * @struct TreeSkeleton
 * @brief What one grown tree is, before anything decides how to draw it.
 */
struct TreeSkeleton {
    lpl::pmr::vector<TreeBranch> branches;
    lpl::pmr::vector<TreeLeaf> leaves;
    math::Fixed32 height{};  ///< Highest point reached, for culling and for shadows.
    math::Fixed32 spread{};  ///< Largest horizontal distance from the trunk.
};

/** @brief The grammar for a species: rewrite rules and how far to run them. */
[[nodiscard]] LSystemParams makeTreeGrammar(TreeSpecies species, core::u32 seed);

/**
 * @brief Grows a tree by walking an expanded L-system string in three dimensions.
 *
 * The alphabet extends the 2D turtle's with the axes it was missing:
 *  - `F` draw a tapered segment forward and advance
 *  - `+` / `-` yaw right / left
 *  - `&` / `^` pitch down / up
 *  - `\` / `/` roll right / left
 *  - `[` / `]` push / pop, and each push is one branch deeper
 *  - `L` place a leaf cluster here
 *
 * A `]` that closes a branch with no `L` in it still gets foliage at its tip:
 * a grammar that forgot to say "leaf" would otherwise grow bare sticks, and the
 * tip of a branch is where leaves are.
 */
[[nodiscard]] TreeSkeleton growTree(const TreeParams &params);

/**
 * @brief FNV-1a over a skeleton's geometry, for the parity gate.
 *
 * Folds the raw Q16.16 of every endpoint and radius. A tree whose signature
 * differs between the host and the kernel is a tree grown by different
 * arithmetic, which is exactly what this module promises never to do.
 */
[[nodiscard]] core::u32 foldTreeSkeleton(const TreeSkeleton &skeleton);

/**
 * @brief The three species the parity gate grows, so both sides grow the same.
 *
 * One definition, read by the host oracle and by the in-kernel smoke alike —
 * the same rule the world recipe follows, for the same reason.
 */
[[nodiscard]] TreeParams parityTreeParams(TreeSpecies species);

} // namespace lpl::procgen

#endif // LPL_PROCGEN_BOTANY_HPP
