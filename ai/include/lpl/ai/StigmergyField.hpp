/**
 * @file StigmergyField.hpp
 * @brief Knowledge stored in the world instead of in the agents.
 *
 * Two literatures arrive at the same structure from opposite directions.
 *
 * Ant colony optimisation says: an agent that has to compute a route is an agent
 * that cannot scale. Let it deposit a scalar on the ground instead, let that
 * scalar evaporate, and the colony's collective memory becomes a field — one that
 * *forgets*, which is what lets it abandon a route when the world changes under
 * it.
 *
 * Ecosystem simulation says: a predator that runs A* per tick is a predator you
 * can only afford ten of. Let prey leave a scent, let it diffuse around walls,
 * and a thousand predators navigate by reading one number from the cell they are
 * standing on.
 *
 * These are the same mechanism. Deposit, evaporate, diffuse, read the gradient.
 * They differ only in **what deposits and how much** — path-length-weighted for
 * the ants, presence for the scent — and that difference is a policy, not a data
 * structure. So there is one field here and two policies, because building two
 * fields would be building the same thing twice and then having to keep them
 * agreeing.
 *
 * Three properties are load-bearing:
 *
 * **Diffusion is blocked by walls.** This is what makes the field a navigator
 * rather than a heatmap: scent has to travel *around* an obstacle, so an agent
 * rolling up the gradient follows the path a body could actually take, and never
 * walks into a wall it cannot see past. It is also what makes the whole approach
 * cheaper than pathfinding rather than merely different.
 *
 * **The update is double-buffered.** A stencil computed in place reads cells that
 * the same sweep has already written, so the result depends on the traversal
 * order. That is not a rounding difference, it is a different answer — and on a
 * deterministic simulation it is a desynchronisation.
 *
 * **Evaporation has a floor.** In Q16.16 a value multiplied by 0.92 each tick
 * reaches zero in about 130 ticks, long before it is negligible in any meaningful
 * sense — a field that silently empties is a colony that forgets everything.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_AI_STIGMERGYFIELD_HPP
#    define LPL_AI_STIGMERGYFIELD_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/math/FixedPoint.hpp>
#    include <lpl/procgen/Grid.hpp>
#    include <lpl/std/vector.hpp>

namespace lpl::ai {

/// Channels a field may carry. Four covers the ecosystem's needs; more is cheap.
inline constexpr core::u32 kMaxStigmergyChannels = 8u;

/**
 * @enum ScentChannel
 * @brief The channels an ecosystem uses, named.
 *
 * Not baked into the field — a caller may use the indices for anything — but
 * naming them is what lets two systems agree about which number means "predator".
 */
enum class ScentChannel : core::u32 {
    Plant = 0, ///< Attracts herbivores.
    Herbivore, ///< Attracts carnivores.
    Carnivore, ///< Repels herbivores: this is what triggers flight.
    Terror,    ///< An apex predator's territory. Repels everything.
    Kin,       ///< Same-species presence. Mildly REPULSIVE, and that is the point.
    Pheromone  ///< Ant-colony trail strength.
};

/**
 * @struct StigmergyParams
 * @brief How fast the field forgets and how far it spreads.
 */
struct StigmergyParams {
    /**
     * @brief Retained share per tick, in (0, 1].
     *
     * @warning Evaporation and diffusion together give the field a **finite
     *          reach**, roughly @f$\sqrt{D / (1 - \lambda)}@f$ cells. At the
     *          defaults that is under two cells: a source is invisible three
     *          cells away, whatever the tick count, because the signal decays
     *          faster than it spreads. This is not a tuning nicety — it is why a
     *          field configured to forget quickly cannot also be a long-range
     *          navigator, and the two uses need different parameters. A trail
     *          meant to guide across a room wants @c evaporation near 0.999.
     */
    core::f32 evaporation{0.92f};
    core::f32 diffusion{0.08f}; ///< Share redistributed to the four neighbours.
    core::f32 maximum{100.0f};  ///< Saturation, so one source cannot dominate forever.

    /**
     * @brief Below this a cell is cleared to exactly zero.
     *
     * Not an optimisation. Without it, evaporation in fixed point walks a value
     * down to the last representable tick and leaves it there, so a trail that
     * should have gone stale keeps a faint permanent trace and the colony never
     * fully re-explores.
     */
    core::f32 floor{0.01f};
};

/**
 * @class StigmergyField
 * @brief A multi-channel scalar field with deposit, evaporation and diffusion.
 */
class StigmergyField {
public:
    StigmergyField() = default;

    /**
     * @brief Allocates @p channels grids of @p width x @p depth.
     * @param width    Cells along X.
     * @param depth    Cells along Z.
     * @param channels Channels; clamped to @ref kMaxStigmergyChannels.
     */
    StigmergyField(core::u32 width, core::u32 depth, core::u32 channels);

    [[nodiscard]] core::u32 width() const noexcept { return _width; }
    [[nodiscard]] core::u32 depth() const noexcept { return _depth; }
    [[nodiscard]] core::u32 channels() const noexcept { return _channels; }
    [[nodiscard]] bool empty() const noexcept { return _width == 0u || _depth == 0u || _channels == 0u; }

    /**
     * @brief Marks which cells block diffusion.
     *
     * The field is useless as a navigator without this: with no obstacles the
     * gradient points straight at the source, through whatever stands between.
     *
     * @param blocked Non-zero marks a cell diffusion may not cross.
     */
    void setObstacles(const procgen::Grid<core::u8> &blocked);

    /// @brief Zeroes every channel, keeping the obstacles.
    void clear();

    /**
     * @brief Adds to a cell, saturating at the configured maximum.
     * @param channel Channel index.
     * @param x       Column.
     * @param z       Row.
     * @param amount  Quantity to add.
     */
    void deposit(core::u32 channel, core::u32 x, core::u32 z, math::Fixed32 amount);

    /**
     * @brief Deposits along a path, weighted by its length.
     *
     * The ant-colony policy: a shorter route gets a stronger trail, so the field
     * converges on short paths without any agent comparing two of them. Deposit
     * is @f$Q / L@f$ — the classic form, and the reason the rule works is that it
     * is the *only* place path quality enters the system.
     *
     * @param channel Channel index.
     * @param cells   Path, as flat cell indices.
     * @param count   Entries in @p cells.
     * @param quality Numerator @f$Q@f$; the trail strength for a unit-length path.
     */
    void depositTrail(core::u32 channel, const core::u32 *cells, core::u32 count, math::Fixed32 quality);

    /**
     * @brief One tick: evaporate, then diffuse.
     *
     * In that order, and both through a second buffer. Evaporating after
     * diffusing would let a value spread and then decay, which loses the property
     * the two together are for: a source that stops emitting fades from its
     * furthest reach inward.
     *
     * @param params Rates and floor.
     */
    void step(const StigmergyParams &params);

    /**
     * @brief Reads a cell.
     * @param channel Channel index.
     * @param x       Column.
     * @param z       Row.
     * @return The value, or zero when out of range.
     */
    [[nodiscard]] math::Fixed32 value(core::u32 channel, core::u32 x, core::u32 z) const;

    /**
     * @brief The steepest uphill 8-neighbour, for an agent rolling up a gradient.
     *
     * Returns an index into @c procgen::kNeighbor8X / @c kNeighbor8Z, or
     * @ref kNoDirection when no neighbour is better. Obstacles are never
     * returned, so following this never walks into a wall.
     *
     * @param channel Channel index.
     * @param x       Column.
     * @param z       Row.
     * @param uphill  true to follow the scent, false to flee it.
     * @return Direction index, or @ref kNoDirection.
     */
    [[nodiscard]] core::u32 gradientDirection(core::u32 channel, core::u32 x, core::u32 z, bool uphill) const;

    /// @brief FNV-1a fold of every channel, for determinism checks.
    [[nodiscard]] core::u32 fold() const;

    /// Returned by @ref gradientDirection when standing still is the best move.
    static constexpr core::u32 kNoDirection = 0xFFFFFFFFu;

private:
    [[nodiscard]] core::u32 index(core::u32 channel, core::u32 x, core::u32 z) const noexcept
    {
        return channel * _width * _depth + z * _width + x;
    }

    core::u32 _width{0u};
    core::u32 _depth{0u};
    core::u32 _channels{0u};
    lpl::pmr::vector<math::Fixed32> _cells;
    lpl::pmr::vector<math::Fixed32> _scratch; ///< The second buffer; never read while writing.
    lpl::pmr::vector<core::u8> _blocked;
};

} // namespace lpl::ai

#endif // LPL_AI_STIGMERGYFIELD_HPP
