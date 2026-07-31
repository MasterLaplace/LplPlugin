/**
 * @file Vegetation.hpp
 * @brief Standing plants: eaten one at a time, regrowing on a timer, counted.
 *
 * The producer level of a food web, as data rather than as a number. A population
 * model can integrate "there is this much grass"; a world where a herd walks needs
 * to know WHICH grass, because grazing a valley bare has to move the number, and it
 * only does if the number is a count of things that are gone.
 *
 * Two rules are encoded here because both were arrived at from watching a herd:
 *
 *  - a grazer eats ONE plant per tick and returns. Eating every plant in reach turns
 *    a herd into a lawnmower that clears a chunk in a second, and the population
 *    model then reports an extinction that nothing on screen explains.
 *  - regrowth is a countdown per plant, not a global refill. A refill makes the whole
 *    map recover at once, which reads as a graphical glitch rather than as growth.
 *
 * Free functions over a span, because where the plants live is the world's business:
 * one list for a bounded map, one list per chunk for a streamed one.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-28
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ECOLOGY_VEGETATION_HPP
#    define LPL_ECOLOGY_VEGETATION_HPP

#    include <lpl/core/Types.hpp>

namespace lpl::ecology {

/**
 * @struct PlantCell
 * @brief One plant: where it stands, whether it does, and when it comes back.
 *
 * Coordinates are SIGNED world cells. A streamed world has no corner to count from,
 * so a plant cannot be identified by an index into a grid.
 */
struct PlantCell {
    core::i32 cellX{0};
    core::i32 cellZ{0};
    core::u32 regrowth{0u}; ///< Ticks until it stands again; 0 when standing.
    bool standing{true};
};

/**
 * @brief Advances regrowth and counts what is standing.
 * @return Plants currently standing, which IS the producer population.
 */
core::u32 tickPlants(PlantCell *plants, core::u32 count) noexcept;

/** @brief Plants standing in a span, without advancing anything. */
[[nodiscard]] core::u32 countStanding(const PlantCell *plants, core::u32 count) noexcept;

/**
 * @brief Eats the first standing plant within @p radius cells of a position.
 * @return True when something was eaten, so a caller can count it.
 */
bool grazeAt(PlantCell *plants, core::u32 count, core::i32 worldX, core::i32 worldZ, core::i32 radius,
             core::u32 regrowthTicks) noexcept;

} // namespace lpl::ecology

// Out-of-line definitions: consumed header-only, the freestanding kernel included.
#    include <lpl/ecology/Vegetation.inl>

#endif // LPL_ECOLOGY_VEGETATION_HPP
