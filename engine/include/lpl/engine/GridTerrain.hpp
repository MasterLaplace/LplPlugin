/**
 * @file GridTerrain.hpp
 * @brief A bounded obstacle grid, answered as the ground a walking animal asks.
 *
 * @ref ITerrainQuery names three questions and says nothing about who answers
 * them, which is correct: a streamed world answers them from noise and a resident
 * chunk, a bounded one from a mask. But *every tool* that wants to watch the
 * creature systems run needs the bounded answer, and it is always the same answer:
 * a walkability mask plus a plant list, on a grid centred on the origin.
 *
 * It was written once inside the map viewer and was about to be written a second
 * time inside the editor. The two would have differed within a week — the viewer's
 * copy of the *creature loop* had already drifted both ways from the engine's
 * before being folded back — so this is the same consolidation, applied one layer
 * down and before the second copy exists rather than after.
 *
 * What is NOT here, deliberately: how a world decides a cell is blocked.
 * @ref procgen::WorldSnapshot::blocked is one mask with one rule, and a tool adds
 * its own obstacles to it (raised buildings, scattered rocks) before binding it.
 * A second slope threshold in this class would be the third notion of "blocked",
 * which is how an animal ends up standing in a lake the scent flows around.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-08-04
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ENGINE_GRID_TERRAIN_HPP
#    define LPL_ENGINE_GRID_TERRAIN_HPP

#    include <lpl/ecology/Vegetation.hpp>
#    include <lpl/engine/ITerrainQuery.hpp>
#    include <lpl/procgen/Heightfield.hpp>
#    include <lpl/std/vector.hpp>

namespace lpl::engine {

/**
 * @class GridTerrain
 * @brief A bounded map, as the three questions of @ref ITerrainQuery.
 *
 * World coordinates are the grid centred on the origin, which is the one
 * conversion the creature systems cannot know and must not guess.
 *
 * The mask is held by POINTER, not copied: the tool that owns it goes on adding
 * obstacles to it (a plot the grammar raised, a boulder the scatter placed) after
 * binding, and a copy taken at bind time would answer for a world that no longer
 * exists.
 */
class GridTerrain final : public ITerrainQuery {
public:
    /**
     * @brief Binds the obstacle mask; plants are added afterwards, in prop order.
     * @param blocked       Non-zero means a body may not stand there. Must outlive this.
     * @param width         Mask width in cells.
     * @param depth         Mask depth in cells.
     * @param regrowthTicks Ticks a grazed plant takes to come back.
     */
    void reset(const procgen::Grid<core::u8> *blocked, core::u32 width, core::u32 depth, core::u32 regrowthTicks)
    {
        _blocked = blocked;
        _width = width;
        _depth = depth;
        _regrowthTicks = regrowthTicks;
        _plants.clear();
        _grazed = 0u;
        _regrown = 0u;
        _standing = 0u;
        _floraDirty = false;
    }

    /**
     * @brief One plant, at a world cell.
     *
     * Order matches the props, so a caller drawing flora can ask whether the i-th
     * prop is standing without keeping a second index.
     */
    void addPlant(core::i32 worldX, core::i32 worldZ)
    {
        ecology::PlantCell plant;
        plant.cellX = worldX;
        plant.cellZ = worldZ;
        _plants.push_back(plant);
        ++_standing;
    }

    [[nodiscard]] bool standable(math::Fixed32 x, math::Fixed32 z) const override
    {
        if (_blocked == nullptr)
            return false;
        core::i32 gx = 0;
        core::i32 gz = 0;
        if (!toGrid(x, z, gx, gz))
            return false;
        return _blocked->at(static_cast<core::u32>(gx), static_cast<core::u32>(gz)) == 0u;
    }

    bool consumePlantAt(core::i32 worldX, core::i32 worldZ) override
    {
        if (_plants.empty())
            return false;
        // ecology::grazeAt with a reach of one cell, which is the module's own
        // function and the module's own reason: measured with an exact match, sixty
        // grazers on a 128x128 map with 360 plants ate NOTHING over a minute — two
        // point sets that sparse almost never coincide, so the producer level sat
        // still while the herd walked over the trees.
        if (!ecology::grazeAt(&_plants[0], static_cast<core::u32>(_plants.size()), worldX, worldZ, 1, _regrowthTicks))
            return false;
        ++_grazed;
        --_standing;
        _floraDirty = true;
        return true;
    }

    /// Toward the middle: a bounded grid's ground is where its centre is.
    [[nodiscard]] bool recoveryDirection(math::Fixed32 x, math::Fixed32 z, math::Fixed32 &outX,
                                         math::Fixed32 &outZ) const override
    {
        outX = x.raw() > 0 ? -math::Fixed32::one() : math::Fixed32::one();
        outZ = z.raw() > 0 ? -math::Fixed32::one() : math::Fixed32::one();
        return true;
    }

    /// Regrowth. @return what is standing, which IS the producer population.
    core::u32 tickVegetation()
    {
        if (_plants.empty())
            return 0u;
        const core::u32 standing = ecology::tickPlants(&_plants[0], static_cast<core::u32>(_plants.size()));
        if (standing != _standing)
        {
            if (standing > _standing)
                _regrown += standing - _standing;
            _floraDirty = true;
            _standing = standing;
        }
        return standing;
    }

    [[nodiscard]] core::u32 standingPlants() const noexcept { return _standing; }
    [[nodiscard]] core::u32 plantCount() const noexcept { return static_cast<core::u32>(_plants.size()); }
    [[nodiscard]] core::u32 grazed() const noexcept { return _grazed; }
    [[nodiscard]] core::u32 regrown() const noexcept { return _regrown; }
    [[nodiscard]] bool plantStanding(core::usize index) const
    {
        return index < _plants.size() ? _plants[index].standing : true;
    }

    /// Whether the flora changed since the last @ref clearFloraDirty — a caller
    /// that rebuilds a mesh per frame regardless does not need to ask.
    [[nodiscard]] bool floraDirty() const noexcept { return _floraDirty; }
    void clearFloraDirty() noexcept { _floraDirty = false; }

private:
    /// World units to a cell of this grid, which is centred on the origin.
    [[nodiscard]] bool toGrid(math::Fixed32 x, math::Fixed32 z, core::i32 &outX, core::i32 &outZ) const
    {
        outX = x.toInt() + static_cast<core::i32>(_width / 2u);
        outZ = z.toInt() + static_cast<core::i32>(_depth / 2u);
        return _blocked != nullptr && _blocked->contains(outX, outZ);
    }

    const procgen::Grid<core::u8> *_blocked{nullptr};
    lpl::pmr::vector<ecology::PlantCell> _plants;
    core::u32 _width{0u};
    core::u32 _depth{0u};
    core::u32 _regrowthTicks{1200u};
    core::u32 _grazed{0u};
    core::u32 _regrown{0u};
    core::u32 _standing{0u};
    bool _floraDirty{false};
};

} // namespace lpl::engine

#endif // LPL_ENGINE_GRID_TERRAIN_HPP
