#ifndef LPL_ENGINE_GRID_TERRAIN_INL
#define LPL_ENGINE_GRID_TERRAIN_INL

namespace lpl::engine {

inline void GridTerrain::reset(const procgen::Grid<core::u8> *blocked, core::u32 width, core::u32 depth,
                               core::u32 regrowthTicks)
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

inline void GridTerrain::addPlant(core::i32 worldX, core::i32 worldZ)
{
    ecology::PlantCell plant;
    plant.cellX = worldX;
    plant.cellZ = worldZ;
    _plants.push_back(plant);
    ++_standing;
}

inline bool GridTerrain::standable(math::Fixed32 x, math::Fixed32 z) const
{
    if (_blocked == nullptr)
        return false;
    core::i32 gx = 0;
    core::i32 gz = 0;
    if (!toGrid(x, z, gx, gz))
        return false;
    return _blocked->at(static_cast<core::u32>(gx), static_cast<core::u32>(gz)) == 0u;
}

inline bool GridTerrain::consumePlantAt(core::i32 worldX, core::i32 worldZ)
{
    if (_plants.empty())
        return false;
    if (!ecology::grazeAt(&_plants[0], static_cast<core::u32>(_plants.size()), worldX, worldZ, 1, _regrowthTicks))
        return false;
    ++_grazed;
    --_standing;
    _floraDirty = true;
    return true;
}

inline bool GridTerrain::recoveryDirection(math::Fixed32 x, math::Fixed32 z, math::Fixed32 &outX,
                                           math::Fixed32 &outZ) const
{
    outX = x.raw() > 0 ? -math::Fixed32::one() : math::Fixed32::one();
    outZ = z.raw() > 0 ? -math::Fixed32::one() : math::Fixed32::one();
    return true;
}

inline core::u32 GridTerrain::tickVegetation()
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

inline bool GridTerrain::toGrid(math::Fixed32 x, math::Fixed32 z, core::i32 &outX, core::i32 &outZ) const
{
    outX = x.toInt() + static_cast<core::i32>(_width / 2u);
    outZ = z.toInt() + static_cast<core::i32>(_depth / 2u);
    return _blocked != nullptr && _blocked->contains(outX, outZ);
}

} // namespace lpl::engine

#endif // LPL_ENGINE_GRID_TERRAIN_INL
