#ifndef LPL_ECOLOGY_VEGETATION_INL
#define LPL_ECOLOGY_VEGETATION_INL

namespace lpl::ecology {

inline core::u32 tickPlants(PlantCell *plants, core::u32 count) noexcept
{
    core::u32 standing = 0u;
    for (core::u32 i = 0u; i < count; ++i)
    {
        PlantCell &plant = plants[i];
        if (!plant.standing && plant.regrowth != 0u && --plant.regrowth == 0u)
            plant.standing = true;
        standing += plant.standing ? 1u : 0u;
    }
    return standing;
}

inline core::u32 countStanding(const PlantCell *plants, core::u32 count) noexcept
{
    core::u32 standing = 0u;
    for (core::u32 i = 0u; i < count; ++i)
        standing += plants[i].standing ? 1u : 0u;
    return standing;
}

inline bool grazeAt(PlantCell *plants, core::u32 count, core::i32 worldX, core::i32 worldZ, core::i32 radius,
                    core::u32 regrowthTicks) noexcept
{
    for (core::u32 i = 0u; i < count; ++i)
    {
        if (!plants[i].standing)
            continue;
        const core::i32 dx = plants[i].cellX - worldX;
        const core::i32 dz = plants[i].cellZ - worldZ;
        if (dx > radius || dx < -radius || dz > radius || dz < -radius)
            continue;
        plants[i].standing = false;
        plants[i].regrowth = regrowthTicks;
        return true;
    }
    return false;
}

} // namespace lpl::ecology

#endif // LPL_ECOLOGY_VEGETATION_INL
