#ifndef LPL_PROCGEN_CAVE_WARREN_INL
#define LPL_PROCGEN_CAVE_WARREN_INL

namespace lpl::procgen {

constexpr core::f32 caveCoverThreshold(const CaveWarrenParams &warren, core::f32 floorY) noexcept
{
    const core::u32 perLayer = warren.levelsPerLayer == 0u ? 1u : warren.levelsPerLayer;
    const core::f32 gallery = static_cast<core::f32>(perLayer) * warren.levelHeight;
    const core::f32 cap = static_cast<core::f32>(warren.capLevels) * warren.levelHeight;
    return floorY + gallery + cap + warren.coverMargin;
}

inline bool CaveWarren::isAperture(core::i32 worldX, core::i32 worldZ) const noexcept
{
    for (core::u32 i = 0u; i < apertureCount; ++i)
        if (apertureX[i] == worldX && apertureZ[i] == worldZ)
            return true;
    return false;
}

inline bool CaveWarren::containsCell(core::i32 worldX, core::i32 worldZ) const noexcept
{
    if (volume.empty())
        return false;
    const core::i32 lx = worldX - originX;
    const core::i32 lz = worldZ - originZ;
    return lx >= 0 && lz >= 0 && static_cast<core::u32>(lx) < volume.width && static_cast<core::u32>(lz) < volume.depth;
}

inline bool CaveWarren::isCavernous(core::i32 worldX, core::i32 worldZ) const noexcept
{
    if (!containsCell(worldX, worldZ) || covered.empty())
        return false;
    return covered.at(static_cast<core::u32>(worldX - originX), static_cast<core::u32>(worldZ - originZ)) != 0u;
}

} // namespace lpl::procgen

#endif // LPL_PROCGEN_CAVE_WARREN_INL
