#ifndef LPL_VOXEL_BRICK_INL
#define LPL_VOXEL_BRICK_INL

namespace lpl::voxel {

constexpr core::u8 BrickView::cellHighest(core::u32 lz, core::u32 ly, core::u32 lx) const noexcept
{
    const core::u32 cell =
        ((lz >> kOccupancyShift) * kOccupancyEdge + (ly >> kOccupancyShift)) * kOccupancyEdge + (lx >> kOccupancyShift);
    return occupancy[cell * 2u + 1u];
}

constexpr core::i64 brickOriginInBaseSamples(const BrickKey &key, core::u32 axis) noexcept
{
    const core::i32 index = axis == 0u ? key.z : (axis == 1u ? key.y : key.x);
    return static_cast<core::i64>(index) * brickSpanInBaseSamples(key.level);
}

template <typename Real> constexpr core::i64 floorToI64(Real position) noexcept
{
    const core::i64 truncated = static_cast<core::i64>(position);
    return (position < Real{0} && static_cast<Real>(truncated) != position) ? truncated - 1 : truncated;
}

} // namespace lpl::voxel

#endif // LPL_VOXEL_BRICK_INL
