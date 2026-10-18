#ifndef LPL_VOXEL_MOSAIC_INL
#define LPL_VOXEL_MOSAIC_INL

namespace lpl::voxel {

inline void BrickMosaic::clear() noexcept
{
    _count = 0u;
    _finest = 0u;
    _coarsest = 0u;
    _levelMask = 0u;
    for (core::u32 i = 0u; i < kSlots; ++i)
        _index[i] = 0u;
}

} // namespace lpl::voxel

#endif // LPL_VOXEL_MOSAIC_INL
