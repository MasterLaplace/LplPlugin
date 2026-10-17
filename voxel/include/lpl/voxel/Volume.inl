#ifndef LPL_VOXEL_VOLUME_INL
#define LPL_VOXEL_VOLUME_INL

namespace lpl::voxel {

constexpr bool VolumeGeometry::valid() const noexcept
{
    return samples[0] > 0 && samples[1] > 0 && samples[2] > 0 && levels > 0u && levels <= kMaxPyramidLevels &&
           voxelMicrometres > 0.0f && metresPerMicrometre > 0.0f;
}

constexpr core::i64 VolumeGeometry::samplesAtLevel(core::u32 axis, core::u32 level) const noexcept
{
    const core::i64 d = static_cast<core::i64>(1) << level;
    return (samples[axis] + d - 1) / d;
}

constexpr core::i32 VolumeGeometry::bricksAtLevel(core::u32 axis, core::u32 level) const noexcept
{
    const core::i64 n = samplesAtLevel(axis, level);
    return static_cast<core::i32>((n + kBrickEdge - 1) / kBrickEdge);
}

} // namespace lpl::voxel

#endif // LPL_VOXEL_VOLUME_INL
