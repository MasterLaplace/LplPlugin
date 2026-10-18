#ifndef LPL_ZARR_ARRAY_INL
#define LPL_ZARR_ARRAY_INL

namespace lpl::zarr {

constexpr bool ArrayMeta::valid() const noexcept
{
    if (dimensions < 1u || dimensions > kMaxDimensions || codec == Codec::Unsupported)
        return false;
    if (separator != '.' && separator != '/')
        return false;
    core::usize bytes = 1u;
    for (core::u32 d = 0u; d < dimensions; ++d)
    {
        if (shape[d] < 0 || chunks[d] < 1)
            return false;
        if (static_cast<core::u64>(chunks[d]) > std::numeric_limits<core::usize>::max() / bytes)
            return false;
        bytes *= static_cast<core::usize>(chunks[d]);
    }
    return true;
}

constexpr core::usize ArrayMeta::chunkBytes() const noexcept
{
    if (!valid())
        return 0u;
    core::usize bytes = 1u;
    for (core::u32 d = 0u; d < dimensions; ++d)
        bytes *= static_cast<core::usize>(chunks[d]);
    return bytes;
}

constexpr core::i64 ArrayMeta::chunkCount(core::u32 axis) const noexcept
{
    if (!valid() || axis >= dimensions)
        return 0;
    return shape[axis] / chunks[axis] + (shape[axis] % chunks[axis] != 0 ? 1 : 0);
}

} // namespace lpl::zarr

#endif // LPL_ZARR_ARRAY_INL
