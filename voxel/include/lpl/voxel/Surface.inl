#ifndef LPL_VOXEL_SURFACE_INL
#define LPL_VOXEL_SURFACE_INL

namespace lpl::voxel {

constexpr bool SurfaceMesh::indicesInRange() const noexcept
{
    for (core::u32 i = 0u; i < indexCount; ++i)
    {
        if (indices[i] >= pointCount || (textured() && textureIndices[i] >= textureCount))
            return false;
    }
    return true;
}

} // namespace lpl::voxel

#endif // LPL_VOXEL_SURFACE_INL
