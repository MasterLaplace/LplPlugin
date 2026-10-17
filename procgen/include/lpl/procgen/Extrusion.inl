#ifndef LPL_PROCGEN_EXTRUSION_INL
#define LPL_PROCGEN_EXTRUSION_INL

namespace lpl::procgen {

template <typename SolidOutside, typename Emit>
void forEachVoxelFace(const VoxelVolume &volume, const VoxelWindow &window, SolidOutside &&solidOutside, Emit &&emit)
{
    if (volume.empty())
        return;

    const core::u32 highX = window.maxX < volume.width ? window.maxX : volume.width;
    const core::u32 highY = window.maxY < volume.levels ? window.maxY : volume.levels;
    const core::u32 highZ = window.maxZ < volume.depth ? window.maxZ : volume.depth;

    const auto solid = [&volume, &solidOutside](core::i32 x, core::i32 y, core::i32 z) {
        if (x < 0 || y < 0 || z < 0 || static_cast<core::u32>(x) >= volume.width ||
            static_cast<core::u32>(y) >= volume.levels || static_cast<core::u32>(z) >= volume.depth)
            return static_cast<bool>(solidOutside(x, y, z));
        return volume.at(static_cast<core::u32>(x), static_cast<core::u32>(y), static_cast<core::u32>(z)) != 0u;
    };

    for (core::u32 y = window.minY; y < highY; ++y)
        for (core::u32 z = window.minZ; z < highZ; ++z)
            for (core::u32 x = window.minX; x < highX; ++x)
            {
                const core::u8 material = volume.at(x, y, z);
                if (material == 0u)
                    continue;

                const core::f32 x0 = static_cast<core::f32>(x);
                const core::f32 x1 = x0 + 1.0f;
                const core::f32 y0 = static_cast<core::f32>(y);
                const core::f32 y1 = y0 + 1.0f;
                const core::f32 z0 = static_cast<core::f32>(z);
                const core::f32 z1 = z0 + 1.0f;

                const core::i32 ix = static_cast<core::i32>(x);
                const core::i32 iy = static_cast<core::i32>(y);
                const core::i32 iz = static_cast<core::i32>(z);

                if (!solid(ix, iy + 1, iz))
                {
                    const core::f32 quad[12] = {x0, y1, z0, x1, y1, z0, x1, y1, z1, x0, y1, z1};
                    emit(quad, 0.0f, 1.0f, 0.0f, material, x, y, z);
                }
                if (!solid(ix, iy - 1, iz))
                {
                    const core::f32 quad[12] = {x0, y0, z1, x1, y0, z1, x1, y0, z0, x0, y0, z0};
                    emit(quad, 0.0f, -1.0f, 0.0f, material, x, y, z);
                }
                if (!solid(ix + 1, iy, iz))
                {
                    const core::f32 quad[12] = {x1, y0, z0, x1, y1, z0, x1, y1, z1, x1, y0, z1};
                    emit(quad, 1.0f, 0.0f, 0.0f, material, x, y, z);
                }
                if (!solid(ix - 1, iy, iz))
                {
                    const core::f32 quad[12] = {x0, y0, z1, x0, y1, z1, x0, y1, z0, x0, y0, z0};
                    emit(quad, -1.0f, 0.0f, 0.0f, material, x, y, z);
                }
                if (!solid(ix, iy, iz + 1))
                {
                    const core::f32 quad[12] = {x0, y0, z1, x0, y1, z1, x1, y1, z1, x1, y0, z1};
                    emit(quad, 0.0f, 0.0f, 1.0f, material, x, y, z);
                }
                if (!solid(ix, iy, iz - 1))
                {
                    const core::f32 quad[12] = {x1, y0, z0, x1, y1, z0, x0, y1, z0, x0, y0, z0};
                    emit(quad, 0.0f, 0.0f, -1.0f, material, x, y, z);
                }
            }
}

} // namespace lpl::procgen

#endif // LPL_PROCGEN_EXTRUSION_INL
