#include <lpl/voxel/Residency.hpp>

namespace lpl::voxel {

namespace {

/**
 * @brief Tells whether a brick lies in the volume.
 *
 * @return Whether the brick is inside the subject at its own level.
 */
bool insideSubject(const VolumeGeometry &geometry, const BrickKey &key) noexcept
{
    const core::i32 idx[3]{key.z, key.y, key.x};
    for (core::u32 axis = 0u; axis < 3u; ++axis)
    {
        if (idx[axis] < 0 || idx[axis] >= geometry.bricksAtLevel(axis, key.level))
            return false;
    }
    return true;
}

} // namespace

core::u32 planResidency(const VolumeGeometry &geometry, const ResidencyParams &params, const core::i64 eye[3],
                        BrickKey *out, core::u32 capacity) noexcept
{
    if (out == nullptr || capacity == 0u || !geometry.valid())
        return 0u;

    const core::u32 finest = params.finestLevel;
    core::u32 coarsest = params.coarsestLevel;
    if (coarsest >= geometry.levels)
        coarsest = geometry.levels - 1u;
    if (coarsest < finest)
        coarsest = finest;

    const core::u32 budget = params.budget < capacity ? params.budget : capacity;
    const core::i32 radius = params.ringRadius < 0 ? 0 : params.ringRadius;

    core::u32 written = 0u;

    for (core::i32 r = 0; r <= radius && written < budget; ++r)
    {
        for (core::u32 level = finest; level <= coarsest && written < budget; ++level)
        {
            const core::i32 cz = brickIndexOfBase(eye[0], level);
            const core::i32 cy = brickIndexOfBase(eye[1], level);
            const core::i32 cx = brickIndexOfBase(eye[2], level);

            for (core::i32 dz = -r; dz <= r && written < budget; ++dz)
            {
                for (core::i32 dy = -r; dy <= r && written < budget; ++dy)
                {
                    for (core::i32 dx = -r; dx <= r && written < budget; ++dx)
                    {
                        const core::i32 az = dz < 0 ? -dz : dz;
                        const core::i32 ay = dy < 0 ? -dy : dy;
                        const core::i32 ax = dx < 0 ? -dx : dx;
                        const core::i32 chebyshev = az > ay ? (az > ax ? az : ax) : (ay > ax ? ay : ax);
                        if (chebyshev != r)
                            continue;

                        const BrickKey key{level, cz + dz, cy + dy, cx + dx};
                        if (!insideSubject(geometry, key))
                            continue;
                        out[written++] = key;
                    }
                }
            }
        }
    }
    return written;
}

} // namespace lpl::voxel
