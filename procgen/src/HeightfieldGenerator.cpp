/**
 * @file HeightfieldGenerator.cpp
 * @brief Implementation of the noise-driven heightfield generator.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-16
 * @copyright MIT License
 */

#include <lpl/procgen/HeightfieldGenerator.hpp>

#include <lpl/ecs/Archetype.hpp>
#include <lpl/ecs/Component.hpp>
#include <lpl/ecs/Partition.hpp>
#include <lpl/ecs/Registry.hpp>
#include <lpl/math/FixedPoint.hpp>
#include <lpl/math/Vec3.hpp>
#include <lpl/procgen/ValueNoise.hpp>

namespace lpl::procgen {

using FVec3 = math::Vec3<math::Fixed32>;

core::u32 generateHeightfield(ecs::Registry &registry, const HeightfieldParams &params)
{
    const core::u32 total = params.cols * params.rows;
    if (total == 0)
        return 0;

    const ecs::ComponentId ids[] = {ecs::ComponentId::Position, ecs::ComponentId::AABB};
    const ecs::Archetype archetype{ids};
    for (core::u32 i = 0; i < total; ++i)
        (void) registry.createEntity(archetype);

    // math::Fixed32 versions of the params (author once, sample deterministically).
    const math::Fixed32 spacing = math::Fixed32::fromFloat(params.spacing);
    const math::Fixed32 noiseScale = math::Fixed32::fromFloat(params.noiseScale);
    const math::Fixed32 amplitude = math::Fixed32::fromFloat(params.amplitude);
    const math::Fixed32 cubeSize = math::Fixed32::fromFloat(2.0f * params.cubeHalf);
    const core::i32 halfCols = static_cast<core::i32>(params.cols / 2u);
    const core::i32 halfRows = static_cast<core::i32>(params.rows / 2u);

    // Seed the chunk buffers in creation order (grid index == creation index).
    core::u32 gi = 0;
    for (const auto &partition : registry.partitions())
    {
        if (!partition)
            continue;
        for (const auto &chunk : partition->chunks())
        {
            if (!chunk)
                continue;
            const core::u32 count = chunk->count();
            auto *pos = static_cast<FVec3 *>(chunk->writeComponent(ecs::ComponentId::Position));
            auto *posR = static_cast<FVec3 *>(const_cast<void *>(chunk->readComponent(ecs::ComponentId::Position)));
            auto *aabb = static_cast<FVec3 *>(chunk->writeComponent(ecs::ComponentId::AABB));
            auto *aabbR = static_cast<FVec3 *>(const_cast<void *>(chunk->readComponent(ecs::ComponentId::AABB)));
            if (!pos)
                continue;
            for (core::u32 li = 0; li < count && gi < total; ++li, ++gi)
            {
                const core::i32 cx = static_cast<core::i32>(gi % params.cols);
                const core::i32 cz = static_cast<core::i32>(gi / params.cols);
                const math::Fixed32 x = math::Fixed32::fromInt(cx - halfCols) * spacing;
                const math::Fixed32 z = math::Fixed32::fromInt(cz - halfRows) * spacing;
                const math::Fixed32 h =
                    ValueNoise2D::fbm(x * noiseScale, z * noiseScale, params.octaves, params.seed) * amplitude;
                const FVec3 p{x, h, z};
                const FVec3 s{cubeSize, cubeSize, cubeSize};
                pos[li] = p;
                if (posR)
                    posR[li] = p;
                if (aabb)
                    aabb[li] = s;
                if (aabbR)
                    aabbR[li] = s;
            }
        }
    }
    return gi;
}

} // namespace lpl::procgen
