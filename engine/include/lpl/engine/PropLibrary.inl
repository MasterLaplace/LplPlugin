/**
 * @file PropLibrary.inl
 * @brief Out-of-line definitions for engine::PropLibrary.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-28
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ENGINE_PROP_LIBRARY_INL
#    define LPL_ENGINE_PROP_LIBRARY_INL

namespace lpl::engine {

inline void PropLibrary::build(const PropLibraryParams &params, core::u32 seed)
{
    _params = params;
    _params.treeSpecies = params.treeSpecies < kMaxSpecies ? params.treeSpecies : kMaxSpecies;
    _params.rockVariants = params.rockVariants < kMaxVariants ? params.rockVariants : kMaxVariants;

    for (core::u32 species = 0u; species < _params.treeSpecies; ++species)
    {
        const procgen::TreeSkeleton skeleton =
            procgen::growTree(procgen::parityTreeParams(static_cast<procgen::TreeSpecies>(species)));

        _segments[species].clear();
        _sprites[species].clear();
        for (core::u32 i = 0u; i < skeleton.branches.size(); ++i)
        {
            const procgen::TreeBranch &b = skeleton.branches[i];
            render::FoliageSegment segment;
            segment.x0 = b.x0.toFloat();
            segment.y0 = b.y0.toFloat();
            segment.z0 = b.z0.toFloat();
            segment.x1 = b.x1.toFloat();
            segment.y1 = b.y1.toFloat();
            segment.z1 = b.z1.toFloat();
            segment.radius0 = b.radius0.toFloat();
            segment.radius1 = b.radius1.toFloat();
            segment.depth = b.depth;
            _segments[species].push_back(segment);
        }
        for (core::u32 i = 0u; i < skeleton.leaves.size(); ++i)
        {
            const procgen::TreeLeaf &l = skeleton.leaves[i];
            render::FoliageSprite sprite;
            sprite.x = l.x.toFloat();
            sprite.y = l.y.toFloat();
            sprite.z = l.z.toFloat();
            sprite.size = l.size.toFloat();
            sprite.depth = l.depth;
            _sprites[species].push_back(sprite);
        }

        _trees[species].segments = _segments[species].empty() ? nullptr : &_segments[species][0];
        _trees[species].segmentCount = static_cast<core::u32>(_segments[species].size());
        _trees[species].sprites = _sprites[species].empty() ? nullptr : &_sprites[species][0];
        _trees[species].spriteCount = static_cast<core::u32>(_sprites[species].size());
        _trees[species].height = skeleton.height.toFloat();
        _trees[species].spread = skeleton.spread.toFloat();
    }

    buildRocks(seed);
}

inline void PropLibrary::plantExtent(core::i32 cellX, core::i32 cellZ, core::f32 &outHeight,
                                     core::f32 &outSpread) const noexcept
{
    const core::u32 hash = render::scatterHash(cellX, cellZ, _params.plantSalt);
    const core::u32 species = hash % _params.treeSpecies;
    const core::f32 scale = scaleFromHash(hash);
    outHeight = _trees[species].height * scale;
    outSpread = _trees[species].spread * scale;
}

inline bool PropLibrary::rockAt(core::i32 cellX, core::i32 cellZ, core::u32 &outVariant,
                                core::f32 &outScale) const noexcept
{
    const core::u32 hash = render::scatterHash(cellX, cellZ, _params.rockSalt);
    if ((hash & (_params.rockOneIn - 1u)) != 0u)
        return false;
    outVariant = (hash >> 6) % _params.rockVariants;
    outScale = 0.5f + static_cast<core::f32>((hash >> 12) & 0x1Fu) * (1.3f / 31.0f);
    return true;
}

inline void PropLibrary::queuePlant(core::i32 cellX, core::i32 cellZ, core::f32 ground, core::f32 light) const
{
    const core::u32 hash = render::scatterHash(cellX, cellZ, _params.plantSalt);
    render::ScatterInstance instance;
    instance.mesh = hash % _params.treeSpecies;
    instance.worldX = static_cast<core::f32>(cellX) + 0.5f;
    instance.worldZ = static_cast<core::f32>(cellZ) + 0.5f;
    instance.worldY = ground;
    instance.scale = scaleFromHash(hash);
    instance.yaw = static_cast<core::f32>((hash >> 16) & 0xFFu) * (6.2831853f / 255.0f);
    instance.light = light;
    _queue.push(instance);
}

inline core::u32 PropLibrary::flushPlants(const render::RenderTarget &rt, const math::Mat4<core::f32> &mvp,
                                          const render::CameraBasis &basis, core::u32 haze) const
{
    render::FoliageStyle style;
    style.hazeTint = haze;
    style.fogDensity = _params.fogDensity;
    style.spriteMinDistance = _params.canopyDistance;
    return _queue.flush(rt, mvp, basis, _trees, _params.treeSpecies, style, _params.viewDistance);
}

inline void PropLibrary::buildRocks(core::u32 seed)
{
    for (core::u32 variant = 0u; variant < _params.rockVariants; ++variant)
    {
        math::Fixed32 control[6][3] = {};
        const core::f32 widths[6] = {0.72f, 0.95f, 0.88f, 0.66f, 0.34f, 0.04f};
        const core::f32 heights[6] = {0.0f, 0.18f, 0.44f, 0.68f, 0.88f, 1.0f};
        for (core::u32 i = 0u; i < 6u; ++i)
        {
            const core::f32 lean =
                1.0f + 0.18f * static_cast<core::f32>(variant) - 0.09f * static_cast<core::f32>(i % 3u);
            control[i][0] = math::Fixed32::fromFloat(widths[i] * lean);
            control[i][1] = math::Fixed32::fromFloat(heights[i]);
            control[i][2] = math::Fixed32{};
        }

        render::Vec3fTopo samples[48];
        const core::u32 written = render::catmullLoopPoints(control, 6u, 6u, samples, 48u);
        if (written < 4u)
            continue;

        // Only the samples that bound a solid: a closed loop comes back through
        // negative radius, and a negative radius revolved is inside out.
        core::f32 profileRadius[48];
        core::f32 profileHeight[48];
        core::u32 kept = 0u;
        for (core::u32 i = 0u; i < written; ++i)
        {
            if (samples[i].x < 0.0f)
                continue;
            profileRadius[kept] = samples[i].x;
            profileHeight[kept] = samples[i].y;
            ++kept;
        }
        if (kept < 2u)
            continue;

        _rocks[variant] =
            render::revolveProfile(profileRadius, profileHeight, kept, 8u, seed ^ (0x8E1u * (variant + 1u)));
    }
}

} // namespace lpl::engine

#endif // LPL_ENGINE_PROP_LIBRARY_INL
