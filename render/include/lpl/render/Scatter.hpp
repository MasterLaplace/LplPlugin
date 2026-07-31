/**
 * @file Scatter.hpp
 * @brief A queue of instanced props, batched by material and drawn near-to-far.
 *
 * Every world that scatters things on a surface — trees, rocks, grass, debris —
 * needs the same three steps, and they were written inside the world viewer:
 * decide what goes where from the coordinates alone, queue it, then draw the queue
 * in an order that batches materials and lets the depth buffer reject overdraw.
 *
 * Deciding from the coordinates is what makes it work in a streamed world: nothing
 * is stored per prop, so two chunks that meet along a border agree about what grows
 * on it without exchanging anything — the same property the terrain gets by
 * sampling absolute coordinates.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-28
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_RENDER_SCATTER_HPP
#    define LPL_RENDER_SCATTER_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/render/CommandBuffer.hpp>
#    include <lpl/render/Foliage.hpp>
#    include <lpl/std/vector.hpp>

namespace lpl::render {

/**
 * @brief A hash of a cell, for deciding what grows there.
 *
 * The salt separates one scatter layer from another: trees and boulders asking the
 * same question of the same cell must not get the same answer, or every boulder
 * stands in a tree. Mixing a constant between the two coordinates matters too —
 * without it the hash is symmetric in x and z, and the world grows a visible
 * diagonal.
 */
[[nodiscard]] inline core::u32 scatterHash(core::i32 cellX, core::i32 cellZ, core::u32 salt) noexcept
{
    core::u32 hash = 0x811C9DC5u ^ salt;
    hash = (hash ^ static_cast<core::u32>(cellX)) * 0x01000193u;
    hash = (hash ^ 0x9E3779B9u) * 0x01000193u;
    hash = (hash ^ static_cast<core::u32>(cellZ)) * 0x01000193u;
    return hash;
}

/** @brief One queued prop, already resolved to a mesh and a transform. */
struct ScatterInstance {
    core::f32 worldX{0.0f};
    core::f32 worldY{0.0f};
    core::f32 worldZ{0.0f};
    core::f32 scale{1.0f};
    core::f32 yaw{0.0f};
    core::f32 light{1.0f};  ///< Lighting term including any shadow.
    core::u32 mesh{0u};     ///< Index into the caller's mesh table; also the material.
};

/**
 * @class ScatterQueue
 * @brief Collects prop instances for a frame, then draws them in a chosen order.
 *
 * Drawn as they are found, props come out in chunk order: species interleaved and
 * far ones in front of near ones, which are the two things a draw order exists to
 * fix. Queueing costs one small struct per prop and buys both.
 */
class ScatterQueue {
public:
    void clear() noexcept { _instances.clear(); }

    void push(const ScatterInstance &instance) { _instances.push_back(instance); }

    [[nodiscard]] core::u32 size() const noexcept { return static_cast<core::u32>(_instances.size()); }

    /**
     * @brief Sorts by (mesh, distance) and draws every instance.
     *
     * @param meshes    Mesh table; @c ScatterInstance::mesh indexes it.
     * @param meshCount Entries in the table.
     * @param style     Style template; its light and maxDepth are overridden per
     *                  instance, everything else (colours, fog) is taken as given.
     * @param farDistance Distance the depth key is normalised against.
     * @return Triangles submitted.
     */
    core::u32 flush(const RenderTarget &rt, const math::Mat4<core::f32> &mvp, const CameraBasis &basis,
                    const FoliageMesh *meshes, core::u32 meshCount, const FoliageStyle &style, core::f32 farDistance,
                    core::f32 nearFullDetail = 18.0f, core::f32 midDetail = 40.0f)
    {
        const core::u32 count = static_cast<core::u32>(_instances.size());
        if (count == 0u || meshes == nullptr || meshCount == 0u)
            return 0u;

        _keys.clear();
        _keys.resize(count);
        _scratch.clear();
        _scratch.resize(count);

        for (core::u32 i = 0u; i < count; ++i)
        {
            const ScatterInstance &instance = _instances[i];
            const core::f32 distance =
                approximateLength(instance.worldX - basis.eye.x, instance.worldZ - basis.eye.z);
            const core::f32 clamped = distance < 0.0f ? 0.0f : (distance > farDistance ? farDistance : distance);
            const core::u32 depth =
                farDistance > 0.0f ? static_cast<core::u32>((clamped / farDistance) * 65535.0f) : 0u;
            _keys[i].key = packDrawKey(instance.mesh, 0u, depth);
            _keys[i].payload = i;
        }

        radixSortDrawKeys(&_keys[0], &_scratch[0], count);
        _orderFold = foldDrawKeys(&_keys[0], count);

        core::u32 triangles = 0u;
        for (core::u32 i = 0u; i < count; ++i)
        {
            const ScatterInstance &instance = _instances[_keys[i].payload];
            if (instance.mesh >= meshCount)
                continue;

            const core::f32 distance =
                approximateLength(instance.worldX - basis.eye.x, instance.worldZ - basis.eye.z);

            FoliageStyle resolved = style;
            resolved.light = instance.light;
            // Level of detail from the DISTANCE, not from whatever region the prop
            // was found in: a prop at the near corner of a far region is closer than
            // one at the far corner of a near region.
            resolved.maxDepth = distance < nearFullDetail ? 255u : (distance < midDetail ? 2u : 1u);

            FoliageInstance transform;
            transform.x = instance.worldX;
            transform.y = instance.worldY;
            transform.z = instance.worldZ;
            transform.scale = instance.scale;
            transform.yaw = instance.yaw;
            triangles += drawFoliage(rt, mvp, basis, meshes[instance.mesh], transform, resolved);
        }
        return triangles;
    }

    /** @brief Fold of the ordered key stream: the order is reproducible, so it folds. */
    [[nodiscard]] core::u32 orderFold() const noexcept { return _orderFold; }

private:
    pmr::vector<ScatterInstance> _instances;
    pmr::vector<DrawKey> _keys;
    pmr::vector<DrawKey> _scratch;
    core::u32 _orderFold{0u};
};

} // namespace lpl::render

#endif // LPL_RENDER_SCATTER_HPP
