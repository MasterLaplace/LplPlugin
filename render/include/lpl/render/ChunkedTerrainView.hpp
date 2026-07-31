/**
 * @file ChunkedTerrainView.hpp
 * @brief Drawing a set of terrain chunks: visibility, order, level of detail.
 *
 * The pass between "here are the chunks that are loaded" and "here are the patches
 * to fill". Three decisions live here, and each one is a measurement rather than a
 * preference:
 *
 *  - CULL by projecting a chunk's box through the same matrix the triangles use. A
 *    cone in the horizontal plane is the tempting version and it is wrong: pitch the
 *    camera down and the forward vector's horizontal part shrinks until the angle it
 *    measures means nothing, and a wedge of the world goes missing at the edges of
 *    the view. That was on screen before it was understood.
 *  - ORDER nearest first. A depth buffer rejects a pixel that is already covered, so
 *    front-to-back turns overdraw into a comparison instead of a shaded fill. The
 *    sort is insertion: the list is tens of entries and nearly sorted from one frame
 *    to the next, which is the case insertion sort is best at.
 *  - LEVEL OF DETAIL by ring, each ring doubling the sampling stride, capped by the
 *    host's ring count.
 *
 * The chunks are reached through callbacks, so this header knows nothing about how a
 * world stores them, what a biome is, or what else stands on a chunk.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-28
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_RENDER_CHUNKED_TERRAIN_VIEW_HPP
#    define LPL_RENDER_CHUNKED_TERRAIN_VIEW_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/render/HeightfieldPatch.hpp>
#    include <lpl/render/OrbitCamera.hpp>
#    include <lpl/std/vector.hpp>

namespace lpl::render {

/**
 * @struct ChunkedViewParams
 * @brief The shape of the chunk grid and the budget the host allows.
 */
struct ChunkedViewParams {
    core::u32 chunkSize{24u};
    core::u32 lodRings{3u};        ///< Rings of detail; each doubles the stride.
    core::f32 centreY{8.0f};       ///< Vertical centre of a chunk's cull box.
    core::f32 halfHeight{72.0f};   ///< Half-extent of it; generous on purpose — a box
                                   ///< too big only costs a chunk drawn and found
                                   ///< invisible, one too small costs terrain that
                                   ///< vanishes.
    core::f32 ambient{0.28f};
    core::f32 skirtDrop{6.0f};
};

/** @brief One chunk that survived the cull, and how far away it is. */
struct VisibleChunkRef {
    core::u32 index{0u};
    core::f32 distance{0.0f};
    core::i32 ring{0}; ///< Chebyshev distance in chunks from the focus.
};

/** @brief What a pass did, for a readout that reports rather than asserts. */
struct ChunkedViewStats {
    core::u32 considered{0u};
    core::u32 drawn{0u};
    core::u32 culled{0u};
    core::u32 triangles{0u};
    core::u32 nodesVisited{0u}; ///< Hierarchy nodes tested; 0 on a linear pass.
    core::u32 nodesPruned{0u};  ///< Subtrees rejected whole; the point of the tree.
};

/**
 * @class ChunkedTerrainView
 * @brief Culls, orders and draws a resident set of chunks.
 *
 * Holds its visible list across frames so the insertion sort starts from a nearly
 * sorted state and no allocation happens in the render path once warm.
 */
class ChunkedTerrainView {
public:
    [[nodiscard]] const pmr::vector<VisibleChunkRef> &visible() const noexcept { return _visible; }
    [[nodiscard]] const ChunkedViewStats &stats() const noexcept { return _stats; }

    /**
     * @brief Opens a selection pass. Empties the list, keeps its capacity.
     *
     * The pass is split into three calls rather than one loop because WHICH chunks
     * are worth testing is not this class's business: a resident set of a few dozen
     * is best walked linearly, and a large one is best walked through a spatial
     * hierarchy that rejects whole subtrees. Both drive the same @ref consider and
     * the same @ref endSelect, so the cull test, the ring rule and the ordering
     * exist once whichever way the candidates arrive.
     */
    void beginSelect() noexcept
    {
        _visible.clear();
        _stats = ChunkedViewStats{};
    }

    /**
     * @brief Tests one candidate chunk and keeps it if it is on screen.
     * @return true if the chunk survived the frustum test.
     */
    bool consider(const math::Mat4<core::f32> &mvp, const math::Vec3<core::f32> &eye, core::u32 targetWidth,
                  core::u32 targetHeight, const ChunkedViewParams &params, core::i32 focusChunkX,
                  core::i32 focusChunkZ, core::u32 index, core::i32 chunkX, core::i32 chunkZ)
    {
        const core::f32 span = static_cast<core::f32>(params.chunkSize);
        const core::f32 half = span * 0.5f;
        const core::f32 centreX = static_cast<core::f32>(chunkX) * span + half;
        const core::f32 centreZ = static_cast<core::f32>(chunkZ) * span + half;
        ++_stats.considered;

        if (boxOutsideFrustum(mvp, centreX, params.centreY, centreZ, half, params.halfHeight, half, targetWidth,
                              targetHeight))
        {
            ++_stats.culled;
            return false;
        }

        const core::i32 dx = chunkX - focusChunkX;
        const core::i32 dz = chunkZ - focusChunkZ;
        const core::i32 ax = dx < 0 ? -dx : dx;
        const core::i32 az = dz < 0 ? -dz : dz;
        VisibleChunkRef ref;
        ref.index = index;
        ref.distance = approximateLength(centreX - eye.x, centreZ - eye.z);
        ref.ring = ax > az ? ax : az;
        _visible.push_back(ref);
        return true;
    }

    /** @brief Closes the pass: orders the survivors nearest first. */
    void endSelect() noexcept
    {
        for (core::u32 i = 1u; i < _visible.size(); ++i)
        {
            const VisibleChunkRef key = _visible[i];
            core::u32 j = i;
            while (j != 0u && _visible[j - 1u].distance > key.distance)
            {
                _visible[j] = _visible[j - 1u];
                --j;
            }
            _visible[j] = key;
        }
        _stats.drawn = static_cast<core::u32>(_visible.size());
    }

    /** @brief Records how the hierarchy did, for a caller that used one. */
    void noteHierarchy(core::u32 nodesVisited, core::u32 nodesPruned) noexcept
    {
        _stats.nodesVisited = nodesVisited;
        _stats.nodesPruned = nodesPruned;
    }

    /**
     * @brief Selects the chunks worth drawing and orders them nearest first.
     *
     * The linear pass: every resident chunk is a candidate.
     *
     * @param count   Chunks in the resident set.
     * @param coordOf (index, outChunkX, outChunkZ) -> void.
     */
    template <typename CoordOf>
    void select(const math::Mat4<core::f32> &mvp, const math::Vec3<core::f32> &eye, core::u32 targetWidth,
                core::u32 targetHeight, const ChunkedViewParams &params, core::i32 focusChunkX, core::i32 focusChunkZ,
                core::u32 count, CoordOf &&coordOf)
    {
        beginSelect();
        for (core::u32 index = 0u; index < count; ++index)
        {
            core::i32 chunkX = 0;
            core::i32 chunkZ = 0;
            coordOf(index, chunkX, chunkZ);
            consider(mvp, eye, targetWidth, targetHeight, params, focusChunkX, focusChunkZ, index, chunkX, chunkZ);
        }
        endSelect();
    }

    /** @brief Sampling stride for a ring, capped by the host's ring count. */
    [[nodiscard]] static core::u32 strideForRing(core::i32 ring, core::u32 lodRings) noexcept
    {
        const core::u32 rings = lodRings == 0u ? 1u : lodRings;
        const core::u32 clamped = static_cast<core::u32>(ring < 0 ? 0 : ring);
        const core::u32 capped = clamped >= rings ? rings - 1u : clamped;
        return 1u << (capped < 3u ? capped : 2u);
    }

    /**
     * @brief Draws every selected chunk: its patch, its skirt, then its extras.
     *
     * @param accessors (index, ChunkAccess&) -> bool; false skips the chunk (an
     *                  empty field, a chunk released between the two passes).
     * @param extras    (index, patchParams, ring) -> u32 triangles; whatever else
     *                  stands on a chunk — water, props, vegetation. Called AFTER the
     *                  ground so those draws test against a depth buffer that already
     *                  has terrain in it.
     */
    template <typename ForEachChunk>
    void draw(const RenderTarget &rt, const math::Mat4<core::f32> &mvp, const ChunkedViewParams &params,
              const SunState &sun, ForEachChunk &&forEachChunk)
    {
        _stats.triangles = 0u;
        for (core::u32 i = 0u; i < _visible.size(); ++i)
        {
            const VisibleChunkRef &ref = _visible[i];
            HeightfieldPatchParams patch;
            patch.size = params.chunkSize;
            patch.stride = strideForRing(ref.ring, params.lodRings);
            patch.ambient = params.ambient;
            // The origin is the caller's to fill: it knows the chunk's coordinates.
            _stats.triangles += forEachChunk(rt, mvp, sun, ref, patch, params);
        }
    }

private:
    pmr::vector<VisibleChunkRef> _visible;
    ChunkedViewStats _stats{};
};

} // namespace lpl::render

#endif // LPL_RENDER_CHUNKED_TERRAIN_VIEW_HPP
