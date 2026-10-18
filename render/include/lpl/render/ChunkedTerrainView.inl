#ifndef LPL_RENDER_CHUNKED_TERRAIN_VIEW_INL
#define LPL_RENDER_CHUNKED_TERRAIN_VIEW_INL

namespace lpl::render {

inline void ChunkedTerrainView::beginSelect() noexcept
{
    _visible.clear();
    _stats = ChunkedViewStats{};
}

inline bool ChunkedTerrainView::consider(const math::Mat4<core::f32> &mvp, const math::Vec3<core::f32> &eye,
                                         core::u32 targetWidth, core::u32 targetHeight, const ChunkedViewParams &params,
                                         core::i32 focusChunkX, core::i32 focusChunkZ, core::u32 index,
                                         core::i32 chunkX, core::i32 chunkZ)
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

inline void ChunkedTerrainView::endSelect() noexcept
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

inline void ChunkedTerrainView::noteHierarchy(core::u32 nodesVisited, core::u32 nodesPruned) noexcept
{
    _stats.nodesVisited = nodesVisited;
    _stats.nodesPruned = nodesPruned;
}

template <typename CoordOf>
inline void ChunkedTerrainView::select(const math::Mat4<core::f32> &mvp, const math::Vec3<core::f32> &eye,
                                       core::u32 targetWidth, core::u32 targetHeight, const ChunkedViewParams &params,
                                       core::i32 focusChunkX, core::i32 focusChunkZ, core::u32 count, CoordOf &&coordOf)
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

inline core::u32 ChunkedTerrainView::strideForRing(core::i32 ring, core::u32 lodRings) noexcept
{
    const core::u32 rings = lodRings == 0u ? 1u : lodRings;
    const core::u32 clamped = static_cast<core::u32>(ring < 0 ? 0 : ring);
    const core::u32 capped = clamped >= rings ? rings - 1u : clamped;
    return 1u << (capped < 3u ? capped : 2u);
}

template <typename ForEachChunk>
inline void ChunkedTerrainView::draw(const RenderTarget &rt, const math::Mat4<core::f32> &mvp,
                                     const ChunkedViewParams &params, const SunState &sun, ForEachChunk &&forEachChunk)
{
    _stats.triangles = 0u;
    for (core::u32 i = 0u; i < _visible.size(); ++i)
    {
        const VisibleChunkRef &ref = _visible[i];
        HeightfieldPatchParams patch;
        patch.size = params.chunkSize;
        patch.stride = strideForRing(ref.ring, params.lodRings);
        patch.ambient = params.ambient;
        patch.cameraX = params.cameraX;
        patch.cameraZ = params.cameraZ;
        patch.curvatureFactor = params.curvatureFactor;
        _stats.triangles += forEachChunk(rt, mvp, sun, ref, patch, params);
    }
}

} // namespace lpl::render

#endif // LPL_RENDER_CHUNKED_TERRAIN_VIEW_INL
