#ifndef LPL_ENGINE_TERRAIN_SHADOWS_INL
#define LPL_ENGINE_TERRAIN_SHADOWS_INL

namespace lpl::engine {

template <typename ForEachProp>
inline void castChunkShadows(const TerrainStreamer &streamer, const TerrainSurface &surface, TerrainChunk &chunk,
                             ForEachProp &&forEachProp)
{
    const core::u32 size = streamer.chunkParams().size;

    render::HeightfieldPatchParams patch;
    patch.size = size;
    patch.stride = 1u;
    patch.originX = chunk.coord.x * static_cast<core::i32>(size);
    patch.originZ = chunk.coord.z * static_cast<core::i32>(size);

    surface.fillShadowMask(
        patch, [&streamer](core::i32 x, core::i32 z) { return streamer.groundAt(x, z); }, forEachProp, chunk.shade);
}

} // namespace lpl::engine

#endif // LPL_ENGINE_TERRAIN_SHADOWS_INL
