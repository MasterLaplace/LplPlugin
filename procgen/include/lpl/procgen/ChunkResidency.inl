#ifndef LPL_PROCGEN_CHUNK_RESIDENCY_INL
#define LPL_PROCGEN_CHUNK_RESIDENCY_INL

namespace lpl::procgen {

inline core::i32 floorDivChunk(core::i32 value, core::i32 divisor) noexcept
{
    const core::i32 quotient = value / divisor;
    return (value % divisor != 0 && ((value < 0) != (divisor < 0))) ? quotient - 1 : quotient;
}

template <typename Chunk>
inline void ChunkResidency<Chunk>::configure(const ChunkParams &chunkParams, const StreamingParams &streamParams,
                                             core::u32 maxResident) noexcept
{
    _chunkParams = chunkParams;
    _streamParams = streamParams;
    _maxResident = maxResident;
}

template <typename Chunk> inline void ChunkResidency<Chunk>::clear() noexcept
{
    _chunks.clear();
    _generated = 0u;
    _released = 0u;
}

template <typename Chunk>
inline const Chunk *ChunkResidency<Chunk>::find(core::i32 chunkX, core::i32 chunkZ) const noexcept
{
    for (core::u32 i = 0u; i < _chunks.size(); ++i)
        if (_chunks[i].coord.x == chunkX && _chunks[i].coord.z == chunkZ)
            return &_chunks[i];
    return nullptr;
}

template <typename Chunk>
inline const Chunk *ChunkResidency<Chunk>::findByCell(core::i32 worldX, core::i32 worldZ) const noexcept
{
    const core::i32 span = static_cast<core::i32>(_chunkParams.size);
    return find(floorDivChunk(worldX, span), floorDivChunk(worldZ, span));
}

template <typename Chunk>
inline const Chunk *ChunkResidency<Chunk>::findByCell(core::i32 worldX, core::i32 worldZ,
                                                      core::u32 &hint) const noexcept
{
    const core::i32 span = static_cast<core::i32>(_chunkParams.size);
    const core::i32 chunkX = floorDivChunk(worldX, span);
    const core::i32 chunkZ = floorDivChunk(worldZ, span);

    if (hint < _chunks.size() && _chunks[hint].coord.x == chunkX && _chunks[hint].coord.z == chunkZ)
        return &_chunks[hint];
    for (core::u32 i = 0u; i < _chunks.size(); ++i)
        if (_chunks[i].coord.x == chunkX && _chunks[i].coord.z == chunkZ)
        {
            hint = i;
            return &_chunks[i];
        }
    return nullptr;
}

template <typename Chunk>
inline void ChunkResidency<Chunk>::localCell(core::i32 worldX, core::i32 worldZ, core::u32 &outX,
                                             core::u32 &outZ) const noexcept
{
    const core::i32 span = static_cast<core::i32>(_chunkParams.size);
    outX = static_cast<core::u32>(worldX - floorDivChunk(worldX, span) * span);
    outZ = static_cast<core::u32>(worldZ - floorDivChunk(worldZ, span) * span);
}

template <typename Chunk>
template <typename Build, typename OnGenerated>
inline void ChunkResidency<Chunk>::stream(core::f32 focusCellX, core::f32 focusCellZ, core::f32 headingX,
                                          core::f32 headingZ, Build &&build, OnGenerated &&onGenerated)
{
    _scratch.clear();
    for (core::u32 i = 0u; i < _chunks.size(); ++i)
        _scratch.push_back(_chunks[i].coord);

    const core::f32 span = static_cast<core::f32>(_chunkParams.size == 0u ? 1u : _chunkParams.size);
    GenerationSource source;
    source.x = math::Fixed32::fromFloat(focusCellX / span);
    source.z = math::Fixed32::fromFloat(focusCellZ / span);
    source.headingX = math::Fixed32::fromFloat(headingX);
    source.headingZ = math::Fixed32::fromFloat(headingZ);

    const StreamingPlan plan = planStreaming(&source, 1u, _scratch.empty() ? nullptr : &_scratch[0],
                                             static_cast<core::u32>(_scratch.size()), _streamParams);

    for (core::u32 i = 0u; i < plan.toRelease.size(); ++i)
        for (core::u32 j = 0u; j < _chunks.size(); ++j)
            if (_chunks[j].coord == plan.toRelease[i])
            {
                _chunks[j] = _chunks[_chunks.size() - 1u];
                _chunks.pop_back();
                ++_released;
                break;
            }

    for (core::u32 i = 0u; i < plan.toGenerate.size(); ++i)
    {
        if (_maxResident != 0u && _chunks.size() >= _maxResident)
            break;
        _chunks.push_back(build(plan.toGenerate[i].coord));
        onGenerated(_chunks[_chunks.size() - 1u]);
        ++_generated;
    }
}

template <typename Chunk>
template <typename HeightOf>
inline core::f32 ChunkResidency<Chunk>::groundAt(core::i32 worldX, core::i32 worldZ, HeightOf &&heightOf) const
{
    core::u32 hint = 0u;
    return groundAt(worldX, worldZ, heightOf, hint);
}

template <typename Chunk>
template <typename HeightOf>
inline core::f32 ChunkResidency<Chunk>::groundAt(core::i32 worldX, core::i32 worldZ, HeightOf &&heightOf,
                                                 core::u32 &hint) const
{
    if (const Chunk *chunk = findByCell(worldX, worldZ, hint); chunk != nullptr)
    {
        core::u32 localX = 0u;
        core::u32 localZ = 0u;
        localCell(worldX, worldZ, localX, localZ);
        return heightOf(*chunk, localX, localZ);
    }
    return sampleWorldHeight(_chunkParams, worldX, worldZ).toFloat();
}

template <typename Chunk>
template <typename HeightOf>
inline math::Fixed32 ChunkResidency<Chunk>::groundFixedAt(core::i32 worldX, core::i32 worldZ, HeightOf &&heightOf) const
{
    if (const Chunk *chunk = findByCell(worldX, worldZ); chunk != nullptr)
    {
        core::u32 localX = 0u;
        core::u32 localZ = 0u;
        localCell(worldX, worldZ, localX, localZ);
        return heightOf(*chunk, localX, localZ);
    }
    return sampleWorldHeight(_chunkParams, worldX, worldZ);
}

} // namespace lpl::procgen

#endif // LPL_PROCGEN_CHUNK_RESIDENCY_INL
