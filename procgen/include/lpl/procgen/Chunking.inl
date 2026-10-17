#ifndef LPL_PROCGEN_CHUNKING_INL
#define LPL_PROCGEN_CHUNKING_INL

namespace lpl::procgen {

constexpr ChunkParams parityChunkParams() noexcept
{
    ChunkParams params{};
    params.size = 24u;
    params.worldSeed = 20260728u;
    params.noise.frequency = 0.09f;
    params.noise.amplitude = 14.0f;
    params.noise.octaves = 5u;
    return params;
}

} // namespace lpl::procgen

#endif // LPL_PROCGEN_CHUNKING_INL
