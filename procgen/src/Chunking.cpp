/**
 * @file Chunking.cpp
 * @brief Implementation of coordinate-driven chunk generation.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/procgen/Chunking.hpp>

#include <lpl/procgen/Random.hpp>
#include <lpl/procgen/ValueNoise.hpp>

namespace lpl::procgen {

core::u32 chunkSeed(const ChunkParams &params, ChunkCoord coord)
{
    // Hash the coordinates rather than combine them arithmetically: chunk
    // (1, 0) and (0, 1) must not share a seed, and a simple sum or xor would
    // give them one.
    const core::u32 mixed = ValueNoise2D::hash2(coord.x, coord.z, params.worldSeed);
    return deriveStream(mixed, 0xC804Bu).state();
}

math::Fixed32 sampleWorldHeight(const ChunkParams &params, core::i32 worldX, core::i32 worldZ)
{
    // World coordinates, not chunk-local ones. This single choice is what makes
    // the whole scheme seamless: the shared edge of two chunks is the same
    // world position, so it evaluates to the same height in both.
    //
    // The layer itself is evaluated by the shared sampler rather than spelled out
    // again here. Spelling it out twice is how a chunked world drifts away from
    // the unchunked one: the copy that is not on the common path stops receiving
    // new parameters, and a chunk then disagrees with the map it belongs to.
    NoiseParams layer = params.noise;
    layer.seed = params.worldSeed;
    return sampleNoiseAt(worldX, worldZ, layer);
}

Heightfield generateChunkTerrain(const ChunkParams &params, ChunkCoord coord)
{
    if (params.size == 0u)
        return Heightfield{};

    Heightfield field{params.size, params.size, math::Fixed32::zero()};
    const core::i32 originX = coord.x * static_cast<core::i32>(params.size);
    const core::i32 originZ = coord.z * static_cast<core::i32>(params.size);

    for (core::u32 z = 0u; z < params.size; ++z)
        for (core::u32 x = 0u; x < params.size; ++x)
            field.at(x, z) =
                sampleWorldHeight(params, originX + static_cast<core::i32>(x), originZ + static_cast<core::i32>(z));

    return field;
}

core::u32 countSeamMismatches(const ChunkParams &params, ChunkCoord a, ChunkCoord b)
{
    if (params.size == 0u)
        return 0u;

    const core::i32 dx = b.x - a.x;
    const core::i32 dz = b.z - a.z;
    // Only 4-adjacent chunks share an edge; anything else has no seam to check.
    if ((dx != 0 && dz != 0) || (dx == 0 && dz == 0))
        return 0u;
    if (dx > 1 || dx < -1 || dz > 1 || dz < -1)
        return 0u;

    const Heightfield fieldA = generateChunkTerrain(params, a);
    const Heightfield fieldB = generateChunkTerrain(params, b);
    const core::u32 last = params.size - 1u;

    const core::i32 originAX = a.x * static_cast<core::i32>(params.size);
    const core::i32 originAZ = a.z * static_cast<core::i32>(params.size);
    const core::i32 originBX = b.x * static_cast<core::i32>(params.size);
    const core::i32 originBZ = b.z * static_cast<core::i32>(params.size);

    core::u32 mismatches = 0u;
    for (core::u32 i = 0u; i < params.size; ++i)
    {
        // The two edge cells are one world cell apart, never the same one, so
        // there is nothing to compare them to each other. What has to hold is that
        // EACH chunk's edge cell equals what the world function says at that
        // chunk's own world position — checked on both sides, because a chunk that
        // agreed with the world function only on the side it was queried from
        // would still leave a step at the seam.
        math::Fixed32 valueA{};
        math::Fixed32 valueB{};
        core::i32 worldAX = 0;
        core::i32 worldAZ = 0;
        core::i32 worldBX = 0;
        core::i32 worldBZ = 0;

        if (dx == 1)
        {
            valueA = fieldA.at(last, i);
            valueB = fieldB.at(0u, i);
            worldAX = originAX + static_cast<core::i32>(last);
            worldAZ = originAZ + static_cast<core::i32>(i);
            worldBX = originBX;
            worldBZ = originBZ + static_cast<core::i32>(i);
        }
        else if (dx == -1)
        {
            valueA = fieldA.at(0u, i);
            valueB = fieldB.at(last, i);
            worldAX = originAX;
            worldAZ = originAZ + static_cast<core::i32>(i);
            worldBX = originBX + static_cast<core::i32>(last);
            worldBZ = originBZ + static_cast<core::i32>(i);
        }
        else if (dz == 1)
        {
            valueA = fieldA.at(i, last);
            valueB = fieldB.at(i, 0u);
            worldAX = originAX + static_cast<core::i32>(i);
            worldAZ = originAZ + static_cast<core::i32>(last);
            worldBX = originBX + static_cast<core::i32>(i);
            worldBZ = originBZ;
        }
        else
        {
            valueA = fieldA.at(i, 0u);
            valueB = fieldB.at(i, last);
            worldAX = originAX + static_cast<core::i32>(i);
            worldAZ = originAZ;
            worldBX = originBX + static_cast<core::i32>(i);
            worldBZ = originBZ + static_cast<core::i32>(last);
        }

        if (valueA.raw() != sampleWorldHeight(params, worldAX, worldAZ).raw())
            ++mismatches;
        if (valueB.raw() != sampleWorldHeight(params, worldBX, worldBZ).raw())
            ++mismatches;
    }
    return mismatches;
}

TileGrid borderConstraintsFrom(core::u32 size, const TileGrid &neighbour, core::u32 neighbourSide)
{
    TileGrid preset{size, size, kNoTile};
    if (size == 0u || neighbour.width() != size || neighbour.depth() != size || neighbourSide >= 4u)
        return preset;

    const core::u32 last = size - 1u;
    for (core::u32 i = 0u; i < size; ++i)
    {
        // Pin the new chunk's edge to the neighbour's touching edge. Direction
        // order is kNeighbor4 {E, W, S, N}, read as "where the neighbour lies".
        switch (neighbourSide)
        {
        case 0u: preset.at(last, i) = neighbour.at(0u, i); break; // neighbour to the east
        case 1u: preset.at(0u, i) = neighbour.at(last, i); break; // to the west
        case 2u: preset.at(i, last) = neighbour.at(i, 0u); break; // to the south
        default: preset.at(i, 0u) = neighbour.at(i, last); break; // to the north
        }
    }
    return preset;
}

} // namespace lpl::procgen
