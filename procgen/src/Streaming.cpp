#include <lpl/procgen/Streaming.hpp>

#include <limits>

namespace lpl::procgen {

namespace {

/** Priority is an integer so ordering is exact. Distances are scaled by this. */
constexpr core::i32 kPriorityScale = 256;

[[nodiscard]] bool sameChunk(ChunkCoord a, ChunkCoord b) noexcept { return a.x == b.x && a.z == b.z; }

[[nodiscard]] bool contains(const ChunkCoord *list, core::u32 count, ChunkCoord coord) noexcept
{
    for (core::u32 i = 0u; i < count; ++i)
        if (sameChunk(list[i], coord))
            return true;
    return false;
}

/**
 * @brief Priority of a chunk for one source: scaled distance, less a heading bonus.
 *
 * Lower is more urgent. The heading term subtracts, so a chunk in front is
 * treated as nearer than it is — the "generate ahead of the player" behaviour,
 * expressed as a discount rather than as a separate cone test, which keeps the
 * ordering total and therefore reproducible.
 *
 * The distance is Chebyshev: the generate radius is a square, so the metric that
 * orders chunks should be the one that defines the square. A Euclidean ordering
 * over a square region schedules the corners last for no reason a player can
 * perceive.
 *
 * The heading term is the dot product of the offset with the heading, normalised
 * by the offset's magnitude (here |dx| + |dz|) so the bonus depends on DIRECTION
 * and not on how far away the chunk is. Without that division a distant chunk
 * roughly ahead would outrank a near one exactly ahead.
 */
[[nodiscard]] core::i32 priorityFor(const GenerationSource &source, ChunkCoord coord, const StreamingParams &params)
{
    const math::Fixed32 dx = math::Fixed32::fromInt(coord.x) - source.x;
    const math::Fixed32 dz = math::Fixed32::fromInt(coord.z) - source.z;

    const math::Fixed32 adx = dx.abs();
    const math::Fixed32 adz = dz.abs();
    const math::Fixed32 distance = adx > adz ? adx : adz;

    core::i32 priority = (distance * math::Fixed32::fromInt(kPriorityScale)).toInt();

    if (params.directionWeight16 != 0u)
    {
        const math::Fixed32 dot = dx * source.headingX + dz * source.headingZ;
        const math::Fixed32 magnitude = adx + adz;
        if (magnitude.raw() != 0)
        {
            const math::Fixed32 aligned = dot / magnitude;
            const core::i32 bonus = (aligned * math::Fixed32::fromInt(static_cast<core::i32>(params.directionWeight16) *
                                                                      kPriorityScale / 16))
                                        .toInt();
            priority -= bonus;
        }
    }
    return priority;
}

/**
 * @brief Chebyshev distance, in whole chunks, from @p coord to the nearest of the sources.
 */
[[nodiscard]] core::i32 distanceToNearestSource(const GenerationSource *sources, core::u32 sourceCount,
                                                ChunkCoord coord) noexcept
{
    core::i32 nearest = std::numeric_limits<core::i32>::max();
    for (core::u32 s = 0u; s < sourceCount; ++s)
    {
        const core::i32 dx = coord.x - sources[s].x.toInt();
        const core::i32 dz = coord.z - sources[s].z.toInt();
        const core::i32 adx = dx < 0 ? -dx : dx;
        const core::i32 adz = dz < 0 ? -dz : dz;
        const core::i32 distance = adx > adz ? adx : adz;
        nearest = distance < nearest ? distance : nearest;
    }
    return nearest;
}

/**
 * @brief Keeps the @p budget releases furthest from every source, in the order they were listed.
 *
 * Two releases at the same distance go in the order of the list, so the plan is the same on every
 * target. With no source there is no distance to compare, and the first @p budget releases of the list
 * are kept. A budget of 0 keeps them all.
 */
void keepFurthestReleases(lpl::pmr::vector<ChunkCoord> &releases, const GenerationSource *sources,
                          core::u32 sourceCount, core::u32 budget)
{
    const core::u32 count = static_cast<core::u32>(releases.size());
    if (budget == 0u || count <= budget)
        return;
    if (sourceCount == 0u)
    {
        releases.resize(budget);
        return;
    }

    lpl::pmr::vector<core::i32> distances(count);
    for (core::u32 i = 0u; i < count; ++i)
        distances[i] = distanceToNearestSource(sources, sourceCount, releases[i]);

    lpl::pmr::vector<core::u8> chosen(count, 0u);
    for (core::u32 picked = 0u; picked < budget; ++picked)
    {
        core::u32 furthest = count;
        for (core::u32 i = 0u; i < count; ++i)
            if (chosen[i] == 0u && (furthest == count || distances[i] > distances[furthest]))
                furthest = i;
        chosen[furthest] = 1u;
    }

    core::u32 kept = 0u;
    for (core::u32 i = 0u; i < count; ++i)
        if (chosen[i] != 0u)
            releases[kept++] = releases[i];
    releases.resize(kept);
}

} // namespace

StreamingPlan planStreaming(const GenerationSource *sources, core::u32 sourceCount, const ChunkCoord *resident,
                            core::u32 residentCount, const StreamingParams &params)
{
    StreamingPlan plan;
    plan.resident = residentCount;
    if (sources == nullptr || sourceCount == 0u)
    {
        for (core::u32 i = 0u; i < residentCount; ++i)
            plan.toRelease.push_back(resident[i]);
        keepFurthestReleases(plan.toRelease, nullptr, 0u, params.maxReleasePerTick);
        return plan;
    }

    const core::i32 radius = static_cast<core::i32>(params.generateRadius);

    for (core::u32 s = 0u; s < sourceCount; ++s)
    {
        const GenerationSource &source = sources[s];
        const core::i32 centreX = source.x.toInt();
        const core::i32 centreZ = source.z.toInt();

        for (core::i32 dz = -radius; dz <= radius; ++dz)
        {
            for (core::i32 dx = -radius; dx <= radius; ++dx)
            {
                const ChunkCoord coord{centreX + dx, centreZ + dz};

                core::i32 best = priorityFor(source, coord, params);
                for (core::u32 other = 0u; other < sourceCount; ++other)
                {
                    if (other == s)
                        continue;
                    const core::i32 candidate = priorityFor(sources[other], coord, params);
                    if (candidate < best)
                        best = candidate;
                }
                if (best < 0)
                    best = 0;

                bool alreadyWanted = false;
                for (core::u32 i = 0u; i < plan.toGenerate.size(); ++i)
                    if (sameChunk(plan.toGenerate[i].coord, coord))
                    {
                        alreadyWanted = true;
                        break;
                    }
                if (alreadyWanted)
                    continue;

                ++plan.wanted;
                if (contains(resident, residentCount, coord))
                    continue;
                plan.toGenerate.push_back(StreamingRequest{coord, static_cast<core::u32>(best)});
            }
        }
    }

    for (core::u32 i = 1u; i < plan.toGenerate.size(); ++i)
    {
        const StreamingRequest key = plan.toGenerate[i];
        core::u32 j = i;
        while (j > 0u)
        {
            const StreamingRequest &prev = plan.toGenerate[j - 1u];
            const bool after =
                prev.priority < key.priority ||
                (prev.priority == key.priority &&
                 (prev.coord.z < key.coord.z || (prev.coord.z == key.coord.z && prev.coord.x <= key.coord.x)));
            if (after)
                break;
            plan.toGenerate[j] = prev;
            --j;
        }
        plan.toGenerate[j] = key;
    }

    const core::i32 releaseRadius = static_cast<core::i32>((params.generateRadius * params.releaseRatio16 + 15u) / 16u);

    for (core::u32 i = 0u; i < residentCount; ++i)
    {
        const ChunkCoord coord = resident[i];
        bool keep = false;
        for (core::u32 s = 0u; s < sourceCount && !keep; ++s)
        {
            const core::i32 dx = coord.x - sources[s].x.toInt();
            const core::i32 dz = coord.z - sources[s].z.toInt();
            const core::i32 adx = dx < 0 ? -dx : dx;
            const core::i32 adz = dz < 0 ? -dz : dz;
            if ((adx > adz ? adx : adz) <= releaseRadius)
                keep = true;
        }
        if (!keep)
            plan.toRelease.push_back(coord);
    }

    if (params.maxGeneratePerTick != 0u)
        while (plan.toGenerate.size() > params.maxGeneratePerTick)
            plan.toGenerate.pop_back();
    keepFurthestReleases(plan.toRelease, sources, sourceCount, params.maxReleasePerTick);

    return plan;
}

void ChunkPool::reserve(core::u32 count)
{
    while (capacity() < count)
    {
        _free.push_back(_created);
        _served.push_back(0u);
        ++_created;
    }
}

core::u32 ChunkPool::acquire()
{
    if (_free.empty())
        return kNoSlot;
    const core::u32 slot = _free[_free.size() - 1u];
    _free.pop_back();
    ++_liveCount;
    if (_served[slot] != 0u)
        ++_recycled;
    _served[slot] = 1u;
    return slot;
}

void ChunkPool::release(core::u32 slot)
{
    if (slot == kNoSlot || slot >= _created || _liveCount == 0u)
        return;
    _free.push_back(slot);
    --_liveCount;
}

} // namespace lpl::procgen
