#include <lpl/procgen/Chunking.hpp>

#include <lpl/procgen/Erosion.hpp>

#include <lpl/math/Random.hpp>
#include <lpl/procgen/ValueNoise.hpp>

namespace lpl::procgen {

namespace {

/**
 * @brief Floor division, so a macro index stays monotonic across the origin.
 *
 * Truncating division folds the world about its own axis: -1 and +1 land in the same cell.
 */
[[nodiscard]] core::i32 floorDivide(core::i32 value, core::i32 divisor)
{
    const core::i32 quotient = value / divisor;
    return (value % divisor != 0 && ((value < 0) != (divisor < 0))) ? quotient - 1 : quotient;
}

/**
 * @brief Height at a macro cell's centre: one sample, at absolute coordinates.
 *
 * The smoothed field, not the detailed one read sparsely. See @ref EndlessRiverParams::trunkOctaves.
 */
[[nodiscard]] math::Fixed32 macroHeight(const ChunkParams &params, const EndlessRiverParams &rivers, core::i32 macroX,
                                        core::i32 macroZ)
{
    const core::i32 span = static_cast<core::i32>(rivers.trunkCells * rivers.coarseCells);
    NoiseParams layer = params.noise;
    layer.seed = params.worldSeed;
    layer.octaves = rivers.trunkOctaves == 0u ? 1u : rivers.trunkOctaves;
    layer.frequency = params.noise.frequency * rivers.trunkFrequencyScale;
    return sampleNoiseAt(macroX * span + span / 2, macroZ * span + span / 2, layer);
}

/**
 * @brief Fills a window of macro heights centred on (@p centreX, @p centreZ).
 *
 * clear() then resize(), never assign(): kstd::vector — the freestanding
 * vector this compiles against in ring 0 — does not have assign(), and the
 * host's std::vector does. A call that only the host can compile is a build
 * that only fails on the target, three build paths later.
 */
void fillMacroWindow(const ChunkParams &params, const EndlessRiverParams &rivers, core::i32 centreX, core::i32 centreZ,
                     core::i32 radius, lpl::pmr::vector<math::Fixed32> &out)
{
    const core::i32 side = 2 * radius + 1;
    out.clear();
    out.resize(static_cast<core::usize>(side) * side, math::Fixed32::zero());
    for (core::i32 z = 0; z < side; ++z)
        for (core::i32 x = 0; x < side; ++x)
            out[static_cast<core::usize>(z) * side + x] =
                macroHeight(params, rivers, centreX - radius + x, centreZ - radius + z);
}

/**
 * @brief Steepest-descent neighbour within a square window.
 *
 * Ties break on the lowest neighbour index so a flat shelf routes identically everywhere.
 */
[[nodiscard]] bool steepestStep(const lpl::pmr::vector<math::Fixed32> &window, core::i32 side, core::i32 &x,
                                core::i32 &z)
{
    const math::Fixed32 here = window[static_cast<core::usize>(z) * side + x];
    core::i32 bestX = x;
    core::i32 bestZ = z;
    math::Fixed32 bestDrop = math::Fixed32::zero();
    for (core::u32 n = 0u; n < 8u; ++n)
    {
        const core::i32 nx = x + kNeighbor8X[n];
        const core::i32 nz = z + kNeighbor8Z[n];
        if (nx < 0 || nz < 0 || nx >= side || nz >= side)
            continue;
        const math::Fixed32 drop = here - window[static_cast<core::usize>(nz) * side + nx];
        if (drop > bestDrop)
        {
            bestDrop = drop;
            bestX = nx;
            bestZ = nz;
        }
    }
    const bool moved = bestX != x || bestZ != z;
    x = bestX;
    z = bestZ;
    return moved;
}

/**
 * @brief Does a trunk segment cover this coarse cell?
 *
 * The ONE implementation. The batched chunk pass and the per-cell reference both
 * call it, so they cannot disagree about what a trunk covers — and "they cannot
 * disagree" is worth more here than any amount of care taken twice, because the
 * whole property being tested is that the answer does not depend on who asked.
 *
 * A macro cell is many coarse cells across, so "this macro cell carries a trunk"
 * cannot mark the cell: that would be a river as wide as the macro grid. What the
 * coarse level draws is the SEGMENT from a trunk cell's centre to its downstream
 * neighbour's, thickened by trunkWidth.
 *
 * It looks one macro cell either side: a segment starting next door still crosses
 * this one, and dropping those breaks every trunk at a macro boundary.
 */
[[nodiscard]] bool trunkCoversCoarseCell(const ChunkParams &params, const EndlessRiverParams &rivers, core::i32 coarseX,
                                         core::i32 coarseZ)
{
    if (!rivers.trunks || rivers.trunkCells == 0u)
        return false;

    const core::i32 macroSpan = static_cast<core::i32>(rivers.trunkCells);
    const core::i32 homeX = floorDivide(coarseX, macroSpan);
    const core::i32 homeZ = floorDivide(coarseZ, macroSpan);
    const core::i32 width = static_cast<core::i32>(rivers.trunkWidth);

    for (core::i32 mz = homeZ - 1; mz <= homeZ + 1; ++mz)
        for (core::i32 mx = homeX - 1; mx <= homeX + 1; ++mx)
        {
            if (!isTrunkMacroCell(params, rivers, mx, mz))
                continue;
            const core::u32 direction = trunkFlowDirection(params, rivers, mx, mz);
            if (direction >= 8u)
                continue;

            const core::i32 fromX = mx * macroSpan + macroSpan / 2;
            const core::i32 fromZ = mz * macroSpan + macroSpan / 2;
            const core::i32 toX = (mx + kNeighbor8X[direction]) * macroSpan + macroSpan / 2;
            const core::i32 toZ = (mz + kNeighbor8Z[direction]) * macroSpan + macroSpan / 2;

            for (core::i32 t = 0; t <= macroSpan; ++t)
            {
                const core::i32 px = fromX + ((toX - fromX) * t) / macroSpan;
                const core::i32 pz = fromZ + ((toZ - fromZ) * t) / macroSpan;
                const core::i32 dx = px - coarseX < 0 ? coarseX - px : px - coarseX;
                const core::i32 dz = pz - coarseZ < 0 ? coarseZ - pz : pz - coarseZ;
                if (dx <= width && dz <= width)
                    return true;
            }
        }
    return false;
}

/** Chunks sampled along each edge of the calibration window. */
constexpr core::i32 kCalibrationSide = 3;

/**
 * @brief Chunks between one sampled chunk and the next.
 *
 * @warning The window used to be nine ADJACENT chunks, and that stopped being enough the moment
 * the walked world's landforms were widened: at a relief frequency of 0.06 a landform spans
 * about a hundred and ten cells, while three chunks of twenty-four span seventy-two. The
 * calibration was measuring less than one landform — sampling a corner and generalising to
 * the world, which is the mistake calibration exists to prevent, one level up. Measured: a
 * two per cent target produced fourteen per cent of the world in river.
 *
 * Spreading the same nine samples over a stride instead of widening the window keeps the
 * cost identical — a bisection already pays nine `markChunkRivers`, and each of those is the
 * expensive part — while covering ground several landforms across.
 */
constexpr core::i32 kCalibrationStride = 7;

} // namespace

core::u32 chunkSeed(const ChunkParams &params, ChunkCoord coord)
{
    const core::u32 mixed = ValueNoise2D::hash2(coord.x, coord.z, params.worldSeed);
    return math::deriveStream(mixed, 0xC804Bu).state();
}

math::Fixed32 applyRelief(const ReliefBlend &blend, core::i32 worldX, core::i32 worldZ, math::Fixed32 invented)
{
    if (blend.mosaic == nullptr || !blend.mosaic->valid())
        return invented;

    math::Fixed32 measured{};
    if (!blend.mosaic->heightAt(worldX, worldZ, measured))
        return invented;

    const math::Fixed32 weight = blend.mosaic->weightAt(worldX, worldZ);
    if (weight.raw() <= 0)
        return invented;

    NoiseParams detail = blend.detail;
    detail.seed = blend.seed ^ 0x5E1F1Eu;
    const math::Fixed32 roughness =
        detail.amplitude > 0.0f ? sampleNoiseAt(worldX, worldZ, detail) : math::Fixed32::zero();

    const math::Fixed32 real = measured + roughness;
    if (weight.raw() >= math::Fixed32::one().raw())
        return real;
    return real * weight + invented * (math::Fixed32::one() - weight);
}

math::Fixed32 sampleWorldHeight(const ChunkParams &params, core::i32 worldX, core::i32 worldZ)
{
    math::wrapCell(params.globe, worldX, worldZ);

    NoiseParams layer = params.noise;
    layer.seed = params.worldSeed;
    const math::Fixed32 invented = sampleNoiseAt(worldX, worldZ, layer);

    ReliefBlend blend{};
    blend.mosaic = params.relief;
    blend.detail = params.reliefDetail;
    blend.seed = params.worldSeed;
    return applyRelief(blend, worldX, worldZ, invented);
}

math::Fixed32 sampleWorldMoisture(const ChunkParams &params, core::i32 worldX, core::i32 worldZ)
{
    NoiseParams moisture{};
    moisture.amplitude = 1.0f;
    moisture.octaves = 3u;
    moisture.seed = params.worldSeed ^ 0x3A15E7u;
    moisture.frequency = params.noise.frequency * 0.6f;

    const core::f32 wet = 0.5f + sampleNoiseAt(worldX, worldZ, moisture).toFloat() * 0.5f;
    return math::Fixed32::fromFloat(wet < 0.0f ? 0.0f : (wet > 1.0f ? 1.0f : wet));
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

bool isTrunkMacroCell(const ChunkParams &params, const EndlessRiverParams &rivers, core::i32 macroX, core::i32 macroZ)
{
    if (!rivers.trunks || rivers.trunkCells == 0u || rivers.coarseCells == 0u)
        return false;
    if (macroHeight(params, rivers, macroX, macroZ) <= math::Fixed32::fromFloat(rivers.seaLevel))
        return false;

    const core::i32 radius = static_cast<core::i32>(rivers.trunkRadius);
    const core::i32 side = 2 * radius + 1;
    lpl::pmr::vector<math::Fixed32> window;
    fillMacroWindow(params, rivers, macroX, macroZ, radius, window);

    core::u32 upstream = 0u;
    for (core::i32 z = 0; z < side; ++z)
        for (core::i32 x = 0; x < side; ++x)
        {
            core::i32 walkX = x;
            core::i32 walkZ = z;
            for (core::i32 step = 0; step < 2 * radius + 2; ++step)
            {
                if (walkX == radius && walkZ == radius)
                {
                    ++upstream;
                    break;
                }
                if (!steepestStep(window, side, walkX, walkZ))
                    break;
            }
        }
    return upstream >= rivers.trunkThreshold;
}

core::u32 trunkFlowDirection(const ChunkParams &params, const EndlessRiverParams &rivers, core::i32 macroX,
                             core::i32 macroZ)
{
    if (rivers.trunkCells == 0u)
        return 0xFFFFFFFFu;

    const math::Fixed32 here = macroHeight(params, rivers, macroX, macroZ);
    core::u32 best = 0xFFFFFFFFu;
    math::Fixed32 bestDrop = math::Fixed32::zero();
    for (core::u32 n = 0u; n < 8u; ++n)
    {
        const math::Fixed32 drop = here - macroHeight(params, rivers, macroX + kNeighbor8X[n], macroZ + kNeighbor8Z[n]);
        if (drop > bestDrop)
        {
            bestDrop = drop;
            best = n;
        }
    }
    return best;
}

bool isRiverCoarseCell(const ChunkParams &params, const EndlessRiverParams &rivers, core::i32 coarseX,
                       core::i32 coarseZ)
{
    if (params.size == 0u || rivers.coarseCells == 0u)
        return false;

    const core::i32 radius = static_cast<core::i32>(rivers.basinRadius);
    const core::i32 side = 2 * radius + 1;
    const core::i32 half = static_cast<core::i32>(rivers.coarseCells / 2u);

    lpl::pmr::vector<math::Fixed32> heights(static_cast<core::usize>(side) * side, math::Fixed32::zero());
    for (core::i32 z = 0; z < side; ++z)
        for (core::i32 x = 0; x < side; ++x)
            heights[static_cast<core::usize>(z) * side + x] =
                sampleWorldHeight(params, (coarseX - radius + x) * static_cast<core::i32>(rivers.coarseCells) + half,
                                  (coarseZ - radius + z) * static_cast<core::i32>(rivers.coarseCells) + half);

    const auto heightAt = [&](core::i32 x, core::i32 z) { return heights[static_cast<core::usize>(z) * side + x]; };

    if (heightAt(radius, radius) <= math::Fixed32::fromFloat(rivers.seaLevel))
        return false;
    if (trunkCoversCoarseCell(params, rivers, coarseX, coarseZ))
        return true;

    core::u32 upstream = 0u;
    for (core::i32 z = 0; z < side; ++z)
        for (core::i32 x = 0; x < side; ++x)
        {
            core::i32 walkX = x;
            core::i32 walkZ = z;
            for (core::i32 step = 0; step < 2 * radius + 2; ++step)
            {
                if (walkX == radius && walkZ == radius)
                {
                    ++upstream;
                    break;
                }
                const math::Fixed32 here = heightAt(walkX, walkZ);
                core::i32 bestX = walkX;
                core::i32 bestZ = walkZ;
                math::Fixed32 bestDrop = math::Fixed32::zero();
                for (core::u32 n = 0u; n < 8u; ++n)
                {
                    const core::i32 nx = walkX + kNeighbor8X[n];
                    const core::i32 nz = walkZ + kNeighbor8Z[n];
                    if (nx < 0 || nz < 0 || nx >= side || nz >= side)
                        continue;
                    const math::Fixed32 drop = here - heightAt(nx, nz);
                    if (drop > bestDrop)
                    {
                        bestDrop = drop;
                        bestX = nx;
                        bestZ = nz;
                    }
                }
                if (bestX == walkX && bestZ == walkZ)
                    break;
                walkX = bestX;
                walkZ = bestZ;
            }
        }
    return upstream >= rivers.riverThreshold;
}

Grid<core::u8> markChunkRivers(const ChunkParams &params, const EndlessRiverParams &rivers, ChunkCoord coord)
{
    if (params.size == 0u || rivers.coarseCells == 0u)
        return Grid<core::u8>{};

    const core::u32 coarsePerChunk = (params.size + rivers.coarseCells - 1u) / rivers.coarseCells;
    const core::i32 radius = static_cast<core::i32>(rivers.basinRadius);
    const core::i32 windowSide = static_cast<core::i32>(coarsePerChunk) + 2 * radius;
    const core::i32 originCoarseX = coord.x * static_cast<core::i32>(coarsePerChunk) - radius;
    const core::i32 originCoarseZ = coord.z * static_cast<core::i32>(coarsePerChunk) - radius;

    lpl::pmr::vector<math::Fixed32> heights(static_cast<core::usize>(windowSide) * windowSide, math::Fixed32::zero());
    const core::i32 half = static_cast<core::i32>(rivers.coarseCells / 2u);
    for (core::i32 z = 0; z < windowSide; ++z)
        for (core::i32 x = 0; x < windowSide; ++x)
        {
            const core::i32 coarseX = originCoarseX + x;
            const core::i32 coarseZ = originCoarseZ + z;
            heights[static_cast<core::usize>(z) * windowSide + x] =
                sampleWorldHeight(params, coarseX * static_cast<core::i32>(rivers.coarseCells) + half,
                                  coarseZ * static_cast<core::i32>(rivers.coarseCells) + half);
        }

    const auto heightAt = [&](core::i32 x, core::i32 z) {
        return heights[static_cast<core::usize>(z) * windowSide + x];
    };

    const auto flowStep = [&](core::i32 &x, core::i32 &z) {
        const math::Fixed32 here = heightAt(x, z);
        core::i32 bestX = x;
        core::i32 bestZ = z;
        math::Fixed32 bestDrop = math::Fixed32::zero();
        for (core::u32 n = 0u; n < 8u; ++n)
        {
            const core::i32 nx = x + kNeighbor8X[n];
            const core::i32 nz = z + kNeighbor8Z[n];
            if (nx < 0 || nz < 0 || nx >= windowSide || nz >= windowSide)
                continue;
            const math::Fixed32 drop = here - heightAt(nx, nz);
            if (drop > bestDrop)
            {
                bestDrop = drop;
                bestX = nx;
                bestZ = nz;
            }
        }
        const bool moved = bestX != x || bestZ != z;
        x = bestX;
        z = bestZ;
        return moved;
    };

    lpl::pmr::vector<core::u32> upstream(static_cast<core::usize>(coarsePerChunk) * coarsePerChunk, 0u);
    const core::i32 maxSteps = 2 * radius + 2;

    for (core::i32 z = 0; z < windowSide; ++z)
        for (core::i32 x = 0; x < windowSide; ++x)
        {
            core::i32 walkX = x;
            core::i32 walkZ = z;
            for (core::i32 step = 0; step < maxSteps; ++step)
            {
                const core::i32 spanX = walkX - x < 0 ? x - walkX : walkX - x;
                const core::i32 spanZ = walkZ - z < 0 ? z - walkZ : walkZ - z;
                const core::i32 localX = walkX - radius;
                const core::i32 localZ = walkZ - radius;
                if (spanX <= radius && spanZ <= radius && localX >= 0 && localZ >= 0 &&
                    localX < static_cast<core::i32>(coarsePerChunk) && localZ < static_cast<core::i32>(coarsePerChunk))
                    ++upstream[static_cast<core::usize>(localZ) * coarsePerChunk + static_cast<core::usize>(localX)];
                if (!flowStep(walkX, walkZ))
                    break;
            }
        }

    lpl::pmr::vector<core::u8> trunkCoarse(static_cast<core::usize>(coarsePerChunk) * coarsePerChunk, 0u);
    for (core::u32 z = 0u; z < coarsePerChunk; ++z)
        for (core::u32 x = 0u; x < coarsePerChunk; ++x)
            trunkCoarse[static_cast<core::usize>(z) * coarsePerChunk + x] =
                trunkCoversCoarseCell(params, rivers,
                                      coord.x * static_cast<core::i32>(coarsePerChunk) + static_cast<core::i32>(x),
                                      coord.z * static_cast<core::i32>(coarsePerChunk) + static_cast<core::i32>(z)) ?
                    1u :
                    0u;

    const math::Fixed32 sea = math::Fixed32::fromFloat(rivers.seaLevel);
    Grid<core::u8> mask{params.size, params.size, 0u};
    for (core::u32 z = 0u; z < params.size; ++z)
        for (core::u32 x = 0u; x < params.size; ++x)
        {
            const core::u32 coarseX = x / rivers.coarseCells;
            const core::u32 coarseZ = z / rivers.coarseCells;
            if (coarseX >= coarsePerChunk || coarseZ >= coarsePerChunk)
                continue;
            const core::usize slot = static_cast<core::usize>(coarseZ) * coarsePerChunk + coarseX;
            const core::u32 count = upstream[slot];
            if (count < rivers.riverThreshold && trunkCoarse[slot] == 0u)
                continue;
            if (heightAt(static_cast<core::i32>(coarseX) + radius, static_cast<core::i32>(coarseZ) + radius) <= sea)
                continue;
            mask.at(x, z) = 1u;
        }
    return mask;
}

core::u32 coarseFlowDirection(const ChunkParams &params, const EndlessRiverParams &rivers, core::i32 coarseX,
                              core::i32 coarseZ)
{
    if (rivers.coarseCells == 0u)
        return 0xFFFFFFFFu;

    const core::i32 span = static_cast<core::i32>(rivers.coarseCells);
    const core::i32 half = span / 2;
    const auto centreHeight = [&](core::i32 cx, core::i32 cz) {
        return sampleWorldHeight(params, cx * span + half, cz * span + half);
    };

    const math::Fixed32 here = centreHeight(coarseX, coarseZ);
    core::u32 best = 0xFFFFFFFFu;
    math::Fixed32 bestDrop = math::Fixed32::zero();
    for (core::u32 n = 0u; n < 8u; ++n)
    {
        const math::Fixed32 drop = here - centreHeight(coarseX + kNeighbor8X[n], coarseZ + kNeighbor8Z[n]);
        if (drop > bestDrop)
        {
            bestDrop = drop;
            best = n;
        }
    }
    return best;
}

FlowDirection markChunkRiverFlow(const ChunkParams &params, const EndlessRiverParams &rivers, ChunkCoord coord,
                                 const Grid<core::u8> &mask)
{
    if (params.size == 0u || rivers.coarseCells == 0u || mask.empty())
        return FlowDirection{};

    const core::u32 coarsePerChunk = (params.size + rivers.coarseCells - 1u) / rivers.coarseCells;
    lpl::pmr::vector<core::u8> coarse(static_cast<core::usize>(coarsePerChunk) * coarsePerChunk, kNoFlow);
    lpl::pmr::vector<core::u8> known(static_cast<core::usize>(coarsePerChunk) * coarsePerChunk, 0u);

    FlowDirection flow{params.size, params.size, kNoFlow};
    for (core::u32 z = 0u; z < params.size; ++z)
        for (core::u32 x = 0u; x < params.size; ++x)
        {
            if (mask.at(x, z) == 0u)
                continue;
            const core::u32 coarseX = x / rivers.coarseCells;
            const core::u32 coarseZ = z / rivers.coarseCells;
            if (coarseX >= coarsePerChunk || coarseZ >= coarsePerChunk)
                continue;
            const core::usize slot = static_cast<core::usize>(coarseZ) * coarsePerChunk + coarseX;
            if (known[slot] == 0u)
            {
                const core::u32 direction = coarseFlowDirection(
                    params, rivers, coord.x * static_cast<core::i32>(coarsePerChunk) + static_cast<core::i32>(coarseX),
                    coord.z * static_cast<core::i32>(coarsePerChunk) + static_cast<core::i32>(coarseZ));
                coarse[slot] = direction == 0xFFFFFFFFu ? kNoFlow : static_cast<core::u8>(direction);
                known[slot] = 1u;
            }
            flow.at(x, z) = coarse[slot];
        }
    return flow;
}

core::f32 measureRiverShare(const ChunkParams &params, const EndlessRiverParams &rivers)
{
    if (params.size == 0u)
        return 0.0f;

    core::u32 cells = 0u;
    core::u32 marked = 0u;
    const core::i32 half = kCalibrationSide / 2;
    for (core::i32 cz = -half; cz <= half; ++cz)
        for (core::i32 cx = -half; cx <= half; ++cx)
        {
            const Grid<core::u8> mask =
                markChunkRivers(params, rivers, ChunkCoord{cx * kCalibrationStride, cz * kCalibrationStride});
            for (core::u32 z = 0u; z < params.size; ++z)
                for (core::u32 x = 0u; x < params.size; ++x)
                {
                    ++cells;
                    if (mask.at(x, z) != 0u)
                        ++marked;
                }
        }
    return cells == 0u ? 0.0f : static_cast<core::f32>(marked) / static_cast<core::f32>(cells);
}

core::u32 calibrateRiverThreshold(const ChunkParams &params, const EndlessRiverParams &rivers, core::f32 targetShare)
{
    if (params.size == 0u || targetShare <= 0.0f)
        return 1u;

    EndlessRiverParams probe = rivers;
    core::u32 low = 1u;
    core::u32 high = (2u * rivers.basinRadius + 1u) * (2u * rivers.basinRadius + 1u);
    if (high < 2u)
        high = 2u;

    while (low < high)
    {
        const core::u32 middle = low + (high - low) / 2u;
        probe.riverThreshold = middle;
        if (measureRiverShare(params, probe) > targetShare)
            low = middle + 1u;
        else
            high = middle;
    }
    return low;
}

Heightfield generateErodedChunkTerrain(const ChunkParams &params, ChunkCoord coord, core::u32 iterations,
                                       core::f32 talus)
{
    if (params.size == 0u)
        return Heightfield{};
    if (iterations == 0u)
        return generateChunkTerrain(params, coord);

    const core::u32 apron = iterations + 1u;
    const core::u32 side = params.size + 2u * apron;
    const core::i32 originX = coord.x * static_cast<core::i32>(params.size) - static_cast<core::i32>(apron);
    const core::i32 originZ = coord.z * static_cast<core::i32>(params.size) - static_cast<core::i32>(apron);

    Heightfield wide{side, side, math::Fixed32::zero()};
    for (core::u32 z = 0u; z < side; ++z)
        for (core::u32 x = 0u; x < side; ++x)
            wide.at(x, z) =
                sampleWorldHeight(params, originX + static_cast<core::i32>(x), originZ + static_cast<core::i32>(z));

    ThermalErosionParams thermal;
    thermal.iterations = iterations;
    thermal.talus = talus;
    (void) thermalErode(wide, thermal);

    Heightfield field{params.size, params.size, math::Fixed32::zero()};
    for (core::u32 z = 0u; z < params.size; ++z)
        for (core::u32 x = 0u; x < params.size; ++x)
            field.at(x, z) = wide.at(x + apron, z + apron);
    return field;
}

core::u32 countSeamMismatches(const ChunkParams &params, ChunkCoord a, ChunkCoord b)
{
    if (params.size == 0u)
        return 0u;

    const core::i32 dx = b.x - a.x;
    const core::i32 dz = b.z - a.z;
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
        switch (neighbourSide)
        {
        case 0u: preset.at(last, i) = neighbour.at(0u, i); break;
        case 1u: preset.at(0u, i) = neighbour.at(last, i); break;
        case 2u: preset.at(i, last) = neighbour.at(i, 0u); break;
        default: preset.at(i, 0u) = neighbour.at(i, last); break;
        }
    }
    return preset;
}

} // namespace lpl::procgen
