/**
 * @file PlayabilityGate.cpp
 * @brief Implementation of the math::Fixed32 Dijkstra reachability gate.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-16
 * @copyright MIT License
 */

#include <lpl/procgen/PlayabilityGate.hpp>

#include <lpl/procgen/ValueNoise.hpp>

#include <vector>

namespace lpl::procgen {

PlayabilityResult evaluateReachability(const PlayabilityParams &params)
{
    PlayabilityResult result;
    const core::u32 cols = params.cols;
    const core::u32 rows = params.rows;
    const core::u32 n = cols * rows;
    if (n == 0 || params.startCol >= cols || params.startRow >= rows || params.goalCol >= cols ||
        params.goalRow >= rows)
        return result;

    const math::Fixed32 noiseScale = math::Fixed32::fromFloat(params.noiseScale);
    const math::Fixed32 wall = math::Fixed32::fromFloat(params.wallThreshold);

    // Rasterise the field into per-cell walkability + traversal cost.
    // cost = 1 + max(0, height): climbing a rise costs more, deterministic.
    std::vector<bool> walkable(n);
    std::vector<math::Fixed32> cost(n, math::Fixed32::one());
    for (core::u32 r = 0; r < rows; ++r)
    {
        for (core::u32 c = 0; c < cols; ++c)
        {
            const math::Fixed32 x = math::Fixed32::fromInt(static_cast<core::i32>(c)) * noiseScale;
            const math::Fixed32 z = math::Fixed32::fromInt(static_cast<core::i32>(r)) * noiseScale;
            const math::Fixed32 h = ValueNoise2D::fbm(x, z, params.octaves, params.seed);
            const core::u32 i = c + r * cols;
            walkable[i] = h.raw() < wall.raw();
            cost[i] = math::Fixed32::one() + (h.raw() > 0 ? h : math::Fixed32::zero());
        }
    }

    const core::u32 start = params.startCol + params.startRow * cols;
    const core::u32 goal = params.goalCol + params.goalRow * cols;
    if (!walkable[start] || !walkable[goal])
        return result;

    // Dijkstra, O(N²) min-select (no heap → deterministic, freestanding-friendly).
    const math::Fixed32 kInf = math::Fixed32::fromRaw(0x7FFFFFFF);
    std::vector<math::Fixed32> dist(n, kInf);
    std::vector<bool> done(n, false);
    dist[start] = math::Fixed32::zero();

    for (core::u32 iter = 0; iter < n; ++iter)
    {
        core::i32 u = -1;
        math::Fixed32 best = kInf;
        for (core::u32 i = 0; i < n; ++i)
            if (!done[i] && dist[i].raw() < best.raw())
            {
                best = dist[i];
                u = static_cast<core::i32>(i);
            }
        if (u < 0)
            break;
        done[static_cast<core::u32>(u)] = true;
        ++result.visited;
        if (static_cast<core::u32>(u) == goal)
            break;

        const core::u32 uc = static_cast<core::u32>(u) % cols;
        const core::u32 ur = static_cast<core::u32>(u) / cols;
        const core::i32 dc[4] = {1, -1, 0, 0};
        const core::i32 dr[4] = {0, 0, 1, -1};
        for (int k = 0; k < 4; ++k)
        {
            const core::i32 nc = static_cast<core::i32>(uc) + dc[k];
            const core::i32 nr = static_cast<core::i32>(ur) + dr[k];
            if (nc < 0 || nr < 0 || nc >= static_cast<core::i32>(cols) || nr >= static_cast<core::i32>(rows))
                continue;
            const core::u32 v = static_cast<core::u32>(nc) + static_cast<core::u32>(nr) * cols;
            if (done[v] || !walkable[v])
                continue;
            const math::Fixed32 nd = dist[static_cast<core::u32>(u)] + cost[v];
            if (nd.raw() < dist[v].raw())
                dist[v] = nd;
        }
    }

    result.reachable = dist[goal].raw() < kInf.raw();
    result.pathCost = result.reachable ? dist[goal] : math::Fixed32::zero();
    return result;
}

} // namespace lpl::procgen
