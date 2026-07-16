/**
 * @file test_procgen.cpp
 * @brief Determinism + pipeline proof for the procgen heightfield generator.
 *
 * Prints an ASCII preview of the Fixed32 fBm relief, then generates a world into
 * an ECS registry and checks that (1) the same seed reproduces the same world
 * bit-for-bit, (2) a different seed yields a different world, and (3) the
 * generated world serializes to a `.lplscene` document — closing the loop
 * procgen → ECS → editor.
 *
 * Host-only. Build via xmake: `xmake run test-procgen`.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-16
 * @copyright MIT License
 */

#include <cstdio>
#include <vector>

#include <lpl/ecs/Component.hpp>
#include <lpl/ecs/Partition.hpp>
#include <lpl/ecs/Registry.hpp>
#include <lpl/editor/SceneSerializer.hpp>
#include <lpl/math/FixedPoint.hpp>
#include <lpl/math/Vec3.hpp>
#include <lpl/procgen/HeightfieldGenerator.hpp>
#include <lpl/procgen/PlayabilityGate.hpp>
#include <lpl/procgen/PoissonScatter.hpp>
#include <lpl/procgen/ValueNoise.hpp>

using namespace lpl;
using math::Fixed32;
using FVec3 = math::Vec3<Fixed32>;

static int failures = 0;
static void check(bool ok, const char *what)
{
    std::printf("  %s: %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok)
        ++failures;
}

// FNV-1a fold of every Position (Fixed32 raw) in creation order.
static core::u32 foldPositions(const ecs::Registry &registry)
{
    core::u32 h = 0x811C9DC5u;
    auto step = [&](core::u32 v) { h = (h ^ v) * 0x01000193u; };
    for (const auto &part : registry.partitions())
    {
        if (!part)
            continue;
        for (const auto &chunk : part->chunks())
        {
            const core::u32 n = chunk->count();
            const auto *pos = static_cast<const FVec3 *>(chunk->readComponent(ecs::ComponentId::Position));
            if (!pos)
                continue;
            for (core::u32 i = 0; i < n; ++i)
            {
                step(static_cast<core::u32>(pos[i].x.raw()));
                step(static_cast<core::u32>(pos[i].y.raw()));
                step(static_cast<core::u32>(pos[i].z.raw()));
            }
        }
    }
    return h;
}

static void printReliefPreview(core::u32 seed)
{
    const char *ramp = " .:-=+*#%@";
    constexpr core::i32 W = 48, H = 20;
    const Fixed32 scale = Fixed32::fromFloat(0.18f);
    std::printf("-- fBm relief preview (seed %u) --\n", seed);
    for (core::i32 y = 0; y < H; ++y)
    {
        char line[W + 1];
        for (core::i32 x = 0; x < W; ++x)
        {
            const Fixed32 fx = Fixed32::fromInt(x) * scale;
            const Fixed32 fz = Fixed32::fromInt(y) * scale;
            const Fixed32 v = procgen::ValueNoise2D::fbm(fx, fz, 5u, seed); // [-1,1)
            // map [-1,1) -> [0,9]
            core::i32 idx = ((v.raw() + 0x10000) * 10) >> 17; // (v+1)/2 * 10
            if (idx < 0)
                idx = 0;
            if (idx > 9)
                idx = 9;
            line[x] = ramp[idx];
        }
        line[W] = '\0';
        std::printf("  %s\n", line);
    }
    std::printf("\n");
}

int main()
{
    std::printf("== procgen heightfield determinism + pipeline ==\n\n");

    printReliefPreview(1337u);

    procgen::HeightfieldParams params;
    params.seed = 1337u;
    params.cols = 24u;
    params.rows = 24u;

    ecs::Registry a, b, c;
    const core::u32 na = procgen::generateHeightfield(a, params);
    const core::u32 nb = procgen::generateHeightfield(b, params); // same seed
    params.seed = 2024u;
    const core::u32 nc = procgen::generateHeightfield(c, params); // different seed

    check(na == params.cols * params.rows && nb == na && nc == na, "entity count = cols*rows");
    check(foldPositions(a) == foldPositions(b), "same seed reproduces the world bit-for-bit");
    check(foldPositions(a) != foldPositions(c), "different seed yields a different world");

    // procgen -> editor: the generated world serializes to a .lplscene document.
    const std::string doc = editor::toLplScene(a);
    check(doc.rfind("{\"format\":\"lplscene/1\"", 0) == 0, "generated world serializes to .lplscene");
    std::printf("\n  seed 1337 fold = 0x%08X | seed 2024 fold = 0x%08X\n", foldPositions(a), foldPositions(c));
    std::printf("  .lplscene (first entity): %.170s...\n", doc.c_str());

    // ── Poisson scatter: deterministic + minimum-distance invariant ──────────
    std::printf("\n-- Poisson-disk scatter --\n");
    procgen::PoissonScatterParams sp;
    sp.seed = 4242u;
    sp.width = 16.0f;
    sp.depth = 16.0f;
    sp.radius = 1.5f;

    ecs::Registry s1, s2, s3;
    const core::u32 n1 = procgen::scatterPoisson(s1, sp);
    const core::u32 n2 = procgen::scatterPoisson(s2, sp); // same seed
    sp.seed = 9001u;
    const core::u32 n3 = procgen::scatterPoisson(s3, sp); // different seed

    check(n1 > 0u, "scatter produces points");
    check(n1 == n2 && foldPositions(s1) == foldPositions(s2), "same seed reproduces scatter bit-for-bit");
    check(foldPositions(s1) != foldPositions(s3), "different seed yields a different scatter");

    // Minimum-distance invariant: no two points closer than radius (squared, Fixed32).
    std::vector<FVec3> pts;
    for (const auto &part : s1.partitions())
        if (part)
            for (const auto &chunk : part->chunks())
            {
                const core::u32 cn = chunk->count();
                const auto *pos = static_cast<const FVec3 *>(chunk->readComponent(ecs::ComponentId::Position));
                for (core::u32 i = 0; i < cn; ++i)
                    pts.push_back(pos[i]);
            }
    const Fixed32 radius = Fixed32::fromFloat(1.5f);
    const Fixed32 r2 = radius * radius;
    bool spaced = true;
    for (std::size_t i = 0; i < pts.size() && spaced; ++i)
        for (std::size_t j = i + 1; j < pts.size() && spaced; ++j)
        {
            const Fixed32 dx = pts[i].x - pts[j].x;
            const Fixed32 dz = pts[i].z - pts[j].z;
            if (dx * dx + dz * dz < r2)
                spaced = false;
        }
    check(spaced, "no two scattered points closer than radius");
    std::printf("  seed 4242 -> %u points (fold 0x%08X) | seed 9001 -> %u points\n", n1, foldPositions(s1), n3);

    // ── Playability gate: deterministic reachability verdict ─────────────────
    std::printf("\n-- Dijkstra playability gate --\n");
    procgen::PlayabilityParams gp;
    gp.seed = 1337u;
    gp.cols = 32u;
    gp.rows = 32u;
    gp.wallThreshold = 0.6f;
    gp.startCol = 0u;
    gp.startRow = 0u;
    gp.goalCol = 31u;
    gp.goalRow = 31u;

    const procgen::PlayabilityResult g1 = procgen::evaluateReachability(gp);
    const procgen::PlayabilityResult g2 = procgen::evaluateReachability(gp);
    check(g1.reachable == g2.reachable && g1.pathCost.raw() == g2.pathCost.raw() && g1.visited == g2.visited,
          "reachability verdict is deterministic");
    std::printf("  seed 1337 (0,0)->(31,31): reachable=%s cost=%.3f visited=%u\n", g1.reachable ? "yes" : "no",
                g1.pathCost.toFloat(), g1.visited);

    // A wall threshold that blocks the start cell must report unreachable.
    procgen::PlayabilityParams blocked = gp;
    blocked.wallThreshold = -1.0f; // everything is a wall
    const procgen::PlayabilityResult gb = procgen::evaluateReachability(blocked);
    check(!gb.reachable, "all-wall grid reports unreachable");

    std::printf("\n%s (%d failures)\n", failures == 0 ? "ALL PASS" : "FAILURES", failures);
    return failures == 0 ? 0 : 1;
}
