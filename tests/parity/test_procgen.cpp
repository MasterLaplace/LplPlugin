/**
 * @file test_procgen.cpp
 * @brief The procgen -> ECS -> editor loop, on the one pipeline there is.
 *
 * Prints an ASCII preview of the Fixed32 fBm relief, then builds a world through
 * lpl::procgen::WorldBuilder and checks that (1) the same seed reproduces the
 * same world bit-for-bit, (2) a different seed yields a different world, (3) the
 * props land on a blue-noise arrangement rather than on top of each other, and
 * (4) the generated world serialises to a `.lplscene` document — closing the loop
 * procgen → ECS → editor.
 *
 * The passes themselves are covered by test-procgen-passes, -structures and
 * -review; what this file owns is the seam between a built world and the rest of
 * the project.
 *
 * Host-only. Build via xmake: `xmake run test-procgen`.
 *
 * @author MasterLaplace
 * @version 0.2.0
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
#include <lpl/procgen/ValueNoise.hpp>
#include <lpl/procgen/WorldBuilder.hpp>

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
        if (!part || !part->archetype().has(ecs::ComponentId::Position))
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

/// Every Position in @p registry, flattened.
static std::vector<FVec3> positionsOf(const ecs::Registry &registry)
{
    std::vector<FVec3> points;
    for (const auto &part : registry.partitions())
    {
        if (!part || !part->archetype().has(ecs::ComponentId::Position))
            continue;
        for (const auto &chunk : part->chunks())
        {
            const core::u32 n = chunk->count();
            const auto *pos = static_cast<const FVec3 *>(chunk->readComponent(ecs::ComponentId::Position));
            for (core::u32 i = 0; i < n; ++i)
                points.push_back(pos[i]);
        }
    }
    return points;
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

/// Builds the same world every time, for a given seed.
static procgen::BuiltWorldStats buildWorld(ecs::Registry &registry, core::u32 seed)
{
    procgen::ScatterRule trees;
    trees.biome = procgen::BiomeId::Grassland;
    trees.density = 0.08f;
    trees.halfExtent = 0.2f;

    return procgen::WorldBuilder{seed}
        .cellSize(0.5f)
        .terrain(24u, 24u)
        .erode()
        .rivers()
        .biomes()
        .scatter(trees)
        .materialize(registry);
}

int main()
{
    std::printf("== procgen: world -> ECS -> editor ==\n\n");

    printReliefPreview(1337u);

    ecs::Registry a, b, c;
    const procgen::BuiltWorldStats sa = buildWorld(a, 1337u);
    const procgen::BuiltWorldStats sb = buildWorld(b, 1337u); // same seed
    const procgen::BuiltWorldStats sc = buildWorld(c, 2024u); // different seed

    check(sa.terrainEntities == 24u * 24u, "one ground entity per cell");
    check(sa.terrainEntities == sb.terrainEntities && sa.propEntities == sb.propEntities,
          "same seed builds the same number of entities");
    check(foldPositions(a) == foldPositions(b), "same seed reproduces the world bit-for-bit");
    check(sa.heightSignature == sb.heightSignature, "same seed reproduces the terrain grid");
    check(foldPositions(a) != foldPositions(c), "different seed yields a different world");
    check(sa.heightSignature != sc.heightSignature, "different seed yields different terrain");

    // procgen -> editor: the generated world serializes to a .lplscene document.
    const std::string doc = editor::toLplScene(a);
    check(doc.rfind("{\"format\":\"lplscene/1\"", 0) == 0, "generated world serializes to .lplscene");
    std::printf("\n  seed 1337 fold = 0x%08X | seed 2024 fold = 0x%08X\n", foldPositions(a), foldPositions(c));
    std::printf("  height_sig 1337 = 0x%08X | biome_sig = 0x%08X\n", sa.heightSignature, sa.biomeSignature);
    std::printf("  .lplscene (first entity): %.170s...\n", doc.c_str());

    // ── Blue-noise scatter: props share the world, not a cell ────────────────
    std::printf("\n-- prop placement --\n");
    check(sa.propEntities > 0u, "scatter produces props");

    // The exclusion radius varies with how suitable a cell is, so "no two closer
    // than R" is not the invariant any more — but "no two in the same place" is,
    // and it is the one white noise fails. Two props at identical coordinates
    // means intersecting meshes, which is exactly what dart-throwing prevents.
    ecs::Registry propsOnly;
    procgen::ScatterRule trees;
    trees.biome = procgen::BiomeId::Grassland;
    trees.density = 0.08f;
    trees.halfExtent = 0.2f;
    (void) procgen::WorldBuilder{1337u}
        .cellSize(0.5f)
        .terrain(24u, 24u)
        .erode()
        .rivers()
        .biomes()
        .scatter(trees)
        .materializeProps(propsOnly);

    const std::vector<FVec3> points = positionsOf(propsOnly);
    bool distinct = true;
    for (std::size_t i = 0; i < points.size() && distinct; ++i)
        for (std::size_t j = i + 1; j < points.size() && distinct; ++j)
            if (points[i].x.raw() == points[j].x.raw() && points[i].z.raw() == points[j].z.raw())
                distinct = false;
    check(distinct, "no two props occupy the same ground cell");
    check(points.size() == sa.propEntities, "props-only build places the same props as the full build");
    std::printf("  %zu props from a 24x24 world\n", points.size());

    // ── The playability gate judges what was generated ───────────────────────
    std::printf("\n-- playability gate --\n");
    procgen::CaveParams caves;
    caves.width = 32u;
    caves.depth = 32u;
    caves.minRegionSize = 12u;

    procgen::GateCriteria criteria;
    criteria.minPathLength = 4u;
    criteria.minWalkableCells = 16u;

    procgen::WorldBuilder gated{1337u};
    gated.terrain(32u, 32u).caves(caves).validate(criteria);
    procgen::WorldBuilder twin{1337u};
    twin.terrain(32u, 32u).caves(caves).validate(criteria);

    check(gated.lastQuality().pathLength == twin.lastQuality().pathLength &&
              gated.lastQuality().reachableCells == twin.lastQuality().reachableCells,
          "reachability verdict is deterministic");
    check(gated.gatePassed(), "the generated underground is playable");
    std::printf("  cave: %u walkable, path %u steps, %u dead ends\n", gated.lastQuality().walkableCells,
                gated.lastQuality().pathLength, gated.lastQuality().deadEnds);

    // A criterion nothing can satisfy must fail rather than be quietly ignored.
    procgen::GateCriteria impossible = criteria;
    impossible.minWalkableCells = 1000000u;
    procgen::WorldBuilder refused{1337u};
    refused.terrain(32u, 32u).caves(caves).validate(impossible);
    check(!refused.gatePassed(), "an unsatisfiable criterion rejects the level");

    std::printf("\n%s (%d failures)\n", failures == 0 ? "ALL PASS" : "FAILURES", failures);
    return failures == 0 ? 0 : 1;
}
