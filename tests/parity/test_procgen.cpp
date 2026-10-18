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

using FVec3 = lpl::math::Vec3<lpl::math::Fixed32>;

static int failures = 0;
static void check(bool ok, const char *what)
{
    std::printf("  %s: %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok)
        ++failures;
}

/**
 * @brief FNV-1a fold of every Position (Fixed32 raw) in creation order.
 */
static lpl::core::u32 foldPositions(const lpl::ecs::Registry &registry)
{
    lpl::core::u32 h = 0x811C9DC5u;
    auto step = [&](lpl::core::u32 v) { h = (h ^ v) * 0x01000193u; };
    for (const auto &part : registry.partitions())
    {
        if (!part || !part->archetype().has(lpl::ecs::ComponentId::Position))
            continue;
        for (const auto &chunk : part->chunks())
        {
            const lpl::core::u32 n = chunk->count();
            const auto *pos = static_cast<const FVec3 *>(chunk->readComponent(lpl::ecs::ComponentId::Position));
            if (!pos)
                continue;
            for (lpl::core::u32 i = 0; i < n; ++i)
            {
                step(static_cast<lpl::core::u32>(pos[i].x.raw()));
                step(static_cast<lpl::core::u32>(pos[i].y.raw()));
                step(static_cast<lpl::core::u32>(pos[i].z.raw()));
            }
        }
    }
    return h;
}

/**
 * @brief Every Position in @p registry, flattened.
 */
static std::vector<FVec3> positionsOf(const lpl::ecs::Registry &registry)
{
    std::vector<FVec3> points;
    for (const auto &part : registry.partitions())
    {
        if (!part || !part->archetype().has(lpl::ecs::ComponentId::Position))
            continue;
        for (const auto &chunk : part->chunks())
        {
            const lpl::core::u32 n = chunk->count();
            const auto *pos = static_cast<const FVec3 *>(chunk->readComponent(lpl::ecs::ComponentId::Position));
            for (lpl::core::u32 i = 0; i < n; ++i)
                points.push_back(pos[i]);
        }
    }
    return points;
}

/**
 * @brief Prints an ASCII preview of the fBm relief of @p seed.
 *
 * Each fBm sample lies in [-1, 1) and maps to a ramp index in [0, 9] as (v + 1) / 2 * 10.
 */
static void printReliefPreview(lpl::core::u32 seed)
{
    const char *ramp = " .:-=+*#%@";
    constexpr lpl::core::i32 W = 48, H = 20;
    const lpl::math::Fixed32 scale = lpl::math::Fixed32::fromFloat(0.18f);
    std::printf("-- fBm relief preview (seed %u) --\n", seed);
    for (lpl::core::i32 y = 0; y < H; ++y)
    {
        char line[W + 1];
        for (lpl::core::i32 x = 0; x < W; ++x)
        {
            const lpl::math::Fixed32 fx = lpl::math::Fixed32::fromInt(x) * scale;
            const lpl::math::Fixed32 fz = lpl::math::Fixed32::fromInt(y) * scale;
            const lpl::math::Fixed32 v = lpl::procgen::ValueNoise2D::fbm(fx, fz, 5u, seed);
            lpl::core::i32 idx = ((v.raw() + 0x10000) * 10) >> 17;
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

/**
 * @brief Builds the same world every time, for a given seed.
 */
static lpl::procgen::BuiltWorldStats buildWorld(lpl::ecs::Registry &registry, lpl::core::u32 seed)
{
    lpl::procgen::ScatterRule trees;
    trees.biome = lpl::procgen::BiomeId::Grassland;
    trees.density = 0.08f;
    trees.halfExtent = 0.2f;

    return lpl::procgen::WorldBuilder{seed}
        .cellSize(0.5f)
        .terrain(24u, 24u)
        .erode()
        .rivers()
        .biomes()
        .scatter(trees)
        .materialize(registry);
}

/**
 * @brief Blue-noise scatter: props share the world, not a cell.
 *
 * The exclusion radius varies with how suitable a cell is, so "no two closer
 * than R" is not the invariant any more — but "no two in the same place" is,
 * and it is the one white noise fails. Two props at identical coordinates
 * means intersecting meshes, which is exactly what dart-throwing prevents.
 */
static void checkPropsShareTheWorldNotACell(const lpl::procgen::BuiltWorldStats &sa)
{
    std::printf("\n-- prop placement --\n");
    check(sa.propEntities > 0u, "scatter produces props");

    lpl::ecs::Registry propsOnly;
    lpl::procgen::ScatterRule trees;
    trees.biome = lpl::procgen::BiomeId::Grassland;
    trees.density = 0.08f;
    trees.halfExtent = 0.2f;
    (void) lpl::procgen::WorldBuilder{1337u}
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
}

/**
 * @brief The playability gate judges what was generated.
 *
 * A criterion nothing can satisfy must fail rather than be quietly ignored.
 */
static void checkTheGateJudgesWhatWasGenerated()
{
    std::printf("\n-- playability gate --\n");
    lpl::procgen::CaveParams caves;
    caves.width = 32u;
    caves.depth = 32u;
    caves.minRegionSize = 12u;

    lpl::procgen::GateCriteria criteria;
    criteria.minPathLength = 4u;
    criteria.minWalkableCells = 16u;

    lpl::procgen::WorldBuilder gated{1337u};
    gated.terrain(32u, 32u).caves(caves).validate(criteria);
    lpl::procgen::WorldBuilder twin{1337u};
    twin.terrain(32u, 32u).caves(caves).validate(criteria);

    check(gated.lastQuality().pathLength == twin.lastQuality().pathLength &&
              gated.lastQuality().reachableCells == twin.lastQuality().reachableCells,
          "reachability verdict is deterministic");
    check(gated.gatePassed(), "the generated underground is playable");
    std::printf("  cave: %u walkable, path %u steps, %u dead ends\n", gated.lastQuality().walkableCells,
                gated.lastQuality().pathLength, gated.lastQuality().deadEnds);

    lpl::procgen::GateCriteria impossible = criteria;
    impossible.minWalkableCells = 1000000u;
    lpl::procgen::WorldBuilder refused{1337u};
    refused.terrain(32u, 32u).caves(caves).validate(impossible);
    check(!refused.gatePassed(), "an unsatisfiable criterion rejects the level");
}

/**
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
 */
int main()
{
    std::printf("== procgen: world -> ECS -> editor ==\n\n");

    printReliefPreview(1337u);

    lpl::ecs::Registry a, b, c;
    const lpl::procgen::BuiltWorldStats sa = buildWorld(a, 1337u);
    const lpl::procgen::BuiltWorldStats sb = buildWorld(b, 1337u);
    const lpl::procgen::BuiltWorldStats sc = buildWorld(c, 2024u);

    check(sa.terrainEntities == 24u * 24u, "one ground entity per cell");
    check(sa.terrainEntities == sb.terrainEntities && sa.propEntities == sb.propEntities,
          "same seed builds the same number of entities");
    check(foldPositions(a) == foldPositions(b), "same seed reproduces the world bit-for-bit");
    check(sa.heightSignature == sb.heightSignature, "same seed reproduces the terrain grid");
    check(foldPositions(a) != foldPositions(c), "different seed yields a different world");
    check(sa.heightSignature != sc.heightSignature, "different seed yields different terrain");

    const std::string doc = lpl::editor::toLplScene(a);
    check(doc.rfind("{\"format\":\"lplscene/1\"", 0) == 0, "generated world serializes to .lplscene");
    std::printf("\n  seed 1337 fold = 0x%08X | seed 2024 fold = 0x%08X\n", foldPositions(a), foldPositions(c));
    std::printf("  height_sig 1337 = 0x%08X | biome_sig = 0x%08X\n", sa.heightSignature, sa.biomeSignature);
    std::printf("  .lplscene (first entity): %.170s...\n", doc.c_str());

    checkPropsShareTheWorldNotACell(sa);
    checkTheGateJudgesWhatWasGenerated();

    std::printf("\n%s (%d failures)\n", failures == 0 ? "ALL PASS" : "FAILURES", failures);
    return failures == 0 ? 0 : 1;
}
