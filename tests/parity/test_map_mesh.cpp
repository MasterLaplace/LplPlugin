/**
 * @file test_map_mesh.cpp
 * @brief The meshers, asserted instead of looked at.
 *
 * Six hundred lines of meshing lived inside `apps/mapview/main.cpp`, where there
 * was no test target and therefore no way to state any of this. Three real defects
 * had already been paid for at that address — a wall wound inside out, a boundary
 * face dropped, and a building sheared into a staircase because the datum that
 * fixed it was written and never wired to a caller. Every one of them needed a
 * human to look at a picture.
 *
 * None of these checks needs an eye:
 *
 *  1. **Only the surface of the void.** A dungeon of pure rock meshes to nothing,
 *     and every open cell contributes exactly one floor quad. Both are counts.
 *  2. **Interior faces are skipped.** A solid 2x2x2 block of voxels has 24 outward
 *     faces, not 48: the six-per-cube answer means the inside is being drawn, which
 *     is what turns a roof into z-fighting noise.
 *  3. **One datum per footprint, not per column.** Meshed against a sloped terrain,
 *     a volume with a flat datum and the same volume without one must differ — that
 *     difference IS the staircase bug, and asserting it is what keeps the fix wired.
 *  4. **The surface is a grid of quads.** (w-1)(d-1) cells, two triangles each.
 *  5. **Deterministic.** The same atlas meshes to the same fold twice, and a
 *     different shading mode folds differently — so the fold is actually reading the
 *     colours it claims to.
 *
 * @author MasterLaplace
 * @copyright MIT License
 */

#include <lpl/procgen/MapMesh.hpp>
#include <lpl/procgen/WorldAtlas.hpp>

#include <cstdio>
#include <string>

using namespace lpl;

static int failures = 0;

static void check(bool ok, const std::string &what)
{
    std::printf("  %s: %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok)
        ++failures;
}

namespace {

/// A ramp rather than a plane: a flat world cannot show a datum bug.
procgen::WorldAtlas slopedAtlas(core::u32 width, core::u32 depth)
{
    procgen::WorldAtlas atlas;
    atlas.height = procgen::Heightfield{width, depth, math::Fixed32{}};
    for (core::u32 z = 0u; z < depth; ++z)
        for (core::u32 x = 0u; x < width; ++x)
            atlas.height.at(x, z) = math::Fixed32::fromFloat(static_cast<float>(x) * 0.5f);
    atlas.width = width;
    atlas.depth = depth;
    (void) procgen::heightRange(atlas.height, atlas.lowest, atlas.highest);
    atlas.biomes = procgen::BiomeMap{width, depth, procgen::BiomeId::Grassland};
    return atlas;
}

} // namespace

int main()
{
    std::printf("=== map mesh ===\n\n");

    constexpr core::u32 kW = 8u;
    constexpr core::u32 kD = 6u;
    const procgen::WorldAtlas atlas = slopedAtlas(kW, kD);

    // ── 1. The surface is a grid of quads ────────────────────────────────────
    std::printf("-- the surface is one quad per cell --\n");
    const procgen::MapMesh surface = procgen::buildSurfaceMesh(atlas, procgen::MapSurfaceStyle{});
    check(surface.size() == static_cast<core::usize>(kW - 1u) * (kD - 1u) * 6u, "(w-1)(d-1) cells, two triangles each");
    procgen::WorldAtlas empty;
    check(procgen::buildSurfaceMesh(empty, procgen::MapSurfaceStyle{}).empty(), "an empty world meshes to nothing");

    // The normal has to point up on a heightfield; a sign slip there lights the
    // whole map from underneath and reads as a flat grey sheet.
    bool everyNormalUp = true;
    for (core::usize i = 0u; i < surface.size(); ++i)
        if (surface[i].ny <= 0.0f)
            everyNormalUp = false;
    check(everyNormalUp, "every surface normal points up");

    // ── 2. Deterministic, and actually reading the shading ───────────────────
    std::printf("\n-- the fold is a fold --\n");
    const core::u32 first = procgen::foldMapMesh(surface);
    const core::u32 again = procgen::foldMapMesh(procgen::buildSurfaceMesh(atlas, procgen::MapSurfaceStyle{}));
    check(first == again, "the same atlas meshes to the same fold");
    std::printf("     surface fold 0x%08X\n", first);

    procgen::MapSurfaceStyle heightStyle;
    heightStyle.shading = procgen::MapShading::Height;
    const core::u32 shaded = procgen::foldMapMesh(procgen::buildSurfaceMesh(atlas, heightStyle));
    check(shaded != first, "a different shading mode folds differently");

    // ── 3. A dungeon is the surface of the void ──────────────────────────────
    std::printf("\n-- the underground is the surface of the void --\n");
    procgen::WorldAtlas solid = slopedAtlas(kW, kD);
    solid.dungeon = procgen::DungeonMap{kW, kD, procgen::DungeonCell::Wall};
    check(procgen::buildDungeonMesh(solid, 4.0f).empty(), "solid rock meshes to nothing");

    procgen::WorldAtlas hollow = slopedAtlas(kW, kD);
    hollow.dungeon = procgen::DungeonMap{kW, kD, procgen::DungeonCell::Wall};
    hollow.dungeon.at(3u, 3u) = procgen::DungeonCell::Floor;
    const procgen::MapMesh one = procgen::buildDungeonMesh(hollow, 4.0f);
    // One floor, four walls (all four neighbours are rock), one cap: six quads.
    check(one.size() == 6u * 6u, "one open cell gives floor + four walls + a cap");

    hollow.dungeon.at(4u, 3u) = procgen::DungeonCell::Floor;
    const procgen::MapMesh two = procgen::buildDungeonMesh(hollow, 4.0f);
    // Two adjacent cells: 2 floors + 6 walls (the shared face is not a boundary) +
    // 2 caps = 10 quads. Twelve would mean the shared wall was drawn twice, which is
    // the "inside of a solid" mistake that reads as noise rather than as a wall.
    check(two.size() == 10u * 6u, "two adjacent cells do not wall themselves off from each other");

    // Outside the map counts as rock, or a cave is open to nothing at the border.
    procgen::WorldAtlas edge = slopedAtlas(kW, kD);
    edge.dungeon = procgen::DungeonMap{kW, kD, procgen::DungeonCell::Wall};
    edge.dungeon.at(0u, 0u) = procgen::DungeonCell::Floor;
    check(procgen::buildDungeonMesh(edge, 4.0f).size() == 6u * 6u,
          "a corner cell is still walled in on all four sides");

    // ── 4. Voxels: interior faces are skipped ────────────────────────────────
    std::printf("\n-- a solid block shows only its outside --\n");
    procgen::VoxelVolume block;
    block.width = 2u;
    block.depth = 2u;
    block.levels = 2u;
    block.cells.assign(8u, static_cast<core::u8>(1u));
    const procgen::Rgb palette[2] = {
        {0.0f, 0.0f, 0.0f},
        {0.5f, 0.5f, 0.5f}
    };
    const procgen::MapMesh cube = procgen::buildVoxelMesh(block, atlas, 0.0f, palette, 2u);
    check(cube.size() == 24u * 6u, "a 2x2x2 block has 24 outward faces, not 48");

    // ── 5. One datum per footprint, not per column ───────────────────────────
    std::printf("\n-- a plan sits on one datum --\n");
    lpl::pmr::vector<float> flat(static_cast<core::usize>(kW) * kD, 0.0f);
    const procgen::MapMesh onDatum = procgen::buildVoxelMesh(block, atlas, 0.0f, palette, 2u, flat.data(), flat.size());
    check(procgen::foldMapMesh(onDatum) != procgen::foldMapMesh(cube),
          "a footprint datum moves the volume off the per-column ground");
    // And it is the FLAT one that is flat: every base vertex at the same height.
    float lowestY = 1.0e9f;
    float highestY = -1.0e9f;
    for (core::usize i = 0u; i < onDatum.size(); ++i)
    {
        if (onDatum[i].y < lowestY)
            lowestY = onDatum[i].y;
        if (onDatum[i].y > highestY)
            highestY = onDatum[i].y;
    }
    check(highestY - lowestY == 2.0f, "on a flat datum the block is exactly two levels tall on a sloped world");

    // ── 6. The palette is honoured, and index 0 is never a colour ────────────
    std::printf("\n-- the palette --\n");
    check(procgen::buildVoxelMesh(block, atlas, 0.0f, nullptr, 0u).empty(), "no palette means no mesh, not a crash");

    // ── 7. The atlas keeps what the snapshot throws away ─────────────────────
    std::printf("\n-- an atlas is a snapshot, extended --\n");
    {
        // The claim the editor's Atlas panel rests on: a world can be re-derived from
        // its recipe WITH the intermediates a game does not read. Unasserted, the panel
        // would silently show empty layers the day a copy-out line went missing.
        const procgen::WorldRecipe recipe = procgen::parityWorldRecipe();
        const procgen::WorldSnapshot lean = procgen::buildSnapshot(recipe, nullptr, nullptr);
        const procgen::WorldAtlas full = procgen::buildAtlas(recipe, nullptr, nullptr);

        check(full.width == lean.width && full.depth == lean.depth, "both describe the same world");
        // Identical where they overlap: an atlas that regenerated differently would be a
        // second source of truth rather than a diagnostic view of the first.
        bool sameHeights = full.height.width() == lean.height.width();
        for (core::u32 z = 0u; sameHeights && z < lean.depth; ++z)
            for (core::u32 x = 0u; x < lean.width; ++x)
                if (full.height.at(x, z).raw() != lean.height.at(x, z).raw())
                {
                    sameHeights = false;
                    break;
                }
        check(sameHeights, "the shared heightfield is bit-identical, not merely similar");

        // And the diagnostics only the atlas has.
        check(!full.drainage.accumulation.empty(), "the atlas keeps the drainage the rivers came from");
        check(full.drainage.maxAccumulation > 0u, "with a real trunk, not an empty grid");
        check(!full.climate.empty(), "and the six climate axes the classifier read");
        std::printf("     trunk drains %u cells; %zu building plots retained\n", full.drainage.maxAccumulation,
                    full.plots.size());
    }

    std::printf("\n%s\n", failures == 0 ? "ALL PASS (0 failures)" : "FAILURES");
    return failures == 0 ? 0 : 1;
}
