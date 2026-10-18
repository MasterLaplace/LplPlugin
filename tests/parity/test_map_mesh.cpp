#include <lpl/procgen/MapMesh.hpp>
#include <lpl/procgen/WorldAtlas.hpp>

#include <cstdio>
#include <string>

static int failures = 0;

static void check(bool ok, const std::string &what)
{
    std::printf("  %s: %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok)
        ++failures;
}

namespace {

constexpr lpl::core::u32 kW = 8u;
constexpr lpl::core::u32 kD = 6u;

/**
 * @brief A ramp rather than a plane: a flat world cannot show a datum bug.
 */
lpl::procgen::WorldAtlas slopedAtlas(lpl::core::u32 width, lpl::core::u32 depth)
{
    lpl::procgen::WorldAtlas atlas;
    atlas.height = lpl::procgen::Heightfield{width, depth, lpl::math::Fixed32{}};
    for (lpl::core::u32 z = 0u; z < depth; ++z)
        for (lpl::core::u32 x = 0u; x < width; ++x)
            atlas.height.at(x, z) = lpl::math::Fixed32::fromFloat(static_cast<float>(x) * 0.5f);
    atlas.width = width;
    atlas.depth = depth;
    (void) lpl::procgen::heightRange(atlas.height, atlas.lowest, atlas.highest);
    atlas.biomes = lpl::procgen::BiomeMap{width, depth, lpl::procgen::BiomeId::Grassland};
    return atlas;
}

/**
 * @brief The surface is a grid of quads.
 *
 * The normal has to point up on a heightfield; a sign slip there lights the whole map from underneath and reads as a
 * flat grey sheet.
 */
void checkSurfaceIsAGridOfQuads(const lpl::procgen::MapMesh &surface)
{
    std::printf("-- the surface is one quad per cell --\n");
    check(surface.size() == static_cast<lpl::core::usize>(kW - 1u) * (kD - 1u) * 6u,
          "(w-1)(d-1) cells, two triangles each");
    lpl::procgen::WorldAtlas empty;
    check(lpl::procgen::buildSurfaceMesh(empty, lpl::procgen::MapSurfaceStyle{}).empty(),
          "an empty world meshes to nothing");

    bool everyNormalUp = true;
    for (lpl::core::usize i = 0u; i < surface.size(); ++i)
        if (surface[i].ny <= 0.0f)
            everyNormalUp = false;
    check(everyNormalUp, "every surface normal points up");
}

/**
 * @brief Deterministic, and actually reading the shading.
 */
void checkFoldIsDeterministicAndReadsTheShading(const lpl::procgen::WorldAtlas &atlas,
                                                const lpl::procgen::MapMesh &surface)
{
    std::printf("\n-- the fold is a fold --\n");
    const lpl::core::u32 first = lpl::procgen::foldMapMesh(surface);
    const lpl::core::u32 again =
        lpl::procgen::foldMapMesh(lpl::procgen::buildSurfaceMesh(atlas, lpl::procgen::MapSurfaceStyle{}));
    check(first == again, "the same atlas meshes to the same fold");
    std::printf("     surface fold 0x%08X\n", first);

    lpl::procgen::MapSurfaceStyle heightStyle;
    heightStyle.shading = lpl::procgen::MapShading::Height;
    const lpl::core::u32 shaded = lpl::procgen::foldMapMesh(lpl::procgen::buildSurfaceMesh(atlas, heightStyle));
    check(shaded != first, "a different shading mode folds differently");
}

/**
 * @brief A dungeon is the surface of the void.
 *
 * One open cell gives one floor, four walls (all four neighbours are rock), one cap: six quads. Two adjacent cells
 * give 2 floors + 6 walls (the shared face is not a boundary) + 2 caps = 10 quads. Twelve would mean the shared wall
 * was drawn twice, which is the "inside of a solid" mistake that reads as noise rather than as a wall.
 *
 * Outside the map counts as rock, or a cave is open to nothing at the border.
 */
void checkDungeonIsTheSurfaceOfTheVoid()
{
    std::printf("\n-- the underground is the surface of the void --\n");
    lpl::procgen::WorldAtlas solid = slopedAtlas(kW, kD);
    solid.dungeon = lpl::procgen::DungeonMap{kW, kD, lpl::procgen::DungeonCell::Wall};
    check(lpl::procgen::buildDungeonMesh(solid, 4.0f).empty(), "solid rock meshes to nothing");

    lpl::procgen::WorldAtlas hollow = slopedAtlas(kW, kD);
    hollow.dungeon = lpl::procgen::DungeonMap{kW, kD, lpl::procgen::DungeonCell::Wall};
    hollow.dungeon.at(3u, 3u) = lpl::procgen::DungeonCell::Floor;
    const lpl::procgen::MapMesh one = lpl::procgen::buildDungeonMesh(hollow, 4.0f);
    check(one.size() == 6u * 6u, "one open cell gives floor + four walls + a cap");

    hollow.dungeon.at(4u, 3u) = lpl::procgen::DungeonCell::Floor;
    const lpl::procgen::MapMesh two = lpl::procgen::buildDungeonMesh(hollow, 4.0f);
    check(two.size() == 10u * 6u, "two adjacent cells do not wall themselves off from each other");

    lpl::procgen::WorldAtlas edge = slopedAtlas(kW, kD);
    edge.dungeon = lpl::procgen::DungeonMap{kW, kD, lpl::procgen::DungeonCell::Wall};
    edge.dungeon.at(0u, 0u) = lpl::procgen::DungeonCell::Floor;
    check(lpl::procgen::buildDungeonMesh(edge, 4.0f).size() == 6u * 6u,
          "a corner cell is still walled in on all four sides");
}

/**
 * @brief Voxels: interior faces are skipped.
 */
void checkSolidBlockShowsOnlyItsOutside(const lpl::procgen::MapMesh &cube)
{
    std::printf("\n-- a solid block shows only its outside --\n");
    check(cube.size() == 24u * 6u, "a 2x2x2 block has 24 outward faces, not 48");
}

/**
 * @brief One datum per footprint, not per column.
 *
 * And it is the FLAT one that is flat: every base vertex at the same height.
 */
void checkOneDatumPerFootprintNotPerColumn(const lpl::procgen::VoxelVolume &block,
                                           const lpl::procgen::WorldAtlas &atlas, const lpl::procgen::Rgb *palette,
                                           const lpl::procgen::MapMesh &cube)
{
    std::printf("\n-- a plan sits on one datum --\n");
    lpl::pmr::vector<float> flat(static_cast<lpl::core::usize>(kW) * kD, 0.0f);
    const lpl::procgen::MapMesh onDatum =
        lpl::procgen::buildVoxelMesh(block, atlas, 0.0f, palette, 2u, flat.data(), flat.size());
    check(lpl::procgen::foldMapMesh(onDatum) != lpl::procgen::foldMapMesh(cube),
          "a footprint datum moves the volume off the per-column ground");
    float lowestY = 1.0e9f;
    float highestY = -1.0e9f;
    for (lpl::core::usize i = 0u; i < onDatum.size(); ++i)
    {
        if (onDatum[i].y < lowestY)
            lowestY = onDatum[i].y;
        if (onDatum[i].y > highestY)
            highestY = onDatum[i].y;
    }
    check(highestY - lowestY == 2.0f, "on a flat datum the block is exactly two levels tall on a sloped world");
}

/**
 * @brief The palette is honoured: no palette means no mesh.
 */
void checkPaletteIsHonoured(const lpl::procgen::VoxelVolume &block, const lpl::procgen::WorldAtlas &atlas)
{
    std::printf("\n-- the palette --\n");
    check(lpl::procgen::buildVoxelMesh(block, atlas, 0.0f, nullptr, 0u).empty(),
          "no palette means no mesh, not a crash");
}

/**
 * @brief The atlas keeps what the snapshot throws away.
 *
 * The claim the editor's Atlas panel rests on: a world can be re-derived from its recipe WITH the intermediates a
 * game does not read. Unasserted, the panel would silently show empty layers the day a copy-out line went missing.
 *
 * The two are identical where they overlap: an atlas that regenerated differently would be a second source of truth
 * rather than a diagnostic view of the first.
 */
void checkAtlasKeepsWhatTheSnapshotThrowsAway()
{
    std::printf("\n-- an atlas is a snapshot, extended --\n");
    const lpl::procgen::WorldRecipe recipe = lpl::procgen::parityWorldRecipe();
    const lpl::procgen::WorldSnapshot lean = lpl::procgen::buildSnapshot(recipe, nullptr, nullptr);
    const lpl::procgen::WorldAtlas full = lpl::procgen::buildAtlas(recipe, nullptr, nullptr);

    check(full.width == lean.width && full.depth == lean.depth, "both describe the same world");
    bool sameHeights = full.height.width() == lean.height.width();
    for (lpl::core::u32 z = 0u; sameHeights && z < lean.depth; ++z)
        for (lpl::core::u32 x = 0u; x < lean.width; ++x)
            if (full.height.at(x, z).raw() != lean.height.at(x, z).raw())
            {
                sameHeights = false;
                break;
            }
    check(sameHeights, "the shared heightfield is bit-identical, not merely similar");

    check(!full.drainage.accumulation.empty(), "the atlas keeps the drainage the rivers came from");
    check(full.drainage.maxAccumulation > 0u, "with a real trunk, not an empty grid");
    check(!full.climate.empty(), "and the six climate axes the classifier read");
    std::printf("     trunk drains %u cells; %zu building plots retained\n", full.drainage.maxAccumulation,
                full.plots.size());
}

} // namespace

/**
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
 */
int main()
{
    std::printf("=== map mesh ===\n\n");

    const lpl::procgen::WorldAtlas atlas = slopedAtlas(kW, kD);
    const lpl::procgen::MapMesh surface = lpl::procgen::buildSurfaceMesh(atlas, lpl::procgen::MapSurfaceStyle{});

    checkSurfaceIsAGridOfQuads(surface);
    checkFoldIsDeterministicAndReadsTheShading(atlas, surface);
    checkDungeonIsTheSurfaceOfTheVoid();

    lpl::procgen::VoxelVolume block;
    block.width = 2u;
    block.depth = 2u;
    block.levels = 2u;
    block.cells.assign(8u, static_cast<lpl::core::u8>(1u));
    const lpl::procgen::Rgb palette[2] = {
        {0.0f, 0.0f, 0.0f},
        {0.5f, 0.5f, 0.5f}
    };
    const lpl::procgen::MapMesh cube = lpl::procgen::buildVoxelMesh(block, atlas, 0.0f, palette, 2u);

    checkSolidBlockShowsOnlyItsOutside(cube);
    checkOneDatumPerFootprintNotPerColumn(block, atlas, palette, cube);
    checkPaletteIsHonoured(block, atlas);
    checkAtlasKeepsWhatTheSnapshotThrowsAway();

    std::printf("\n%s\n", failures == 0 ? "ALL PASS (0 failures)" : "FAILURES");
    return failures == 0 ? 0 : 1;
}
