#include <lpl/ecs/Registry.hpp>
#include <lpl/procgen/WorldAtlas.hpp>
#include <lpl/procgen/WorldRecipe.hpp>

#include <cstdio>

namespace {

int g_failures = 0;

void check(bool condition, const char *label)
{
    std::printf("  %s: %s\n", condition ? "PASS" : "FAIL", label);
    if (!condition)
        ++g_failures;
}

lpl::procgen::WorldRecipeResult bakeWith(const lpl::procgen::WorldRecipe &recipe)
{
    lpl::ecs::Registry registry;
    return lpl::procgen::bakeWorld(registry, recipe);
}

/**
 * @brief Terracing reshapes the terrain.
 */
void checkTerracingReshapesTheTerrain(const lpl::procgen::WorldRecipeResult &baked)
{
    lpl::procgen::WorldRecipe stepped = lpl::procgen::parityWorldRecipe();
    stepped.terraceSteps = 6u;
    const lpl::procgen::WorldRecipeResult result = bakeWith(stepped);
    check(result.heightSignature != baked.heightSignature, "terracing reshapes the terrain");
}

/**
 * @brief Provinces partition the surface without moving it.
 *
 * So the HEIGHT must not budge — that is the claim, and a pass that moved it would be a bug.
 */
void checkProvincesDoNotDisturbTheTerrain(const lpl::procgen::WorldRecipeResult &baked)
{
    lpl::procgen::WorldRecipe divided = lpl::procgen::parityWorldRecipe();
    divided.partitionRegions = true;
    divided.provinces.width = divided.width;
    divided.provinces.depth = divided.depth;
    divided.provinces.cellSize = 6u;
    const lpl::procgen::WorldRecipeResult result = bakeWith(divided);
    check(result.heightSignature == baked.heightSignature, "provinces do not disturb the terrain");
    check(result.ok == 1u, "and the world still passes its gate");
}

/**
 * @brief Every alternative underground must actually DIG something.
 *
 * The check has to read the grid each generator fills rather than one summary field. The first version of this
 * asked bakeWorld for `dungeonFloor` and passed for the wrong reason: the layered system fills its own CaveSystem
 * and leaves `_dungeon` empty, so it reported ZERO open cells and still counted as "different from the default". A
 * check that cannot fail for the right reason is worse than no check.
 *
 * The playability gate is left off in these recipes: leaving it on here would test the gate rather than the
 * generator.
 */
void checkEveryUndergroundDigsItsOwnCave()
{
    const auto atlasFor = [](lpl::procgen::CaveKind kind) {
        lpl::procgen::WorldRecipe underground = lpl::procgen::parityWorldRecipe();
        underground.caveKind = kind;
        underground.rooms.width = underground.width;
        underground.rooms.depth = underground.depth;
        underground.aggregation.width = underground.width;
        underground.aggregation.depth = underground.depth;
        underground.aggregation.particles = underground.width * 8u;
        underground.caveSystem.width = underground.width;
        underground.caveSystem.depth = underground.depth;
        underground.checkPlayability = false;
        return lpl::procgen::buildAtlas(underground, nullptr, nullptr);
    };

    lpl::core::u32 openFor[3] = {0u, 0u, 0u};
    const lpl::procgen::CaveKind kinds[3] = {lpl::procgen::CaveKind::Bsp, lpl::procgen::CaveKind::Dla,
                                             lpl::procgen::CaveKind::Layered};
    for (lpl::core::u32 i = 0u; i < 3u; ++i)
    {
        const lpl::procgen::WorldAtlas atlas = atlasFor(kinds[i]);
        if (kinds[i] == lpl::procgen::CaveKind::Layered)
            openFor[i] = atlas.caveSystem.hollowCells;
        else
            for (lpl::core::u32 z = 0u; z < atlas.dungeon.depth(); ++z)
                for (lpl::core::u32 x = 0u; x < atlas.dungeon.width(); ++x)
                    if (lpl::procgen::isWalkable(atlas.dungeon.at(x, z)))
                        ++openFor[i];
        std::printf("  %-9s open cells = %u\n", lpl::procgen::caveKindName(kinds[i]), openFor[i]);
    }
    check(openFor[0] > 0u, "the room partition digs rooms");
    check(openFor[1] > 0u, "the aggregation digs a branching cave");
    check(openFor[2] > 0u, "the layered system digs layers");
    check(openFor[0] != openFor[1], "and they are not the same cave");
}

/**
 * @brief The grammar raises voxels; it does not move the ground or the plots.
 */
void checkRaisingATownDoesNotMoveTheGround(const lpl::procgen::WorldRecipeResult &baked)
{
    lpl::procgen::WorldRecipe raised = lpl::procgen::parityWorldRecipe();
    raised.raiseBuildings = true;
    const lpl::procgen::WorldRecipeResult result = bakeWith(raised);
    check(result.heightSignature == baked.heightSignature, "raising a town does not move the ground");
    check(result.settlementPlots == baked.settlementPlots, "nor change how many plots there are");
}

/**
 * @brief The passes a recipe could not name until now.
 *
 * Four underground generators existed and a recipe could ask for exactly one of them; provinces, terraces, the
 * building grammar and the roadside decoration could not be asked for at all. They were reachable only by writing
 * WorldBuilder calls by hand, which is what a viewer did — so its world could not be saved, baked, replayed in ring
 * 0, or asked for by an intelligence.
 *
 * Each check is the anti-orphan proof for one of them: a switch that does not change the world it claims to change
 * is a field, not a feature.
 */
void checkThePassesARecipeCouldNotName(const lpl::procgen::WorldRecipeResult &baked)
{
    std::printf("\n-- the passes a recipe could not name --\n");

    checkTerracingReshapesTheTerrain(baked);
    checkProvincesDoNotDisturbTheTerrain(baked);
    checkEveryUndergroundDigsItsOwnCave();
    checkRaisingATownDoesNotMoveTheGround(baked);
}

/**
 * @brief A word, never an index.
 */
void checkEveryKindIsReadBackByItsName()
{
    std::printf("\n-- named, so a reordering cannot reinterpret a document --\n");
    for (lpl::core::u32 i = 0u; i <= static_cast<lpl::core::u32>(lpl::procgen::CaveKind::Layered); ++i)
    {
        const lpl::procgen::CaveKind kind = static_cast<lpl::procgen::CaveKind>(i);
        lpl::procgen::CaveKind round = lpl::procgen::CaveKind::Cellular;
        if (!lpl::procgen::caveKindByName(lpl::procgen::caveKindName(kind), round) || round != kind)
        {
            check(false, "every cave kind round-trips through its own name");
            break;
        }
        if (i == static_cast<lpl::core::u32>(lpl::procgen::CaveKind::Layered))
            check(true, "every cave kind round-trips through its own name");
    }
    lpl::procgen::CaveKind rejected = lpl::procgen::CaveKind::Layered;
    check(!lpl::procgen::caveKindByName("spelunk", rejected), "an unknown word is refused, not defaulted");
    check(rejected == lpl::procgen::CaveKind::Layered, "and the caller's value is left alone");
    lpl::procgen::DistanceMetric metric = lpl::procgen::DistanceMetric::Euclidean;
    check(lpl::procgen::distanceMetricByName("chebyshev", metric) && metric == lpl::procgen::DistanceMetric::Chebyshev,
          "province metrics are named too");
}

} // namespace

/**
 * @brief The passes a recipe can name change the world they claim to change, and each name reads
 *        back as the pass it names.
 *
 * The parity world itself, baked from the reference pack on both targets, is the engine test
 * `world_recipe` (tests/procgen/world_recipe.cpp).
 */
int main()
{
    std::printf("== procedural world recipe: the passes it can name ==\n");

    const lpl::procgen::WorldRecipe recipe = lpl::procgen::parityWorldRecipe();
    lpl::ecs::Registry world;
    const lpl::procgen::WorldRecipeResult baked = lpl::procgen::bakeWorld(world, recipe);

    checkThePassesARecipeCouldNotName(baked);
    checkEveryKindIsReadBackByItsName();

    std::printf("\n%s (%d failures)\n", g_failures == 0 ? "ALL PASS" : "FAILURES", g_failures);
    return g_failures == 0 ? 0 : 1;
}
