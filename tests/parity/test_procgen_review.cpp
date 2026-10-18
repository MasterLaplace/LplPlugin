#include <lpl/ecs/Registry.hpp>
#include <lpl/math/FixedMath.hpp>
#include <lpl/procgen/Biome.hpp>
#include <lpl/procgen/Chunking.hpp>
#include <lpl/procgen/Dungeon.hpp>
#include <lpl/procgen/Heightfield.hpp>
#include <lpl/procgen/LSystem.hpp>
#include <lpl/procgen/QualityGate.hpp>
#include <lpl/procgen/ValueNoise.hpp>
#include <lpl/procgen/Voronoi.hpp>
#include <lpl/procgen/WorldBuilder.hpp>
#include <lpl/procgen/WorldRecipe.hpp>

#include <cmath>
#include <cstdio>

namespace {

int g_failures = 0;

void check(bool condition, const char *label)
{
    std::printf("  %s: %s\n", condition ? "PASS" : "FAIL", label);
    if (!condition)
        ++g_failures;
}

/**
 * @brief Integer roots and logarithms: exact where they can be, monotone, and close where they cannot.
 *
 * The reason the integer logarithm exists: the drainage area of a large map exceeds what a Fixed32 can even hold, so
 * the compression has to happen in integers.
 */
void checkIntegerRootsAndLogarithmsAreExact()
{
    std::printf("-- integer roots and logarithms --\n");
    bool exact = true;
    for (lpl::core::u32 v = 0u; v < 5000u; ++v)
    {
        const lpl::core::u32 r = lpl::math::integerSqrt(v);
        exact = exact && r * r <= v && (r + 1u) * (r + 1u) > v;
    }
    check(exact, "math::integerSqrt returns the exact floor of the root");
    check(lpl::math::integerSqrt(0xFFFFFFFFu) == 65535u, "math::integerSqrt handles the full 32-bit range");

    check(lpl::math::fixedLog2(1u).raw() == 0, "log2(1) is zero");
    check(lpl::math::fixedLog2(1024u) == lpl::math::Fixed32::fromInt(10), "log2 is exact on powers of two");
    bool monotone = true;
    for (lpl::core::u32 v = 1u; v < 4096u; ++v)
        monotone = monotone && lpl::math::fixedLog2(v) <= lpl::math::fixedLog2(v + 1u);
    check(monotone, "log2 never decreases");

    lpl::math::Fixed32 rootError = lpl::math::Fixed32::zero();
    for (lpl::core::i32 v = 1; v < 400; ++v)
    {
        const lpl::math::Fixed32 x = lpl::math::Fixed32::fromInt(v);
        const lpl::math::Fixed32 r = lpl::math::fixedSqrt(x);
        const lpl::math::Fixed32 error = (r * r - x).abs();
        if (error > rootError)
            rootError = error;
    }
    check(rootError < lpl::math::Fixed32::fromFloat(0.05f), "math::fixedSqrt squares back to its input");
}

/**
 * @brief Noise constructions: the variants are differently SHAPED, not merely different.
 *
 * Ridged noise concentrates its mass near the crest value, which is what makes it read as a mountain range rather
 * than as hills, so its distribution has to be skewed where fBm's is symmetric.
 *
 * The skew is measured as skewness, the third standardised moment — NOT the share of cells above the mean, which is
 * what this used to measure and which cannot detect the property being claimed. Five octaves are five sums, and the
 * central limit theorem drives any sum toward a symmetric distribution: measured, all three kinds sit within a couple
 * of points of 0.5 above their own mean whatever their shape. The old form passed only because the noise lattice was
 * biased; once the lattice hash was fixed it read 0.500 for ridged against 0.515 for fBm and failed, correctly.
 *
 * And sharpness: the mean absolute discrete Laplacian, normalised by the field's range. This is the half of "reads as
 * a mountain range" that a moment cannot express — crests are a LOCAL property.
 *
 * Domain warping perturbs where the field is read, not what it returns, so the fold must move while the statistics
 * stay recognisable. Lacunarity and persistence have to be live parameters, not decoration.
 */
void checkNoiseConstructionsAreDifferentlyShaped()
{
    std::printf("\n-- noise constructions --\n");
    const auto profile = [](lpl::procgen::NoiseKind kind, lpl::core::f32 warp) {
        lpl::procgen::NoiseParams params;
        params.seed = 4242u;
        params.kind = kind;
        params.warpStrength = warp;
        return lpl::procgen::generateNoiseHeightfield(64u, 64u, params);
    };

    const lpl::procgen::Heightfield fbm = profile(lpl::procgen::NoiseKind::Fbm, 0.0f);
    const lpl::procgen::Heightfield ridged = profile(lpl::procgen::NoiseKind::Ridged, 0.0f);
    const lpl::procgen::Heightfield billow = profile(lpl::procgen::NoiseKind::Billow, 0.0f);

    check(lpl::procgen::foldHeightfield(fbm) != lpl::procgen::foldHeightfield(ridged), "ridged differs from fBm");
    check(lpl::procgen::foldHeightfield(billow) != lpl::procgen::foldHeightfield(ridged), "billow differs from ridged");

    const auto meanOf = [](const lpl::procgen::Heightfield &field) {
        double total = 0.0;
        for (lpl::core::u32 i = 0u; i < field.cellCount(); ++i)
            total += static_cast<double>(field[i].toFloat());
        return total / static_cast<double>(field.cellCount());
    };
    const auto skewness = [&meanOf](const lpl::procgen::Heightfield &field) {
        const double mean = meanOf(field);
        double variance = 0.0;
        double third = 0.0;
        for (lpl::core::u32 i = 0u; i < field.cellCount(); ++i)
        {
            const double d = static_cast<double>(field[i].toFloat()) - mean;
            variance += d * d;
            third += d * d * d;
        }
        variance /= static_cast<double>(field.cellCount());
        third /= static_cast<double>(field.cellCount());
        return variance > 0.0 ? third / (variance * std::sqrt(variance)) : 0.0;
    };

    const auto sharpness = [](const lpl::procgen::Heightfield &field) {
        double low = 1e30;
        double high = -1e30;
        for (lpl::core::u32 i = 0u; i < field.cellCount(); ++i)
        {
            const double v = static_cast<double>(field[i].toFloat());
            low = v < low ? v : low;
            high = v > high ? v : high;
        }
        double total = 0.0;
        lpl::core::u32 counted = 0u;
        for (lpl::core::u32 z = 1u; z + 1u < field.depth(); ++z)
            for (lpl::core::u32 x = 1u; x + 1u < field.width(); ++x)
            {
                const double centre = static_cast<double>(field.at(x, z).toFloat());
                const double laplacian = static_cast<double>(field.at(x - 1u, z).toFloat()) +
                                         static_cast<double>(field.at(x + 1u, z).toFloat()) +
                                         static_cast<double>(field.at(x, z - 1u).toFloat()) +
                                         static_cast<double>(field.at(x, z + 1u).toFloat()) - 4.0 * centre;
                total += laplacian < 0.0 ? -laplacian : laplacian;
                ++counted;
            }
        const double mean = counted > 0u ? total / static_cast<double>(counted) : 0.0;
        return high > low ? mean / (high - low) : 0.0;
    };

    const double fbmSkew = skewness(fbm);
    const double ridgedSkew = skewness(ridged);
    std::printf("    skew: fbm=%+.3f ridged=%+.3f | sharpness: fbm=%.4f ridged=%.4f\n", fbmSkew, ridgedSkew,
                sharpness(fbm), sharpness(ridged));
    check(ridgedSkew > fbmSkew + 0.10, "ridged noise is skewed toward its crests where fBm is symmetric");
    check(sharpness(ridged) > sharpness(fbm) * 1.5, "ridged noise carries sharper crests than fBm");
    check(skewness(billow) > fbmSkew + 0.05, "billow noise is skewed too, less than ridged");

    const lpl::procgen::Heightfield warped = profile(lpl::procgen::NoiseKind::Fbm, 6.0f);
    check(lpl::procgen::foldHeightfield(warped) != lpl::procgen::foldHeightfield(fbm),
          "domain warping changes the field");

    lpl::math::Fixed32 plainLow{}, plainHigh{}, warpLow{}, warpHigh{};
    (void) lpl::procgen::heightRange(fbm, plainLow, plainHigh);
    (void) lpl::procgen::heightRange(warped, warpLow, warpHigh);
    const double plainSpan = static_cast<double>((plainHigh - plainLow).toFloat());
    const double warpSpan = static_cast<double>((warpHigh - warpLow).toFloat());
    check(warpSpan > plainSpan * 0.6 && warpSpan < plainSpan * 1.6,
          "warping folds the domain without changing the amplitude");

    lpl::procgen::NoiseParams tuned;
    tuned.seed = 4242u;
    tuned.persistence = 0.8f;
    check(lpl::procgen::foldHeightfield(lpl::procgen::generateNoiseHeightfield(64u, 64u, tuned)) !=
              lpl::procgen::foldHeightfield(fbm),
          "persistence changes the terrain");
    tuned.persistence = 0.5f;
    tuned.lacunarity = 2.7f;
    check(lpl::procgen::foldHeightfield(lpl::procgen::generateNoiseHeightfield(64u, 64u, tuned)) !=
              lpl::procgen::foldHeightfield(fbm),
          "lacunarity changes the terrain");
}

/**
 * @brief Chunking agrees with the whole map.
 *
 * The point of routing both through one sampler. A chunk must be a window onto the same world function, or a
 * chunked world and an unchunked one of the same seed are two different worlds.
 *
 * A chunk with a warped layer must still be seamless: the warp is a pure function of the world coordinate, so it
 * cannot break at a boundary.
 */
void checkChunkingAgreesWithTheWholeMap()
{
    std::printf("\n-- chunking agrees with the whole map --\n");
    lpl::procgen::ChunkParams chunks;
    chunks.size = 16u;
    chunks.worldSeed = 909u;

    check(lpl::procgen::countSeamMismatches(chunks, lpl::procgen::ChunkCoord{0, 0}, lpl::procgen::ChunkCoord{1, 0}) ==
              0u,
          "east seam is exact");
    check(lpl::procgen::countSeamMismatches(chunks, lpl::procgen::ChunkCoord{0, 0}, lpl::procgen::ChunkCoord{0, 1}) ==
              0u,
          "south seam is exact");
    check(lpl::procgen::countSeamMismatches(chunks, lpl::procgen::ChunkCoord{3, -2}, lpl::procgen::ChunkCoord{2, -2}) ==
              0u,
          "seams hold at negative coordinates too");

    chunks.noise.warpStrength = 4.0f;
    check(lpl::procgen::countSeamMismatches(chunks, lpl::procgen::ChunkCoord{5, 5}, lpl::procgen::ChunkCoord{6, 5}) ==
              0u,
          "a warped layer stays seamless across chunks");
    chunks.noise.kind = lpl::procgen::NoiseKind::Ridged;
    check(lpl::procgen::countSeamMismatches(chunks, lpl::procgen::ChunkCoord{5, 5}, lpl::procgen::ChunkCoord{5, 6}) ==
              0u,
          "a ridged layer stays seamless across chunks");

    check(lpl::procgen::chunkSeed(chunks, lpl::procgen::ChunkCoord{1, 0}) !=
              lpl::procgen::chunkSeed(chunks, lpl::procgen::ChunkCoord{0, 1}),
          "transposed chunk coordinates do not share a seed");
}

/**
 * @brief Voronoi metrics and warping change the SHAPE of a region, not just its labels.
 *
 * The property that makes Manhattan the city-block metric: its borders run along the axes, so a border cell is far
 * more likely to have an axial neighbour in a different region than a Euclidean one is. A cell on a straight run of
 * border differs on one axis only; the share of such cells is printed for each metric.
 *
 * The u16 sentinel guard: a partition finer than 65535 regions would wrap ids into "no region" and silently leave
 * cells unclaimed.
 */
void checkVoronoiMetricsChangeTheShapeOfRegions()
{
    std::printf("\n-- voronoi metrics and warping --\n");
    lpl::procgen::VoronoiParams params;
    params.width = 64u;
    params.depth = 64u;
    params.seed = 77u;
    params.cellSize = 10u;

    params.metric = lpl::procgen::DistanceMetric::Euclidean;
    const lpl::procgen::VoronoiDiagram euclidean = lpl::procgen::computeVoronoi(params);
    params.metric = lpl::procgen::DistanceMetric::Manhattan;
    const lpl::procgen::VoronoiDiagram manhattan = lpl::procgen::computeVoronoi(params);
    params.metric = lpl::procgen::DistanceMetric::Chebyshev;
    const lpl::procgen::VoronoiDiagram chebyshev = lpl::procgen::computeVoronoi(params);

    check(lpl::procgen::foldRegionMap(euclidean.regions) != lpl::procgen::foldRegionMap(manhattan.regions),
          "the Manhattan metric repartitions the space");
    check(lpl::procgen::foldRegionMap(manhattan.regions) != lpl::procgen::foldRegionMap(chebyshev.regions),
          "the Chebyshev metric differs again");
    check(euclidean.regionCount == manhattan.regionCount, "the metric changes region shapes, not how many there are");

    const auto axialBorderShare = [](const lpl::procgen::VoronoiDiagram &diagram) {
        const lpl::procgen::Grid<lpl::core::u8> borders = lpl::procgen::regionBorders(diagram);
        lpl::core::u32 total = 0u;
        lpl::core::u32 axial = 0u;
        for (lpl::core::u32 z = 1u; z + 1u < diagram.regions.depth(); ++z)
            for (lpl::core::u32 x = 1u; x + 1u < diagram.regions.width(); ++x)
            {
                if (borders.at(x, z) == 0u)
                    continue;
                ++total;
                const lpl::core::u16 here = diagram.regions.at(x, z);
                const bool horizontal = diagram.regions.at(x - 1u, z) != here || diagram.regions.at(x + 1u, z) != here;
                const bool vertical = diagram.regions.at(x, z - 1u) != here || diagram.regions.at(x, z + 1u) != here;
                if (horizontal != vertical)
                    ++axial;
            }
        return total ? static_cast<double>(axial) / static_cast<double>(total) : 0.0;
    };
    std::printf("    straight-border share: euclidean=%.3f manhattan=%.3f chebyshev=%.3f\n",
                axialBorderShare(euclidean), axialBorderShare(manhattan), axialBorderShare(chebyshev));

    params.metric = lpl::procgen::DistanceMetric::Euclidean;
    params.warpStrength = 5.0f;
    const lpl::procgen::VoronoiDiagram warped = lpl::procgen::computeVoronoi(params);
    check(lpl::procgen::foldRegionMap(warped.regions) != lpl::procgen::foldRegionMap(euclidean.regions),
          "domain warping folds the region borders");
    bool everyCellClaimed = true;
    for (lpl::core::u32 i = 0u; i < warped.regions.cellCount(); ++i)
        everyCellClaimed = everyCellClaimed && warped.regions[i] != lpl::procgen::kNoRegion;
    check(everyCellClaimed, "a warped diagram still claims every cell");

    lpl::procgen::VoronoiParams excessive;
    excessive.width = 512u;
    excessive.depth = 512u;
    excessive.cellSize = 1u;
    check(lpl::procgen::computeVoronoi(excessive).regionCount == 0u,
          "a partition too fine for a u16 id is refused, not wrapped");
}

/**
 * @brief A world past what a Fixed32 counts.
 *
 * 65,536 cells: a cell count as a Fixed32 wraps to zero, and a share of the map divided by it traps; 33,124 cells
 * wrap it negative.
 */
void checkAWorldPastWhatAFixed32Counts()
{
    std::printf("\n-- a world past what a Fixed32 counts --\n");
    for (const lpl::core::u32 side : {256u, 182u})
    {
        lpl::procgen::WorldRecipe recipe = lpl::procgen::parityWorldRecipe();
        recipe.width = side;
        recipe.depth = side;
        lpl::ecs::Registry first;
        lpl::ecs::Registry second;
        const lpl::procgen::WorldRecipeResult baked = lpl::procgen::bakeWorld(first, recipe);
        check(baked.entityCount > 0u && baked.biomeSignature != 0u, "a world past 32,767 cells bakes");
        check(baked == lpl::procgen::bakeWorld(second, recipe), "and bakes the same twice");
    }
}

/**
 * @brief Tensor fields steer the turtle.
 *
 * The two regions must genuinely disagree, or blending them is pointless.
 *
 * Conformity zero must reproduce the unsteered walk exactly: the field is an influence, and an influence of zero is
 * not an influence.
 */
void checkTensorFieldsSteerTheTurtle()
{
    std::printf("\n-- tensor fields steer the turtle --\n");
    lpl::pmr::vector<lpl::procgen::FieldRegion> influences;
    lpl::procgen::FieldRegion downtown;
    downtown.pattern = lpl::procgen::FieldPattern::Radial;
    downtown.centerX = 32u;
    downtown.centerZ = 32u;
    downtown.strength = 1.0f;
    downtown.falloff = 0.02f;
    influences.push_back(downtown);

    lpl::procgen::FieldRegion suburb;
    suburb.pattern = lpl::procgen::FieldPattern::Grid;
    suburb.centerX = 8u;
    suburb.centerZ = 56u;
    suburb.bearing = 0u;
    suburb.strength = 1.0f;
    suburb.falloff = 0.03f;
    influences.push_back(suburb);

    const lpl::procgen::HeadingField field = lpl::procgen::bakeHeadingField(64u, 64u, influences);
    check(!field.empty(), "the field bakes");

    lpl::core::u32 distinctHeadings = 0u;
    bool seen[lpl::procgen::kTurtleDirections] = {};
    for (lpl::core::u32 i = 0u; i < field.cellCount(); ++i)
        if (!seen[field[i]])
        {
            seen[field[i]] = true;
            ++distinctHeadings;
        }
    check(distinctHeadings >= 8u, "a radial region produces headings all the way round");

    const lpl::procgen::LSystemParams grammar = lpl::procgen::makeRoadGrammar();
    const lpl::pmr::string expanded = lpl::procgen::expandLSystem(grammar);

    lpl::procgen::TurtleParams turtle;
    turtle.startX = 32u;
    turtle.startZ = 58u;
    turtle.stepLength = 3u;

    lpl::procgen::Grid<lpl::core::u8> free{64u, 64u, 0u};
    lpl::procgen::Grid<lpl::core::u8> steered{64u, 64u, 0u};
    const lpl::core::u32 freeCells = lpl::procgen::drawTurtle(expanded, turtle, free);
    const lpl::core::u32 steeredCells = lpl::procgen::drawTurtleInField(expanded, turtle, field, 0.7f, steered);

    check(freeCells > 0u && steeredCells > 0u, "both turtles draw something");
    bool differs = false;
    for (lpl::core::u32 i = 0u; i < free.cellCount() && !differs; ++i)
        differs = free[i] != steered[i];
    check(differs, "the field changes where the roads go");

    lpl::procgen::Grid<lpl::core::u8> unsteered{64u, 64u, 0u};
    (void) lpl::procgen::drawTurtleInField(expanded, turtle, field, 0.0f, unsteered);
    bool identical = true;
    for (lpl::core::u32 i = 0u; i < free.cellCount(); ++i)
        identical = identical && free[i] == unsteered[i];
    check(identical, "conformity of zero reproduces the plain turtle exactly");
}

/**
 * @brief Weighted grammar alternatives: the seed picks among them, and a single-rule grammar ignores it.
 *
 * A grammar with one rule per symbol must stay a pure function, so an existing single-rule grammar keeps behaving
 * exactly as before.
 */
void checkWeightedGrammarAlternativesFollowTheSeed()
{
    std::printf("\n-- weighted grammar alternatives --\n");
    lpl::procgen::LSystemParams grammar = lpl::procgen::makeBranchingGrammar();
    grammar.seed = 11u;
    const lpl::pmr::string a = lpl::procgen::expandLSystem(grammar);
    grammar.seed = 12u;
    const lpl::pmr::string b = lpl::procgen::expandLSystem(grammar);
    grammar.seed = 11u;
    const lpl::pmr::string again = lpl::procgen::expandLSystem(grammar);

    check(a == again, "the same seed regrows the same structure");
    check(a != b, "a different seed picks different alternatives");

    lpl::procgen::LSystemParams single;
    single.axiom = "F";
    single.rules.push_back(lpl::procgen::LRule{'F', lpl::pmr::string{"F+F"}});
    single.iterations = 4u;
    single.seed = 1u;
    const lpl::pmr::string one = lpl::procgen::expandLSystem(single);
    single.seed = 999u;
    check(one == lpl::procgen::expandLSystem(single), "a single-rule grammar ignores the seed");
}

/**
 * @brief The analysis FURNISHES the level, not just judges it.
 *
 * Both answers were already computed and neither was used by anything: the spine is where the player certainly goes,
 * so an encounter placed on it is met rather than missed, and the deepest dead ends are where something is worth hiding
 * for the same reason they were architectural excrescences.
 *
 * A prize the player walks over by accident is litter, not a reward, so every reward is a dead end. Encounters
 * spread through the ROUTE, not through the grid: the whole point of measuring progress rather than counting cells.
 *
 * Furnishing is deterministic, because a furnished level is content and content is folded. A capacity of one is not
 * an error: it is a caller with room for one spot.
 */
void checkTheAnalysisFurnishesTheLevel(const lpl::procgen::DungeonMap &level, const lpl::procgen::HotPathAnalysis &hot,
                                       lpl::core::u32 startX, lpl::core::u32 startZ)
{
    lpl::procgen::PlacementParams furnish;
    furnish.encounters = 4u;
    furnish.rewards = 3u;
    furnish.minSpacing = 3u;
    furnish.rewardMinDetour = 2u;
    lpl::procgen::Placement spots[16]{};
    const lpl::core::u32 placed = lpl::procgen::placeAlongHotPath(level, hot, startX, startZ, furnish, spots, 16u);
    std::printf("    furnished %u spots\n", placed);
    check(placed > 0u, "the level gets furnished from its own measurements");

    lpl::core::u32 encounters = 0u;
    lpl::core::u32 rewards = 0u;
    bool encountersOnSpine = true;
    bool rewardsOffSpine = true;
    bool rewardsAreDeadEnds = true;
    for (lpl::core::u32 i = 0u; i < placed; ++i)
    {
        const lpl::core::u32 cell = level.index(spots[i].x, spots[i].z);
        if (spots[i].role == lpl::procgen::PlacementRole::Encounter)
        {
            ++encounters;
            encountersOnSpine = encountersOnSpine && hot.onPath[cell] != 0u;
        }
        else
        {
            ++rewards;
            rewardsOffSpine = rewardsOffSpine && spots[i].detour >= furnish.rewardMinDetour;
            lpl::core::u32 adjacent = 0u;
            for (lpl::core::u32 n = 0u; n < 4u; ++n)
            {
                const lpl::core::i32 nx = static_cast<lpl::core::i32>(spots[i].x) + lpl::procgen::kNeighbor4X[n];
                const lpl::core::i32 nz = static_cast<lpl::core::i32>(spots[i].z) + lpl::procgen::kNeighbor4Z[n];
                if (level.contains(nx, nz) && lpl::procgen::isWalkable(level.at(static_cast<lpl::core::u32>(nx),
                                                                                static_cast<lpl::core::u32>(nz))))
                    ++adjacent;
            }
            rewardsAreDeadEnds = rewardsAreDeadEnds && adjacent == 1u;
        }
    }
    std::printf("    %u encounters on the spine, %u rewards off it\n", encounters, rewards);
    check(encounters > 0u && encountersOnSpine, "every encounter sits ON the critical path");
    check(rewards > 0u && rewardsOffSpine, "every reward sits off it, past the threshold");
    check(rewardsAreDeadEnds, "and every reward is a dead end, not a cell in a room");

    lpl::core::u32 lastProgress = 0u;
    bool spreads = true;
    for (lpl::core::u32 i = 0u; i < placed; ++i)
    {
        if (spots[i].role != lpl::procgen::PlacementRole::Encounter)
            continue;
        spreads = spreads && spots[i].progress > lastProgress;
        lastProgress = spots[i].progress;
    }
    check(spreads, "encounters advance through the player's progress, in order");

    lpl::procgen::Placement again[16]{};
    const lpl::core::u32 twice = lpl::procgen::placeAlongHotPath(level, hot, startX, startZ, furnish, again, 16u);
    bool identical = twice == placed;
    for (lpl::core::u32 i = 0u; i < placed && identical; ++i)
        identical = again[i].x == spots[i].x && again[i].z == spots[i].z && again[i].role == spots[i].role &&
                    again[i].detour == spots[i].detour;
    check(identical, "furnishing the same level twice gives the same spots");

    lpl::procgen::Placement single{};
    check(lpl::procgen::placeAlongHotPath(level, hot, startX, startZ, furnish, &single, 1u) == 1u,
          "a caller with room for one gets one");
}

/**
 * @brief Hot path analysis: the spine of a level, how far the rest lies from it, and a way to flee.
 *
 * Every cell on the path must be at detour zero, and no other cell may be: that is what makes the detour map a
 * measure of distance FROM the spine. The path must be a connected walk, not a scatter of cells.
 *
 * A flee map must not have a local minimum at a dead end: that is the corner-trapping it exists to remove.
 */
void checkHotPathAnalysisFindsTheSpineAndItsDeepestDetour()
{
    std::printf("\n-- hot path analysis --\n");
    lpl::procgen::CaveParams cave;
    cave.width = 56u;
    cave.depth = 56u;
    cave.seed = 4242u;
    const lpl::procgen::DungeonMap level = lpl::procgen::generateCellularCave(cave);

    lpl::core::u32 startX = 0u, startZ = 0u, goalX = 0u, goalZ = 0u;
    check(lpl::procgen::findFarthestPair(level, startX, startZ, goalX, goalZ), "the level has two ends");

    const lpl::procgen::HotPathAnalysis hot = lpl::procgen::analyseHotPath(level, startX, startZ, goalX, goalZ, 6u);
    check(hot.valid, "the hot path is found");
    check(hot.pathCells > 8u, "the spine has real length");

    bool consistent = true;
    for (lpl::core::u32 i = 0u; i < level.cellCount(); ++i)
    {
        const bool onPath = hot.onPath[i] != 0u;
        const bool atZero = hot.detour[i] == 0u;
        consistent = consistent && onPath == atZero;
    }
    check(consistent, "detour zero is exactly the spine");

    bool contiguous = true;
    lpl::core::u32 pathNeighbours = 0u;
    for (lpl::core::u32 z = 0u; z < level.depth(); ++z)
        for (lpl::core::u32 x = 0u; x < level.width(); ++x)
        {
            if (hot.onPath.at(x, z) == 0u)
                continue;
            lpl::core::u32 adjacent = 0u;
            for (lpl::core::u32 n = 0u; n < 4u; ++n)
            {
                const lpl::core::i32 nx = static_cast<lpl::core::i32>(x) + lpl::procgen::kNeighbor4X[n];
                const lpl::core::i32 nz = static_cast<lpl::core::i32>(z) + lpl::procgen::kNeighbor4Z[n];
                if (hot.onPath.contains(nx, nz) &&
                    hot.onPath.at(static_cast<lpl::core::u32>(nx), static_cast<lpl::core::u32>(nz)) != 0u)
                    ++adjacent;
            }
            pathNeighbours += adjacent;
            if (adjacent == 0u)
                contiguous = false;
        }
    check(contiguous, "the spine is a connected walk");

    check(hot.deepestDetour > 0u, "some of the level lies off the spine");
    check(hot.detour[hot.farthestCell] == hot.deepestDetour,
          "the reported cell really is the deepest — where a secret goes");
    std::printf("    spine=%u cells, deepest detour=%u, cells beyond the limit=%u\n", hot.pathCells, hot.deepestDetour,
                hot.excessiveCells);

    const lpl::procgen::DistanceMap danger = lpl::procgen::computeDistanceMap(level, startX, startZ);
    const lpl::procgen::DistanceMap flee = lpl::procgen::computeFleeMap(level, danger);

    lpl::core::u32 traps = 0u;
    for (lpl::core::u32 z = 0u; z < level.depth(); ++z)
        for (lpl::core::u32 x = 0u; x < level.width(); ++x)
        {
            const lpl::core::u32 cell = level.index(x, z);
            if (flee[cell] == lpl::procgen::kUnreachable || flee[cell] == 0u)
                continue;
            bool anyLower = false;
            for (lpl::core::u32 n = 0u; n < 4u; ++n)
            {
                const lpl::core::i32 nx = static_cast<lpl::core::i32>(x) + lpl::procgen::kNeighbor4X[n];
                const lpl::core::i32 nz = static_cast<lpl::core::i32>(z) + lpl::procgen::kNeighbor4Z[n];
                if (!level.contains(nx, nz))
                    continue;
                const lpl::core::u32 index =
                    level.index(static_cast<lpl::core::u32>(nx), static_cast<lpl::core::u32>(nz));
                if (flee[index] != lpl::procgen::kUnreachable && flee[index] < flee[cell])
                    anyLower = true;
            }
            if (!anyLower)
                ++traps;
        }
    check(traps == 0u, "the flee map has no local minimum to wedge an agent into");

    checkTheAnalysisFurnishesTheLevel(level, hot, startX, startZ);
}

/**
 * @brief Blue-noise scatter: a subset of its biome, reproducible, monotone in density, and filtered by moisture.
 *
 * Reproducibility covers the whole scatter, shuffle included. Density has to be monotone: asking for more must not
 * give less. A moisture window has to exclude something, or the filter is decoration.
 */
void checkBlueNoiseScatterIsReproducibleAndMonotone()
{
    std::printf("\n-- blue-noise scatter --\n");
    lpl::procgen::WorldBuilder builder{31337u};
    builder.terrain(96u, 96u).erode().rivers().biomes();

    lpl::procgen::ScatterRule rule;
    rule.biome = lpl::procgen::BiomeId::Grassland;
    rule.density = 0.1f;
    builder.scatter(rule);

    lpl::ecs::Registry registry;
    const lpl::procgen::BuiltWorldStats stats = builder.materialize(registry);
    check(stats.propEntities > 0u, "props were placed");

    lpl::procgen::Grid<lpl::core::u8> occupied{96u, 96u, 0u};
    lpl::core::u32 propCells = 0u;
    const lpl::procgen::BiomeMap &biomes = builder.biomeMap();
    for (lpl::core::u32 i = 0u; i < biomes.cellCount(); ++i)
        if (biomes[i] == lpl::procgen::BiomeId::Grassland)
            ++propCells;
    check(propCells > stats.propEntities, "props are a subset of their biome, not all of it");

    lpl::procgen::WorldBuilder twin{31337u};
    twin.terrain(96u, 96u).erode().rivers().biomes().scatter(rule);
    lpl::ecs::Registry twinRegistry;
    const lpl::procgen::BuiltWorldStats twinStats = twin.materialize(twinRegistry);
    check(twinStats.propEntities == stats.propEntities, "the scatter is reproducible");

    lpl::procgen::ScatterRule denser = rule;
    denser.density = 0.3f;
    lpl::procgen::WorldBuilder thick{31337u};
    thick.terrain(96u, 96u).erode().rivers().biomes().scatter(denser);
    lpl::ecs::Registry thickRegistry;
    const lpl::procgen::BuiltWorldStats thickStats = thick.materialize(thickRegistry);
    check(thickStats.propEntities > stats.propEntities, "a higher density places more props");
    std::printf("    density 0.10 -> %u props, density 0.30 -> %u props, over %u eligible cells\n", stats.propEntities,
                thickStats.propEntities, propCells);

    lpl::procgen::ScatterRule wetOnly = rule;
    wetOnly.minMoisture = 0.75f;
    lpl::procgen::WorldBuilder marshy{31337u};
    marshy.terrain(96u, 96u).erode().rivers().biomes().scatter(wetOnly);
    lpl::ecs::Registry marshyRegistry;
    const lpl::procgen::BuiltWorldStats marshyStats = marshy.materialize(marshyRegistry);
    check(marshyStats.propEntities < stats.propEntities, "a moisture window rejects dry ground");
}

/**
 * @brief The builder reaches the whole module.
 *
 * A gate nobody can satisfy has to fail, or it is not a gate. Validating with no dungeon must not claim success for
 * a level that was never generated.
 */
void checkTheBuilderReachesTheWholeModule()
{
    std::printf("\n-- the builder reaches the whole module --\n");
    lpl::procgen::WorldBuilder builder{5150u};
    builder.terrain(48u, 48u).erode().rivers().biomes();

    const lpl::procgen::TileSet tiles = lpl::procgen::makeTerrainTileSet();
    lpl::procgen::WfcParams wfc;
    wfc.maxAttempts = 4u;
    builder.tiles(tiles, wfc);
    check(!builder.tileMap().empty(), "the builder can solve tiles over its own terrain");
    check(builder.tileMap().width() == 48u, "the tile solve took its extent from the terrain");
    check(lpl::procgen::countAdjacencyViolations(builder.tileMap(), tiles) == 0u,
          "the tile solve respects the rules while pinned to the biome map");

    lpl::procgen::CaveParams cave;
    cave.width = 48u;
    cave.depth = 48u;
    builder.caves(cave).validate(lpl::procgen::GateCriteria{});
    check(builder.lastQuality().walkableCells > 0u, "the gate measured the cave");
    check(builder.gatePassed(), "a default cave passes the default criteria");

    lpl::procgen::GateCriteria impossible;
    impossible.minPathLength = 100000u;
    builder.validate(impossible);
    check(!builder.gatePassed(), "an unsatisfiable criterion is reported as a failure");

    lpl::procgen::WorldBuilder bare{1u};
    bare.terrain(32u, 32u).validate(lpl::procgen::GateCriteria{});
    check(!bare.gatePassed(), "validating nothing is not a pass");
}

} // namespace

/**
 * @brief The properties the module gained when it was audited against the surveys.
 *
 * Every check here corresponds to something the reference material asks for that
 * the implementation did not do, and each is written to fail if the behaviour is
 * ever removed. The four that matter most are not "does it fold the same twice" —
 * a broken pass folds beautifully — but measurements of whether the algorithm is
 * doing the thing it is named after:
 *
 *  - noise variants must produce genuinely different profiles, not the same field
 *    scaled;
 *  - a Voronoi metric must change the SHAPE of a region, not just its labels;
 *  - a tensor field must actually steer the turtle;
 *  - and the hot-path analysis must find the far corner of a level, because that
 *    is simultaneously the diagnosis of a bad layout and the place to hide a
 *    reward.
 */
int main()
{
    std::printf("== procgen review: the properties the surveys ask for ==\n\n");

    checkIntegerRootsAndLogarithmsAreExact();
    checkNoiseConstructionsAreDifferentlyShaped();
    checkChunkingAgreesWithTheWholeMap();
    checkVoronoiMetricsChangeTheShapeOfRegions();
    checkAWorldPastWhatAFixed32Counts();
    checkTensorFieldsSteerTheTurtle();
    checkWeightedGrammarAlternativesFollowTheSeed();
    checkHotPathAnalysisFindsTheSpineAndItsDeepestDetour();
    checkBlueNoiseScatterIsReproducibleAndMonotone();
    checkTheBuilderReachesTheWholeModule();

    std::printf("\n%s (%d failures)\n", g_failures == 0 ? "ALL PASS" : "FAILURES", g_failures);
    return g_failures == 0 ? 0 : 1;
}
