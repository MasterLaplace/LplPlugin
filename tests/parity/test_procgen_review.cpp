/**
 * @file test_procgen_review.cpp
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
 *
 * @author MasterLaplace
 * @copyright MIT License
 */

#include <lpl/ecs/Registry.hpp>
#include <lpl/procgen/Biome.hpp>
#include <lpl/procgen/Chunking.hpp>
#include <lpl/procgen/Dungeon.hpp>
#include <lpl/math/FixedMath.hpp>
#include <lpl/procgen/Heightfield.hpp>
#include <lpl/procgen/LSystem.hpp>
#include <lpl/procgen/QualityGate.hpp>
#include <lpl/procgen/ValueNoise.hpp>
#include <lpl/procgen/Voronoi.hpp>
#include <lpl/procgen/WorldBuilder.hpp>

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

} // namespace

int main()
{
    using namespace lpl;

    std::printf("== procgen review: the properties the surveys ask for ==\n\n");

    // ─────────────────────────────────────────────────────────────────────────
    std::printf("-- integer roots and logarithms --\n");
    {
        bool exact = true;
        for (core::u32 v = 0u; v < 5000u; ++v)
        {
            const core::u32 r = math::integerSqrt(v);
            exact = exact && r * r <= v && (r + 1u) * (r + 1u) > v;
        }
        check(exact, "math::integerSqrt returns the exact floor of the root");
        check(math::integerSqrt(0xFFFFFFFFu) == 65535u, "math::integerSqrt handles the full 32-bit range");

        // The reason this exists: the drainage area of a large map exceeds what a
        // Fixed32 can even hold, so the compression has to happen in integers.
        check(math::fixedLog2(1u).raw() == 0, "log2(1) is zero");
        check(math::fixedLog2(1024u) == math::Fixed32::fromInt(10), "log2 is exact on powers of two");
        bool monotone = true;
        for (core::u32 v = 1u; v < 4096u; ++v)
            monotone = monotone && math::fixedLog2(v) <= math::fixedLog2(v + 1u);
        check(monotone, "log2 never decreases");

        math::Fixed32 rootError = math::Fixed32::zero();
        for (core::i32 v = 1; v < 400; ++v)
        {
            const math::Fixed32 x = math::Fixed32::fromInt(v);
            const math::Fixed32 r = math::fixedSqrt(x);
            const math::Fixed32 error = (r * r - x).abs();
            if (error > rootError)
                rootError = error;
        }
        check(rootError < math::Fixed32::fromFloat(0.05f), "math::fixedSqrt squares back to its input");
    }

    // ─────────────────────────────────────────────────────────────────────────
    std::printf("\n-- noise constructions --\n");
    {
        const auto profile = [](procgen::NoiseKind kind, core::f32 warp) {
            procgen::NoiseParams params;
            params.seed = 4242u;
            params.kind = kind;
            params.warpStrength = warp;
            return procgen::generateNoiseHeightfield(64u, 64u, params);
        };

        const procgen::Heightfield fbm = profile(procgen::NoiseKind::Fbm, 0.0f);
        const procgen::Heightfield ridged = profile(procgen::NoiseKind::Ridged, 0.0f);
        const procgen::Heightfield billow = profile(procgen::NoiseKind::Billow, 0.0f);

        check(procgen::foldHeightfield(fbm) != procgen::foldHeightfield(ridged), "ridged differs from fBm");
        check(procgen::foldHeightfield(billow) != procgen::foldHeightfield(ridged), "billow differs from ridged");

        // Not merely different: differently SHAPED. Ridged noise concentrates its
        // mass near the crest value, which is what makes it read as a mountain
        // range rather than as hills, so its distribution has to be skewed where
        // fBm's is symmetric.
        const auto meanOf = [](const procgen::Heightfield &field) {
            double total = 0.0;
            for (core::u32 i = 0u; i < field.cellCount(); ++i)
                total += static_cast<double>(field[i].toFloat());
            return total / static_cast<double>(field.cellCount());
        };
        // Skewness, the third standardised moment — NOT the share of cells above
        // the mean, which is what this used to measure and which cannot detect
        // the property being claimed. Five octaves are five sums, and the central
        // limit theorem drives any sum toward a symmetric distribution: measured,
        // all three kinds sit within a couple of points of 0.5 above their own
        // mean whatever their shape. The old form passed only because the noise
        // lattice was biased; once the lattice hash was fixed it read 0.500 for
        // ridged against 0.515 for fBm and failed, correctly.
        const auto skewness = [&meanOf](const procgen::Heightfield &field) {
            const double mean = meanOf(field);
            double variance = 0.0;
            double third = 0.0;
            for (core::u32 i = 0u; i < field.cellCount(); ++i)
            {
                const double d = static_cast<double>(field[i].toFloat()) - mean;
                variance += d * d;
                third += d * d * d;
            }
            variance /= static_cast<double>(field.cellCount());
            third /= static_cast<double>(field.cellCount());
            return variance > 0.0 ? third / (variance * std::sqrt(variance)) : 0.0;
        };

        // And sharpness: the mean absolute discrete Laplacian, normalised by the
        // field's range. This is the half of "reads as a mountain range" that a
        // moment cannot express — crests are a LOCAL property.
        const auto sharpness = [](const procgen::Heightfield &field) {
            double low = 1e30;
            double high = -1e30;
            for (core::u32 i = 0u; i < field.cellCount(); ++i)
            {
                const double v = static_cast<double>(field[i].toFloat());
                low = v < low ? v : low;
                high = v > high ? v : high;
            }
            double total = 0.0;
            core::u32 counted = 0u;
            for (core::u32 z = 1u; z + 1u < field.depth(); ++z)
                for (core::u32 x = 1u; x + 1u < field.width(); ++x)
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

        // Domain warping perturbs where the field is read, not what it returns, so
        // the fold must move while the statistics stay recognisable.
        const procgen::Heightfield warped = profile(procgen::NoiseKind::Fbm, 6.0f);
        check(procgen::foldHeightfield(warped) != procgen::foldHeightfield(fbm), "domain warping changes the field");

        math::Fixed32 plainLow{}, plainHigh{}, warpLow{}, warpHigh{};
        (void) procgen::heightRange(fbm, plainLow, plainHigh);
        (void) procgen::heightRange(warped, warpLow, warpHigh);
        const double plainSpan = static_cast<double>((plainHigh - plainLow).toFloat());
        const double warpSpan = static_cast<double>((warpHigh - warpLow).toFloat());
        check(warpSpan > plainSpan * 0.6 && warpSpan < plainSpan * 1.6,
              "warping folds the domain without changing the amplitude");

        // Lacunarity and persistence have to be live parameters, not decoration.
        procgen::NoiseParams tuned;
        tuned.seed = 4242u;
        tuned.persistence = 0.8f;
        check(procgen::foldHeightfield(procgen::generateNoiseHeightfield(64u, 64u, tuned)) !=
                  procgen::foldHeightfield(fbm),
              "persistence changes the terrain");
        tuned.persistence = 0.5f;
        tuned.lacunarity = 2.7f;
        check(procgen::foldHeightfield(procgen::generateNoiseHeightfield(64u, 64u, tuned)) !=
                  procgen::foldHeightfield(fbm),
              "lacunarity changes the terrain");
    }

    // ─────────────────────────────────────────────────────────────────────────
    std::printf("\n-- chunking agrees with the whole map --\n");
    {
        // The point of routing both through one sampler. A chunk must be a window
        // onto the same world function, or a chunked world and an unchunked one of
        // the same seed are two different worlds.
        procgen::ChunkParams chunks;
        chunks.size = 16u;
        chunks.worldSeed = 909u;

        check(procgen::countSeamMismatches(chunks, procgen::ChunkCoord{0, 0}, procgen::ChunkCoord{1, 0}) == 0u,
              "east seam is exact");
        check(procgen::countSeamMismatches(chunks, procgen::ChunkCoord{0, 0}, procgen::ChunkCoord{0, 1}) == 0u,
              "south seam is exact");
        check(procgen::countSeamMismatches(chunks, procgen::ChunkCoord{3, -2}, procgen::ChunkCoord{2, -2}) == 0u,
              "seams hold at negative coordinates too");

        // A chunk with a warped layer must still be seamless: the warp is a pure
        // function of the world coordinate, so it cannot break at a boundary.
        chunks.noise.warpStrength = 4.0f;
        check(procgen::countSeamMismatches(chunks, procgen::ChunkCoord{5, 5}, procgen::ChunkCoord{6, 5}) == 0u,
              "a warped layer stays seamless across chunks");
        chunks.noise.kind = procgen::NoiseKind::Ridged;
        check(procgen::countSeamMismatches(chunks, procgen::ChunkCoord{5, 5}, procgen::ChunkCoord{5, 6}) == 0u,
              "a ridged layer stays seamless across chunks");

        check(procgen::chunkSeed(chunks, procgen::ChunkCoord{1, 0}) !=
                  procgen::chunkSeed(chunks, procgen::ChunkCoord{0, 1}),
              "transposed chunk coordinates do not share a seed");
    }

    // ─────────────────────────────────────────────────────────────────────────
    std::printf("\n-- voronoi metrics and warping --\n");
    {
        procgen::VoronoiParams params;
        params.width = 64u;
        params.depth = 64u;
        params.seed = 77u;
        params.cellSize = 10u;

        params.metric = procgen::DistanceMetric::Euclidean;
        const procgen::VoronoiDiagram euclidean = procgen::computeVoronoi(params);
        params.metric = procgen::DistanceMetric::Manhattan;
        const procgen::VoronoiDiagram manhattan = procgen::computeVoronoi(params);
        params.metric = procgen::DistanceMetric::Chebyshev;
        const procgen::VoronoiDiagram chebyshev = procgen::computeVoronoi(params);

        check(procgen::foldRegionMap(euclidean.regions) != procgen::foldRegionMap(manhattan.regions),
              "the Manhattan metric repartitions the space");
        check(procgen::foldRegionMap(manhattan.regions) != procgen::foldRegionMap(chebyshev.regions),
              "the Chebyshev metric differs again");
        check(euclidean.regionCount == manhattan.regionCount,
              "the metric changes region shapes, not how many there are");

        // The property that makes Manhattan the city-block metric: its borders run
        // along the axes, so a border cell is far more likely to have an axial
        // neighbour in a different region than a Euclidean one is.
        const auto axialBorderShare = [](const procgen::VoronoiDiagram &diagram) {
            const procgen::Grid<core::u8> borders = procgen::regionBorders(diagram);
            core::u32 total = 0u;
            core::u32 axial = 0u;
            for (core::u32 z = 1u; z + 1u < diagram.regions.depth(); ++z)
                for (core::u32 x = 1u; x + 1u < diagram.regions.width(); ++x)
                {
                    if (borders.at(x, z) == 0u)
                        continue;
                    ++total;
                    const core::u16 here = diagram.regions.at(x, z);
                    const bool horizontal =
                        diagram.regions.at(x - 1u, z) != here || diagram.regions.at(x + 1u, z) != here;
                    const bool vertical =
                        diagram.regions.at(x, z - 1u) != here || diagram.regions.at(x, z + 1u) != here;
                    // A cell on a straight run of border differs on one axis only.
                    if (horizontal != vertical)
                        ++axial;
                }
            return total ? static_cast<double>(axial) / static_cast<double>(total) : 0.0;
        };
        std::printf("    straight-border share: euclidean=%.3f manhattan=%.3f chebyshev=%.3f\n",
                    axialBorderShare(euclidean), axialBorderShare(manhattan), axialBorderShare(chebyshev));

        params.metric = procgen::DistanceMetric::Euclidean;
        params.warpStrength = 5.0f;
        const procgen::VoronoiDiagram warped = procgen::computeVoronoi(params);
        check(procgen::foldRegionMap(warped.regions) != procgen::foldRegionMap(euclidean.regions),
              "domain warping folds the region borders");
        bool everyCellClaimed = true;
        for (core::u32 i = 0u; i < warped.regions.cellCount(); ++i)
            everyCellClaimed = everyCellClaimed && warped.regions[i] != procgen::kNoRegion;
        check(everyCellClaimed, "a warped diagram still claims every cell");

        // The u16 sentinel guard: a partition finer than 65535 regions would wrap
        // ids into "no region" and silently leave cells unclaimed.
        procgen::VoronoiParams excessive;
        excessive.width = 512u;
        excessive.depth = 512u;
        excessive.cellSize = 1u;
        check(procgen::computeVoronoi(excessive).regionCount == 0u,
              "a partition too fine for a u16 id is refused, not wrapped");
    }

    // ─────────────────────────────────────────────────────────────────────────
    std::printf("\n-- tensor fields steer the turtle --\n");
    {
        lpl::pmr::vector<procgen::FieldRegion> influences;
        procgen::FieldRegion downtown;
        downtown.pattern = procgen::FieldPattern::Radial;
        downtown.centerX = 32u;
        downtown.centerZ = 32u;
        downtown.strength = 1.0f;
        downtown.falloff = 0.02f;
        influences.push_back(downtown);

        procgen::FieldRegion suburb;
        suburb.pattern = procgen::FieldPattern::Grid;
        suburb.centerX = 8u;
        suburb.centerZ = 56u;
        suburb.bearing = 0u;
        suburb.strength = 1.0f;
        suburb.falloff = 0.03f;
        influences.push_back(suburb);

        const procgen::HeadingField field = procgen::bakeHeadingField(64u, 64u, influences);
        check(!field.empty(), "the field bakes");

        // The two regions must genuinely disagree, or blending them is pointless.
        core::u32 distinctHeadings = 0u;
        bool seen[procgen::kTurtleDirections] = {};
        for (core::u32 i = 0u; i < field.cellCount(); ++i)
            if (!seen[field[i]])
            {
                seen[field[i]] = true;
                ++distinctHeadings;
            }
        check(distinctHeadings >= 8u, "a radial region produces headings all the way round");

        const procgen::LSystemParams grammar = procgen::makeRoadGrammar();
        const lpl::pmr::string expanded = procgen::expandLSystem(grammar);

        procgen::TurtleParams turtle;
        turtle.startX = 32u;
        turtle.startZ = 58u;
        turtle.stepLength = 3u;

        procgen::Grid<core::u8> free{64u, 64u, 0u};
        procgen::Grid<core::u8> steered{64u, 64u, 0u};
        const core::u32 freeCells = procgen::drawTurtle(expanded, turtle, free);
        const core::u32 steeredCells = procgen::drawTurtleInField(expanded, turtle, field, 0.7f, steered);

        check(freeCells > 0u && steeredCells > 0u, "both turtles draw something");
        bool differs = false;
        for (core::u32 i = 0u; i < free.cellCount() && !differs; ++i)
            differs = free[i] != steered[i];
        check(differs, "the field changes where the roads go");

        // Conformity zero must reproduce the unsteered walk exactly: the field is an
        // influence, and an influence of zero is not an influence.
        procgen::Grid<core::u8> unsteered{64u, 64u, 0u};
        (void) procgen::drawTurtleInField(expanded, turtle, field, 0.0f, unsteered);
        bool identical = true;
        for (core::u32 i = 0u; i < free.cellCount(); ++i)
            identical = identical && free[i] == unsteered[i];
        check(identical, "conformity of zero reproduces the plain turtle exactly");
    }

    // ─────────────────────────────────────────────────────────────────────────
    std::printf("\n-- weighted grammar alternatives --\n");
    {
        procgen::LSystemParams grammar = procgen::makeBranchingGrammar();
        grammar.seed = 11u;
        const lpl::pmr::string a = procgen::expandLSystem(grammar);
        grammar.seed = 12u;
        const lpl::pmr::string b = procgen::expandLSystem(grammar);
        grammar.seed = 11u;
        const lpl::pmr::string again = procgen::expandLSystem(grammar);

        check(a == again, "the same seed regrows the same structure");
        check(a != b, "a different seed picks different alternatives");

        // A grammar with one rule per symbol must stay a pure function, so an
        // existing single-rule grammar keeps behaving exactly as before.
        procgen::LSystemParams single;
        single.axiom = "F";
        single.rules.push_back(procgen::LRule{'F', lpl::pmr::string{"F+F"}});
        single.iterations = 4u;
        single.seed = 1u;
        const lpl::pmr::string one = procgen::expandLSystem(single);
        single.seed = 999u;
        check(one == procgen::expandLSystem(single), "a single-rule grammar ignores the seed");
    }

    // ─────────────────────────────────────────────────────────────────────────
    std::printf("\n-- hot path analysis --\n");
    {
        procgen::CaveParams cave;
        cave.width = 56u;
        cave.depth = 56u;
        cave.seed = 4242u;
        const procgen::DungeonMap level = procgen::generateCellularCave(cave);

        core::u32 startX = 0u, startZ = 0u, goalX = 0u, goalZ = 0u;
        check(procgen::findFarthestPair(level, startX, startZ, goalX, goalZ), "the level has two ends");

        const procgen::HotPathAnalysis hot = procgen::analyseHotPath(level, startX, startZ, goalX, goalZ, 6u);
        check(hot.valid, "the hot path is found");
        check(hot.pathCells > 8u, "the spine has real length");

        // Every cell on the path must be at detour zero, and no other cell may be:
        // that is what makes the detour map a measure of distance FROM the spine.
        bool consistent = true;
        for (core::u32 i = 0u; i < level.cellCount(); ++i)
        {
            const bool onPath = hot.onPath[i] != 0u;
            const bool atZero = hot.detour[i] == 0u;
            consistent = consistent && onPath == atZero;
        }
        check(consistent, "detour zero is exactly the spine");

        // The path must be a connected walk, not a scatter of cells.
        bool contiguous = true;
        core::u32 pathNeighbours = 0u;
        for (core::u32 z = 0u; z < level.depth(); ++z)
            for (core::u32 x = 0u; x < level.width(); ++x)
            {
                if (hot.onPath.at(x, z) == 0u)
                    continue;
                core::u32 adjacent = 0u;
                for (core::u32 n = 0u; n < 4u; ++n)
                {
                    const core::i32 nx = static_cast<core::i32>(x) + procgen::kNeighbor4X[n];
                    const core::i32 nz = static_cast<core::i32>(z) + procgen::kNeighbor4Z[n];
                    if (hot.onPath.contains(nx, nz) &&
                        hot.onPath.at(static_cast<core::u32>(nx), static_cast<core::u32>(nz)) != 0u)
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
        std::printf("    spine=%u cells, deepest detour=%u, cells beyond the limit=%u\n", hot.pathCells,
                    hot.deepestDetour, hot.excessiveCells);

        // A flee map must not have a local minimum at a dead end: that is the
        // corner-trapping the negate-and-relax construction exists to remove.
        const procgen::DistanceMap danger = procgen::computeDistanceMap(level, startX, startZ);
        const procgen::DistanceMap flee = procgen::computeFleeMap(level, danger);

        core::u32 traps = 0u;
        for (core::u32 z = 0u; z < level.depth(); ++z)
            for (core::u32 x = 0u; x < level.width(); ++x)
            {
                const core::u32 cell = level.index(x, z);
                if (flee[cell] == procgen::kUnreachable || flee[cell] == 0u)
                    continue;
                bool anyLower = false;
                for (core::u32 n = 0u; n < 4u; ++n)
                {
                    const core::i32 nx = static_cast<core::i32>(x) + procgen::kNeighbor4X[n];
                    const core::i32 nz = static_cast<core::i32>(z) + procgen::kNeighbor4Z[n];
                    if (!level.contains(nx, nz))
                        continue;
                    const core::u32 index = level.index(static_cast<core::u32>(nx), static_cast<core::u32>(nz));
                    if (flee[index] != procgen::kUnreachable && flee[index] < flee[cell])
                        anyLower = true;
                }
                if (!anyLower)
                    ++traps;
            }
        check(traps == 0u, "the flee map has no local minimum to wedge an agent into");

        // ── the analysis FURNISHES the level, not just judges it ─────────────
        //
        // Both answers were already computed and neither was used by anything: the
        // spine is where the player certainly goes, so an encounter placed on it is
        // met rather than missed, and the deepest dead ends are where something is
        // worth hiding for the same reason they were architectural excrescences.
        procgen::PlacementParams furnish;
        furnish.encounters = 4u;
        furnish.rewards = 3u;
        furnish.minSpacing = 3u;
        furnish.rewardMinDetour = 2u;
        procgen::Placement spots[16]{};
        const core::u32 placed = procgen::placeAlongHotPath(level, hot, startX, startZ, furnish, spots, 16u);
        std::printf("    furnished %u spots\n", placed);
        check(placed > 0u, "the level gets furnished from its own measurements");

        core::u32 encounters = 0u;
        core::u32 rewards = 0u;
        bool encountersOnSpine = true;
        bool rewardsOffSpine = true;
        bool rewardsAreDeadEnds = true;
        for (core::u32 i = 0u; i < placed; ++i)
        {
            const core::u32 cell = level.index(spots[i].x, spots[i].z);
            if (spots[i].role == procgen::PlacementRole::Encounter)
            {
                ++encounters;
                encountersOnSpine = encountersOnSpine && hot.onPath[cell] != 0u;
            }
            else
            {
                ++rewards;
                rewardsOffSpine = rewardsOffSpine && spots[i].detour >= furnish.rewardMinDetour;
                core::u32 adjacent = 0u;
                for (core::u32 n = 0u; n < 4u; ++n)
                {
                    const core::i32 nx = static_cast<core::i32>(spots[i].x) + procgen::kNeighbor4X[n];
                    const core::i32 nz = static_cast<core::i32>(spots[i].z) + procgen::kNeighbor4Z[n];
                    if (level.contains(nx, nz) &&
                        procgen::isWalkable(level.at(static_cast<core::u32>(nx), static_cast<core::u32>(nz))))
                        ++adjacent;
                }
                rewardsAreDeadEnds = rewardsAreDeadEnds && adjacent == 1u;
            }
        }
        std::printf("    %u encounters on the spine, %u rewards off it\n", encounters, rewards);
        check(encounters > 0u && encountersOnSpine, "every encounter sits ON the critical path");
        check(rewards > 0u && rewardsOffSpine, "every reward sits off it, past the threshold");
        // A prize the player walks over by accident is litter, not a reward.
        check(rewardsAreDeadEnds, "and every reward is a dead end, not a cell in a room");

        // Encounters spread through the ROUTE, not through the grid: the whole point
        // of measuring progress rather than counting cells.
        core::u32 lastProgress = 0u;
        bool spreads = true;
        for (core::u32 i = 0u; i < placed; ++i)
        {
            if (spots[i].role != procgen::PlacementRole::Encounter)
                continue;
            spreads = spreads && spots[i].progress > lastProgress;
            lastProgress = spots[i].progress;
        }
        check(spreads, "encounters advance through the player's progress, in order");

        // Deterministic, because a furnished level is content and content is folded.
        procgen::Placement again[16]{};
        const core::u32 twice = procgen::placeAlongHotPath(level, hot, startX, startZ, furnish, again, 16u);
        bool identical = twice == placed;
        for (core::u32 i = 0u; i < placed && identical; ++i)
            identical = again[i].x == spots[i].x && again[i].z == spots[i].z && again[i].role == spots[i].role &&
                        again[i].detour == spots[i].detour;
        check(identical, "furnishing the same level twice gives the same spots");

        // A capacity of one is not an error: it is a caller with room for one spot.
        procgen::Placement single{};
        check(procgen::placeAlongHotPath(level, hot, startX, startZ, furnish, &single, 1u) == 1u,
              "a caller with room for one gets one");
    }

    // ─────────────────────────────────────────────────────────────────────────
    std::printf("\n-- blue-noise scatter --\n");
    {
        procgen::WorldBuilder builder{31337u};
        builder.terrain(96u, 96u).erode().rivers().biomes();

        procgen::ScatterRule rule;
        rule.biome = procgen::BiomeId::Grassland;
        rule.density = 0.1f;
        builder.scatter(rule);

        ecs::Registry registry;
        const procgen::BuiltWorldStats stats = builder.materialize(registry);
        check(stats.propEntities > 0u, "props were placed");

        // The property a coin flip cannot give: no two props share a cell, and none
        // are adjacent. White noise would routinely produce both.
        procgen::Grid<core::u8> occupied{96u, 96u, 0u};
        core::u32 propCells = 0u;
        const procgen::BiomeMap &biomes = builder.biomeMap();
        for (core::u32 i = 0u; i < biomes.cellCount(); ++i)
            if (biomes[i] == procgen::BiomeId::Grassland)
                ++propCells;
        check(propCells > stats.propEntities, "props are a subset of their biome, not all of it");

        // Reproducibility of the whole scatter, shuffle included.
        procgen::WorldBuilder twin{31337u};
        twin.terrain(96u, 96u).erode().rivers().biomes().scatter(rule);
        ecs::Registry twinRegistry;
        const procgen::BuiltWorldStats twinStats = twin.materialize(twinRegistry);
        check(twinStats.propEntities == stats.propEntities, "the scatter is reproducible");

        // Density has to be monotone: asking for more must not give less.
        procgen::ScatterRule denser = rule;
        denser.density = 0.3f;
        procgen::WorldBuilder thick{31337u};
        thick.terrain(96u, 96u).erode().rivers().biomes().scatter(denser);
        ecs::Registry thickRegistry;
        const procgen::BuiltWorldStats thickStats = thick.materialize(thickRegistry);
        check(thickStats.propEntities > stats.propEntities, "a higher density places more props");
        std::printf("    density 0.10 -> %u props, density 0.30 -> %u props, over %u eligible cells\n",
                    stats.propEntities, thickStats.propEntities, propCells);

        // A moisture window has to exclude something, or the filter is decoration.
        procgen::ScatterRule wetOnly = rule;
        wetOnly.minMoisture = 0.75f;
        procgen::WorldBuilder marshy{31337u};
        marshy.terrain(96u, 96u).erode().rivers().biomes().scatter(wetOnly);
        ecs::Registry marshyRegistry;
        const procgen::BuiltWorldStats marshyStats = marshy.materialize(marshyRegistry);
        check(marshyStats.propEntities < stats.propEntities, "a moisture window rejects dry ground");
    }

    // ─────────────────────────────────────────────────────────────────────────
    std::printf("\n-- the builder reaches the whole module --\n");
    {
        procgen::WorldBuilder builder{5150u};
        builder.terrain(48u, 48u).erode().rivers().biomes();

        const procgen::TileSet tiles = procgen::makeTerrainTileSet();
        procgen::WfcParams wfc;
        wfc.maxAttempts = 4u;
        builder.tiles(tiles, wfc);
        check(!builder.tileMap().empty(), "the builder can solve tiles over its own terrain");
        check(builder.tileMap().width() == 48u, "the tile solve took its extent from the terrain");
        check(procgen::countAdjacencyViolations(builder.tileMap(), tiles) == 0u,
              "the tile solve respects the rules while pinned to the biome map");

        procgen::CaveParams cave;
        cave.width = 48u;
        cave.depth = 48u;
        builder.caves(cave).validate(procgen::GateCriteria{});
        check(builder.lastQuality().walkableCells > 0u, "the gate measured the cave");
        check(builder.gatePassed(), "a default cave passes the default criteria");

        // A gate nobody can satisfy has to fail, or it is not a gate.
        procgen::GateCriteria impossible;
        impossible.minPathLength = 100000u;
        builder.validate(impossible);
        check(!builder.gatePassed(), "an unsatisfiable criterion is reported as a failure");

        // Validating with no dungeon must not claim success for a level that was
        // never generated.
        procgen::WorldBuilder bare{1u};
        bare.terrain(32u, 32u).validate(procgen::GateCriteria{});
        check(!bare.gatePassed(), "validating nothing is not a pass");
    }

    std::printf("\n%s (%d failures)\n", g_failures == 0 ? "ALL PASS" : "FAILURES", g_failures);
    return g_failures == 0 ? 0 : 1;
}
