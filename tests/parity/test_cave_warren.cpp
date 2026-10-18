#include <lpl/procgen/CaveWarren.hpp>
#include <lpl/procgen/EndlessPlan.hpp>
#include <lpl/procgen/WorldRecipe.hpp>

#include <cstdio>

namespace {

/**
 * @brief Half the side of the window of landmark cells every survey here covers.
 *
 * A WINDOW, not a corner: sampling one neighbourhood and generalising to the world
 * has already produced a confident wrong answer in this repository.
 */
constexpr lpl::core::i32 kHalf = 10;

lpl::core::u32 gChecks = 0u;
lpl::core::u32 gFailures = 0u;

void check(bool condition, const char *what, long long value = 0)
{
    ++gChecks;
    if (condition)
        return;
    ++gFailures;
    std::printf("  FAIL  %s (value %lld)\n", what, value);
}

/**
 * @brief The world the walked client actually streams, so a measurement here is a
 *        measurement of that world and not of a world invented for the test.
 */
[[nodiscard]] lpl::procgen::EndlessPlan walkedPlan()
{
    lpl::procgen::WorldRecipe recipe = lpl::procgen::parityWorldRecipe();
    recipe.seed = 0x5EEDCA7Eu;
    recipe.terrain.seed = recipe.seed;
    return lpl::procgen::endlessPlanFromRecipe(recipe, 24u);
}

/**
 * @brief How many caves the walked world actually has.
 *
 * Eighteen, not the sixty-eight the raw defaults site: endlessPlanFromRecipe
 * CALIBRATES the relief a mouth needs against the terrain that actually exists, so
 * this measures the world rather than the header. Roughly one cave every hundred
 * and twenty metres of walking, which is what "you will find one" costs.
 *
 * The caves are counted as a SHARE, not a count. The absolute number moves whenever
 * the terrain does, and an absolute threshold against a distribution that moves is
 * this repository's most-repeated mistake.
 *
 * No mouth is found outside its own volume: the invariant behind the halfSpan clamp in
 * planCaveAdit. It cost five caves in fifty-two before the reach was bounded by the
 * footprint as well as by the array.
 *
 * The document PINNED a kind, so every cave has it. That is the developer's half of
 * the bargain and it has to hold exactly: a recipe that names `bsp` and gets caves
 * anyway is a document that does not mean anything.
 */
void checkHowManyCavesTheWalkedWorldHas(const lpl::procgen::EndlessPlan &plan, const lpl::procgen::ChunkParams &chunk,
                                        const lpl::procgen::LandmarkParams &mouths,
                                        const lpl::procgen::CaveWarrenParams &warrenParams, lpl::core::f32 drop)
{
    std::printf("-- how many caves the walked world actually has --\n");

    lpl::core::u32 sites = 0u;
    lpl::core::u32 valid = 0u;
    lpl::core::u32 navigable = 0u;
    lpl::core::u32 byKind[5] = {0u, 0u, 0u, 0u, 0u};
    lpl::core::u32 aditTotal = 0u;
    lpl::core::u32 openTotal = 0u;
    lpl::core::u32 mouthOutsideVolume = 0u;
    lpl::core::u32 apertureless = 0u;

    for (lpl::core::i32 lz = -kHalf; lz <= kHalf; ++lz)
        for (lpl::core::i32 lx = -kHalf; lx <= kHalf; ++lx)
        {
            lpl::procgen::LandmarkSite site;
            if (!lpl::procgen::landmarkAt(chunk, mouths, lpl::procgen::LandmarkKind::CaveMouth, plan.rule.seaLevel, lx,
                                          lz, site))
                continue;
            ++sites;

            const lpl::procgen::CaveWarren warren = lpl::procgen::buildCaveWarren(chunk, site, warrenParams, drop);
            if (!warren.adit.found)
                continue;
            if (!warren.containsCell(warren.adit.mouthX, warren.adit.mouthZ))
                ++mouthOutsideVolume;
            if (!warren.valid)
                continue;
            ++valid;
            aditTotal += warren.adit.length;
            openTotal += warren.openCells;
            navigable += warren.navigable ? 1u : 0u;
            if (warren.apertureCount == 0u)
                ++apertureless;
            ++byKind[static_cast<lpl::core::u32>(warren.kind)];
        }

    std::printf("  %d landmark cells, %u sited mouths, %u caves, %u reach their bottom\n",
                (2 * kHalf + 1) * (2 * kHalf + 1), sites, valid, navigable);
    std::printf("  mean trench %.1f cells, mean %.0f open cells\n",
                valid ? static_cast<double>(aditTotal) / valid : 0.0,
                valid ? static_cast<double>(openTotal) / valid : 0.0);
    std::printf("  kinds:");
    for (lpl::core::u32 i = 0u; i < 5u; ++i)
        std::printf(" %s=%u", lpl::procgen::caveKindName(static_cast<lpl::procgen::CaveKind>(i)), byKind[i]);
    std::printf("\n");

    check(sites > 10u, "the walked world sites cave mouths at all", sites);
    check(valid * 2u > sites, "more than half of the sited mouths carry a cave", valid);
    check(navigable * 3u > valid, "and a third of those reach their deepest gallery", navigable);
    check(mouthOutsideVolume == 0u, "no mouth is found outside its own volume", mouthOutsideVolume);
    check(apertureless == 0u, "every cave that built has a doorway", apertureless);
    check(byKind[static_cast<lpl::core::u32>(plan.rule.warren.kind)] == valid, "a named kind is honoured by every cave",
          byKind[static_cast<lpl::core::u32>(plan.rule.warren.kind)]);
}

/**
 * @brief And the other half: `auto` reads the place.
 *
 * Run over the same sites with the choice handed back, because the two claims are
 * different and only one of them can be true of any single world. A rule that
 * resolved to one generator everywhere would fire on every site and say nothing,
 * which is indistinguishable from no rule at all.
 *
 * Evidence, not a coin: the rule can only vary where the world varies, so a world
 * with no settlements and no wet ground SHOULD come out all cellular. Asserting the
 * variety without asserting the evidence exists would be a check that passes for a
 * reason nobody stated.
 */
void checkAutoReadsThePlace(const lpl::procgen::EndlessPlan &plan, const lpl::procgen::ChunkParams &chunk,
                            const lpl::procgen::LandmarkParams &mouths,
                            const lpl::procgen::CaveWarrenParams &warrenParams, lpl::core::f32 drop)
{
    std::printf("\n-- what `auto` makes of the same places --\n");
    lpl::procgen::CaveWarrenParams autoParams = warrenParams;
    autoParams.kind = lpl::procgen::CaveKind::Auto;
    lpl::core::u32 autoKind[5] = {0u, 0u, 0u, 0u, 0u};
    lpl::core::u32 settled = 0u;
    lpl::core::u32 wet = 0u;
    for (lpl::core::i32 lz = -kHalf; lz <= kHalf; ++lz)
        for (lpl::core::i32 lx = -kHalf; lx <= kHalf; ++lx)
        {
            lpl::procgen::LandmarkSite site;
            if (!lpl::procgen::landmarkAt(chunk, mouths, lpl::procgen::LandmarkKind::CaveMouth, plan.rule.seaLevel, lx,
                                          lz, site))
                continue;
            const lpl::procgen::CaveWarren warren = lpl::procgen::buildCaveWarren(chunk, site, autoParams, drop);
            if (!warren.valid)
                continue;
            ++autoKind[static_cast<lpl::core::u32>(warren.kind)];
            settled += lpl::procgen::settledNearSite(chunk, autoParams.villages, autoParams.seaLevel, site,
                                                     autoParams.settlementReach) ?
                           1u :
                           0u;
            wet += lpl::procgen::sampleWorldMoisture(chunk, site.cellX, site.cellZ) >=
                           lpl::math::Fixed32::fromFloat(lpl::procgen::kKarstWetness) ?
                       1u :
                       0u;
        }
    std::printf("  kinds:");
    for (lpl::core::u32 i = 0u; i < 5u; ++i)
        std::printf(" %s=%u", lpl::procgen::caveKindName(static_cast<lpl::procgen::CaveKind>(i)), autoKind[i]);
    std::printf("   (%u settled sites, %u wet)\n", settled, wet);

    lpl::core::u32 distinctKinds = 0u;
    for (lpl::core::u32 i = 0u; i < 5u; ++i)
        distinctKinds += autoKind[i] != 0u ? 1u : 0u;
    check(distinctKinds >= 2u, "the place decides, and places differ", distinctKinds);
    check(autoKind[static_cast<lpl::core::u32>(lpl::procgen::CaveKind::Auto)] == 0u,
          "and `auto` always resolves to a real kind");
    check(settled != 0u || wet != 0u, "and the evidence it reads is actually present in this world",
          static_cast<long long>(settled + wet));
}

/**
 * @brief The cover rule, in both directions.
 *
 * "There is rock over the cave" is the property the whole design turns on, and
 * asserting only that some columns are covered would be satisfied by a mask that
 * said yes everywhere.
 *
 * The span, where it matters: a doorway column has rock over it and the shelf it
 * opens onto does not, and the collider is told both.
 */
void checkTheCoverRuleInBothDirections(const lpl::procgen::EndlessPlan &plan, const lpl::procgen::ChunkParams &chunk,
                                       const lpl::procgen::LandmarkParams &mouths,
                                       const lpl::procgen::CaveWarrenParams &warrenParams, lpl::core::f32 drop)
{
    std::printf("\n-- rock over the roof --\n");
    lpl::procgen::CaveWarren warren;
    for (lpl::core::i32 lz = -kHalf; lz <= kHalf && !warren.valid; ++lz)
        for (lpl::core::i32 lx = -kHalf; lx <= kHalf; ++lx)
        {
            lpl::procgen::LandmarkSite site;
            if (!lpl::procgen::landmarkAt(chunk, mouths, lpl::procgen::LandmarkKind::CaveMouth, plan.rule.seaLevel, lx,
                                          lz, site))
                continue;
            lpl::procgen::CaveWarren candidate = lpl::procgen::buildCaveWarren(chunk, site, warrenParams, drop);
            if (candidate.valid)
            {
                warren = static_cast<lpl::procgen::CaveWarren &&>(candidate);
                break;
            }
        }
    check(warren.valid, "the survey found a cave to examine");

    const lpl::core::f32 threshold = lpl::procgen::caveCoverThreshold(warrenParams, warren.adit.floorY);
    lpl::core::u32 coveredButBare = 0u;
    lpl::core::u32 bareButCovered = 0u;
    for (lpl::core::u32 z = 0u; z < warren.covered.depth(); ++z)
        for (lpl::core::u32 x = 0u; x < warren.covered.width(); ++x)
        {
            const lpl::core::i32 worldX = warren.originX + static_cast<lpl::core::i32>(x);
            const lpl::core::i32 worldZ = warren.originZ + static_cast<lpl::core::i32>(z);
            lpl::core::f32 ground = lpl::procgen::sampleWorldHeight(chunk, worldX, worldZ).toFloat();
            lpl::core::f32 cut = 0.0f;
            if (lpl::procgen::caveMouthFloorAt(warren.site, warren.adit, worldX, worldZ, cut) && cut < ground)
                ground = cut;
            const bool covered = warren.covered.at(x, z) != 0u;
            coveredButBare += covered && ground < threshold ? 1u : 0u;
            bareButCovered += !covered && ground >= threshold ? 1u : 0u;
        }
    std::printf("  %u covered columns of %u, threshold %.2f\n", warren.coveredColumns, warren.covered.cellCount(),
                static_cast<double>(threshold));
    check(coveredButBare == 0u, "nothing is called covered that the ground does not roof", coveredButBare);
    check(bareButCovered == 0u, "and nothing roofed is left out of the cave", bareButCovered);
    check(warren.coveredColumns != 0u && warren.coveredColumns < warren.covered.cellCount(),
          "the mask is a shape, not everything and not nothing", warren.coveredColumns);

    const lpl::procgen::VerticalSpan inside = lpl::procgen::caveWarrenSpanAt(
        warren, warren.apertureX[0], warren.apertureZ[0], lpl::math::Fixed32::fromFloat(warren.adit.floorY + 0.1f),
        lpl::math::Fixed32::fromFloat(warren.adit.floorY));
    check(inside.enclosed, "the doorway has rock over it");
    check(inside.headroom() > lpl::math::Fixed32::fromFloat(1.8f), "and room enough for a body under it",
          static_cast<long long>(inside.headroom().toFloat() * 100.0f));

    const lpl::procgen::VerticalSpan outside = lpl::procgen::caveWarrenSpanAt(
        warren, warren.site.cellX, warren.site.cellZ, lpl::math::Fixed32::fromFloat(warren.adit.floorY),
        lpl::math::Fixed32::fromFloat(warren.adit.floorY));
    check(!outside.enclosed, "and the shelf it opens onto is open sky");
}

} // namespace

/**
 * @brief The cave you can walk into: siting, cover, the way in, and the collider.
 *
 * A generator that produces a cave is easy to assert and easy to be wrong about. The
 * claims here are the ones a player can feel — there is a way in, the way in leads
 * somewhere, a body fits through it, and rock stops it — and each of them is paired
 * with the run that has to FAIL, because "the body got inside" is satisfied by a
 * collider that lets everything through.
 *
 * The walk in and its sealed control are the engine test `caves` (tests/engine/caves.cpp),
 * which runs on both targets.
 */
int main()
{
    const lpl::procgen::EndlessPlan plan = walkedPlan();
    const lpl::procgen::ChunkParams &chunk = plan.chunk;
    const lpl::procgen::LandmarkParams mouths = plan.rule.caveMouths;
    const lpl::procgen::CaveWarrenParams warrenParams = plan.rule.warren;
    const lpl::core::f32 drop = plan.rule.caveMouthDrop;

    checkHowManyCavesTheWalkedWorldHas(plan, chunk, mouths, warrenParams, drop);
    checkAutoReadsThePlace(plan, chunk, mouths, warrenParams, drop);
    checkTheCoverRuleInBothDirections(plan, chunk, mouths, warrenParams, drop);

    std::printf("\n%s (%u failures, %u checks)\n", gFailures == 0u ? "ALL PASS" : "FAILURES", gFailures, gChecks);
    return gFailures == 0u ? 0 : 1;
}
