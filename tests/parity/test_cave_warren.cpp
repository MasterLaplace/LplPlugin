/**
 * @file test_cave_warren.cpp
 * @brief The cave you can walk into: siting, cover, the way in, and the collider.
 *
 * A generator that produces a cave is easy to assert and easy to be wrong about. The
 * claims here are the ones a player can feel — there is a way in, the way in leads
 * somewhere, a body fits through it, and rock stops it — and each of them is paired
 * with the run that has to FAIL, because "the body got inside" is satisfied by a
 * collider that lets everything through.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-08-06
 * @copyright MIT License
 */

#include <lpl/engine/CaveParity.hpp>
#include <lpl/procgen/CaveWarren.hpp>
#include <lpl/procgen/EndlessPlan.hpp>
#include <lpl/procgen/WorldRecipe.hpp>

#include <cstdio>

namespace {

using namespace lpl;

core::u32 gChecks = 0u;
core::u32 gFailures = 0u;

void check(bool condition, const char *what, long long value = 0)
{
    ++gChecks;
    if (condition)
        return;
    ++gFailures;
    std::printf("  FAIL  %s (value %lld)\n", what, value);
}

/// The world the walked client actually streams, so a measurement here is a
/// measurement of that world and not of a world invented for the test.
[[nodiscard]] procgen::EndlessPlan walkedPlan()
{
    procgen::WorldRecipe recipe = procgen::parityWorldRecipe();
    recipe.seed = 0x5EEDCA7Eu;
    recipe.terrain.seed = recipe.seed;
    return procgen::endlessPlanFromRecipe(recipe, 24u);
}

} // namespace

int main()
{
    const procgen::EndlessPlan plan = walkedPlan();
    const procgen::ChunkParams &chunk = plan.chunk;
    const procgen::LandmarkParams mouths = plan.rule.caveMouths;
    const procgen::CaveWarrenParams warrenParams = plan.rule.warren;
    const core::f32 drop = plan.rule.caveMouthDrop;

    std::printf("-- how many caves the walked world actually has --\n");

    // A WINDOW, not a corner: sampling one neighbourhood and generalising to the world
    // has already produced a confident wrong answer in this repository.
    constexpr core::i32 kHalf = 10;
    core::u32 sites = 0u;
    core::u32 valid = 0u;
    core::u32 navigable = 0u;
    core::u32 byKind[5] = {0u, 0u, 0u, 0u, 0u};
    core::u32 aditTotal = 0u;
    core::u32 openTotal = 0u;
    core::u32 mouthOutsideVolume = 0u;
    core::u32 apertureless = 0u;

    for (core::i32 lz = -kHalf; lz <= kHalf; ++lz)
        for (core::i32 lx = -kHalf; lx <= kHalf; ++lx)
        {
            procgen::LandmarkSite site;
            if (!procgen::landmarkAt(chunk, mouths, procgen::LandmarkKind::CaveMouth, plan.rule.seaLevel, lx, lz, site))
                continue;
            ++sites;

            const procgen::CaveWarren warren = procgen::buildCaveWarren(chunk, site, warrenParams, drop);
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
            ++byKind[static_cast<core::u32>(warren.kind)];
        }

    std::printf("  %d landmark cells, %u sited mouths, %u caves, %u reach their bottom\n",
                (2 * kHalf + 1) * (2 * kHalf + 1), sites, valid, navigable);
    std::printf("  mean trench %.1f cells, mean %.0f open cells\n", valid ? static_cast<double>(aditTotal) / valid : 0.0,
                valid ? static_cast<double>(openTotal) / valid : 0.0);
    std::printf("  kinds:");
    for (core::u32 i = 0u; i < 5u; ++i)
        std::printf(" %s=%u", procgen::caveKindName(static_cast<procgen::CaveKind>(i)), byKind[i]);
    std::printf("\n");

    // Eighteen, not the sixty-eight the raw defaults site: endlessPlanFromRecipe
    // CALIBRATES the relief a mouth needs against the terrain that actually exists, so
    // this measures the world rather than the header. Roughly one cave every hundred
    // and twenty metres of walking, which is what "you will find one" costs.
    check(sites > 10u, "the walked world sites cave mouths at all", sites);
    // A SHARE, not a count. The absolute number moves whenever the terrain does, and
    // an absolute threshold against a distribution that moves is this repository's
    // most-repeated mistake.
    check(valid * 2u > sites, "more than half of the sited mouths carry a cave", valid);
    check(navigable * 3u > valid, "and a third of those reach their deepest gallery", navigable);
    // The invariant behind the halfSpan clamp in planCaveAdit. It cost five caves in
    // fifty-two before the reach was bounded by the footprint as well as by the array.
    check(mouthOutsideVolume == 0u, "no mouth is found outside its own volume", mouthOutsideVolume);
    check(apertureless == 0u, "every cave that built has a doorway", apertureless);
    // The document PINNED a kind, so every cave has it. That is the developer's half of
    // the bargain and it has to hold exactly: a recipe that names `bsp` and gets caves
    // anyway is a document that does not mean anything.
    check(byKind[static_cast<core::u32>(plan.rule.warren.kind)] == valid, "a named kind is honoured by every cave",
          byKind[static_cast<core::u32>(plan.rule.warren.kind)]);

    // ── And the other half: `auto` reads the place ───────────────────────────
    //
    // Run over the same sites with the choice handed back, because the two claims are
    // different and only one of them can be true of any single world. A rule that
    // resolved to one generator everywhere would fire on every site and say nothing,
    // which is indistinguishable from no rule at all.
    std::printf("\n-- what `auto` makes of the same places --\n");
    {
        procgen::CaveWarrenParams autoParams = warrenParams;
        autoParams.kind = procgen::CaveKind::Auto;
        core::u32 autoKind[5] = {0u, 0u, 0u, 0u, 0u};
        core::u32 settled = 0u;
        core::u32 wet = 0u;
        for (core::i32 lz = -kHalf; lz <= kHalf; ++lz)
            for (core::i32 lx = -kHalf; lx <= kHalf; ++lx)
            {
                procgen::LandmarkSite site;
                if (!procgen::landmarkAt(chunk, mouths, procgen::LandmarkKind::CaveMouth, plan.rule.seaLevel, lx, lz,
                                         site))
                    continue;
                const procgen::CaveWarren warren = procgen::buildCaveWarren(chunk, site, autoParams, drop);
                if (!warren.valid)
                    continue;
                ++autoKind[static_cast<core::u32>(warren.kind)];
                settled += procgen::settledNearSite(chunk, autoParams.villages, autoParams.seaLevel, site,
                                                    autoParams.settlementReach)
                               ? 1u
                               : 0u;
                wet += procgen::sampleWorldMoisture(chunk, site.cellX, site.cellZ) >=
                               math::Fixed32::fromFloat(procgen::kKarstWetness)
                           ? 1u
                           : 0u;
            }
        std::printf("  kinds:");
        for (core::u32 i = 0u; i < 5u; ++i)
            std::printf(" %s=%u", procgen::caveKindName(static_cast<procgen::CaveKind>(i)), autoKind[i]);
        std::printf("   (%u settled sites, %u wet)\n", settled, wet);

        core::u32 distinctKinds = 0u;
        for (core::u32 i = 0u; i < 5u; ++i)
            distinctKinds += autoKind[i] != 0u ? 1u : 0u;
        check(distinctKinds >= 2u, "the place decides, and places differ", distinctKinds);
        check(autoKind[static_cast<core::u32>(procgen::CaveKind::Auto)] == 0u,
              "and `auto` always resolves to a real kind");
        // Evidence, not a coin: the rule can only vary where the world varies, so a
        // world with no settlements and no wet ground SHOULD come out all cellular.
        // Asserting the variety without asserting the evidence exists would be a check
        // that passes for a reason nobody stated.
        check(settled != 0u || wet != 0u, "and the evidence it reads is actually present in this world",
              static_cast<long long>(settled + wet));
    }

    // ── The cover rule, in both directions ───────────────────────────────────
    //
    // "There is rock over the cave" is the property the whole design turns on, and
    // asserting only that some columns are covered would be satisfied by a mask that
    // said yes everywhere.
    std::printf("\n-- rock over the roof --\n");
    {
        procgen::CaveWarren warren;
        for (core::i32 lz = -kHalf; lz <= kHalf && !warren.valid; ++lz)
            for (core::i32 lx = -kHalf; lx <= kHalf; ++lx)
            {
                procgen::LandmarkSite site;
                if (!procgen::landmarkAt(chunk, mouths, procgen::LandmarkKind::CaveMouth, plan.rule.seaLevel, lx, lz,
                                         site))
                    continue;
                procgen::CaveWarren candidate = procgen::buildCaveWarren(chunk, site, warrenParams, drop);
                if (candidate.valid)
                {
                    warren = static_cast<procgen::CaveWarren &&>(candidate);
                    break;
                }
            }
        check(warren.valid, "the survey found a cave to examine");

        const core::f32 threshold = procgen::caveCoverThreshold(warrenParams, warren.adit.floorY);
        core::u32 coveredButBare = 0u;
        core::u32 bareButCovered = 0u;
        for (core::u32 z = 0u; z < warren.covered.depth(); ++z)
            for (core::u32 x = 0u; x < warren.covered.width(); ++x)
            {
                const core::i32 worldX = warren.originX + static_cast<core::i32>(x);
                const core::i32 worldZ = warren.originZ + static_cast<core::i32>(z);
                core::f32 ground = procgen::sampleWorldHeight(chunk, worldX, worldZ).toFloat();
                core::f32 cut = 0.0f;
                if (procgen::caveMouthFloorAt(warren.site, warren.adit, worldX, worldZ, cut) && cut < ground)
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

        // The span, where it matters: a doorway column has rock over it and the shelf
        // it opens onto does not, and the collider is told both.
        const procgen::VerticalSpan inside =
            procgen::caveWarrenSpanAt(warren, warren.apertureX[0], warren.apertureZ[0],
                                      math::Fixed32::fromFloat(warren.adit.floorY + 0.1f),
                                      math::Fixed32::fromFloat(warren.adit.floorY));
        check(inside.enclosed, "the doorway has rock over it");
        check(inside.headroom() > math::Fixed32::fromFloat(1.8f), "and room enough for a body under it",
              static_cast<long long>(inside.headroom().toFloat() * 100.0f));

        const procgen::VerticalSpan outside =
            procgen::caveWarrenSpanAt(warren, warren.site.cellX, warren.site.cellZ,
                                      math::Fixed32::fromFloat(warren.adit.floorY),
                                      math::Fixed32::fromFloat(warren.adit.floorY));
        check(!outside.enclosed, "and the shelf it opens onto is open sky");
    }

    // ── The walk, and the walk that must not work ────────────────────────────
    std::printf("\n-- walking in --\n");
    const engine::CaveFoldResult open = engine::foldCaveParity();
    const engine::CaveFoldResult sealed = engine::foldSealedCaveParity();

    std::printf("  open:   enclosed %u ticks, descended %u levels, blocked %u, head %u\n", open.enclosedTicks,
                open.descendedLevels, open.blocked, open.headBumps);
    std::printf("  sealed: enclosed %u ticks, blocked %u\n", sealed.enclosedTicks, sealed.blocked);

    check(open.warrenSignature != 0u, "the gate found a cave to walk into");
    check(open.enclosedTicks > 0u, "a body walking at the mouth ends up under rock", open.enclosedTicks);
    // The control. Without it the check above is satisfied by a collider that lets
    // everything through, and a collider that let everything through would also let a
    // body through a mountain.
    check(sealed.enclosedTicks == 0u, "and a doorway filled with rock lets nobody in", sealed.enclosedTicks);
    check(sealed.blocked > open.blocked, "the sealed run is stopped, not merely slower", sealed.blocked);
    check(open.walkSignature != sealed.walkSignature, "the two walks are genuinely different runs");
    check(open.spanSignature != sealed.spanSignature, "and they disagree about where the rock is");

    std::printf("\n-- signatures the kernel must reproduce --\n");
    std::printf("  warren_sig = 0x%08X\n", open.warrenSignature);
    std::printf("  walk_sig   = 0x%08X\n", open.walkSignature);
    std::printf("  span_sig   = 0x%08X\n", open.spanSignature);
    std::printf("  sealed_sig = 0x%08X\n", sealed.walkSignature);
    std::printf("  covered    = %u\n", open.coveredColumns);
    std::printf("  open       = %u\n", open.openCells);
    std::printf("  reachable  = %u\n", open.reachableCells);
    std::printf("  aperture   = %u\n", open.apertureCells);
    std::printf("  path       = %u\n", open.pathLength);
    std::printf("  enclosed   = %u\n", open.enclosedTicks);
    std::printf("  descended  = %u\n", open.descendedLevels);
    std::printf("  kind       = %u\n", open.kind);

    std::printf("\nALL PASS (%u failures, %u checks)\n", gFailures, gChecks);
    return gFailures == 0u ? 0 : 1;
}
