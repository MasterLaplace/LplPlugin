/**
 * @file CaveParity.cpp
 * @brief Implementation of the cave determinism gate.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-08-06
 * @copyright MIT License
 */

#include <lpl/engine/CaveParity.hpp>

#include <lpl/engine/CharacterController.hpp>
#include <lpl/procgen/EndlessPlan.hpp>
#include <lpl/procgen/WorldRecipe.hpp>

namespace lpl::engine {

namespace {

constexpr core::u32 kFnv1aOffsetBasis = 0x811C9DC5u;
constexpr core::u32 kFnv1aPrime = 0x01000193u;

/// Ticks the walk runs for. Long enough to cross the trench at a jog and be inside.
constexpr core::u32 kWalkTicks = 260u;
/// Cells outside the shelf the walk starts from, downhill of the mouth.
constexpr core::i32 kStartBack = 4;
/// Landmark cells searched for a site that carries a cave. Bounded, and it is a bound.
constexpr core::i32 kSearchHalf = 6;

/**
 * @brief The world the walked client streams, seeded once for the gate.
 *
 * A `constexpr`-shaped constant in the same sense @ref procgen::parityChunkParams is:
 * both sides of the gate call this, so a parameter that moves moves both or neither.
 */
[[nodiscard]] procgen::EndlessPlan parityPlan()
{
    procgen::WorldRecipe recipe = procgen::parityWorldRecipe();
    recipe.seed = 0x5EEDCA7Eu;
    recipe.terrain.seed = recipe.seed;
    return procgen::endlessPlanFromRecipe(recipe, 24u);
}

/**
 * @brief The first site of the lattice that actually carries a cave.
 *
 * Found rather than named, because which landmark cells carry a mouth is a property of
 * the world and not something to hard-code beside it: a constant here would silently
 * stop pointing at a cave the day the terrain moved, and the gate would fold an empty
 * warren without anything saying so.
 *
 * @param plan   The world.
 * @param warren Receives the cave.
 * @return true when one was found within the search.
 */
[[nodiscard]] bool findParityWarren(const procgen::EndlessPlan &plan, procgen::CaveWarren &warren)
{
    const procgen::LandmarkParams mouths = plan.rule.caveMouths;
    for (core::i32 lz = -kSearchHalf; lz <= kSearchHalf; ++lz)
        for (core::i32 lx = -kSearchHalf; lx <= kSearchHalf; ++lx)
        {
            procgen::LandmarkSite site;
            if (!procgen::landmarkAt(plan.chunk, mouths, procgen::LandmarkKind::CaveMouth, plan.rule.seaLevel, lx, lz,
                                     site))
                continue;
            procgen::CaveWarren candidate =
                procgen::buildCaveWarren(plan.chunk, site, plan.rule.warren, plan.rule.caveMouthDrop);
            if (!candidate.valid)
                continue;
            warren = static_cast<procgen::CaveWarren &&>(candidate);
            return true;
        }
    return false;
}

/**
 * @brief The ground of the parity world at one cell, carved exactly as a chunk carves it.
 *
 * The raw field lowered by the mouth this warren belongs to. Not the streamer's field:
 * a gate that needed a resident set would need a streaming schedule, and then what it
 * folded would depend on the order chunks happened to arrive in.
 */
[[nodiscard]] math::Fixed32 parityGround(const procgen::EndlessPlan &plan, const procgen::CaveWarren &warren,
                                         core::i32 worldX, core::i32 worldZ)
{
    const math::Fixed32 raw = procgen::sampleWorldHeight(plan.chunk, worldX, worldZ);
    core::f32 floor = 0.0f;
    if (!procgen::caveMouthFloorAt(warren.site, warren.adit, worldX, worldZ, floor))
        return raw;
    const math::Fixed32 cut = math::Fixed32::fromFloat(floor);
    return cut < raw ? cut : raw;
}

/**
 * @brief Runs the gate against a warren, sealed or not.
 *
 * @param sealed When set, the doorway columns are filled with rock before the walk.
 * @return The fold.
 */
[[nodiscard]] CaveFoldResult runCaveParity(bool sealed)
{
    CaveFoldResult out;
    const procgen::EndlessPlan plan = parityPlan();

    procgen::CaveWarren warren;
    if (!findParityWarren(plan, warren))
        return out;

    if (sealed)
    {
        // Rock in the doorway, and NOWHERE else: the two runs have to differ in one
        // thing or the control proves nothing about the doorway in particular. Every
        // level of the column, because a wall with a gap over it is a doorway.
        for (core::u32 a = 0u; a < warren.apertureCount; ++a)
        {
            const core::i32 lx = warren.apertureX[a] - warren.originX;
            const core::i32 lz = warren.apertureZ[a] - warren.originZ;
            if (lx < 0 || lz < 0 || static_cast<core::u32>(lx) >= warren.volume.width ||
                static_cast<core::u32>(lz) >= warren.volume.depth)
                continue;
            for (core::u32 y = 0u; y < warren.volume.levels; ++y)
                warren.volume.at(static_cast<core::u32>(lx), y, static_cast<core::u32>(lz)) = 1u;
        }
    }

    out.warrenSignature = procgen::foldCaveWarren(warren);
    out.coveredColumns = warren.coveredColumns;
    out.openCells = warren.openCells;
    out.reachableCells = warren.reachableCells;
    out.apertureCells = warren.apertureCount;
    out.pathLength = warren.pathLength;
    out.navigable = warren.navigable ? 1u : 0u;
    out.kind = static_cast<core::u32>(warren.kind);

    const auto space = [&plan, &warren](core::i32 x, core::i32 z, math::Fixed32 y) {
        return procgen::caveWarrenSpanAt(warren, x, z, y, parityGround(plan, warren, x, z));
    };

    // ── The way in, folded as a shape before anybody walks it ────────────────
    //
    // Along the adit, at the height the trench was cut to. Separate from the walk
    // because they fail differently: a span signature that moves means the two targets
    // disagree about where the rock IS, and a walk signature that moves on an
    // unchanged span means they disagree about what a body does with it.
    core::u32 spanHash = kFnv1aOffsetBasis;
    const math::Fixed32 floorY = math::Fixed32::fromFloat(warren.adit.floorY);
    for (core::i32 i = -kStartBack; i <= static_cast<core::i32>(procgen::kMaxAditCells); ++i)
    {
        const core::i32 cellX = warren.site.cellX + warren.adit.stepX * i;
        const core::i32 cellZ = warren.site.cellZ + warren.adit.stepZ * i;
        const procgen::VerticalSpan span = space(cellX, cellZ, floorY);
        spanHash = (spanHash ^ static_cast<core::u32>(span.floor.raw())) * kFnv1aPrime;
        spanHash = (spanHash ^ static_cast<core::u32>(span.ceiling.raw())) * kFnv1aPrime;
        spanHash = (spanHash ^ (span.enclosed ? 1u : 0u)) * kFnv1aPrime;
    }
    out.spanSignature = spanHash;

    // ── The walk ─────────────────────────────────────────────────────────────
    CharacterController body;
    const core::i32 startX = warren.site.cellX - warren.adit.stepX * kStartBack;
    const core::i32 startZ = warren.site.cellZ - warren.adit.stepZ * kStartBack;
    body.placeAt(math::Fixed32::fromInt(startX), math::Fixed32::fromInt(startZ),
                 parityGround(plan, warren, startX, startZ), space);

    // Facing uphill, through CORDIC. The body's own convention is
    // wish = (-forward * sin(yaw), -forward * cos(yaw)), so the heading that walks
    // along (stepX, stepZ) is atan2 of their negations — derived rather than tabulated,
    // or the gate would only work for a site whose adit happens to run north.
    body.setYaw(
        math::Cordic::atan2(math::Fixed32::fromInt(-warren.adit.stepX), math::Fixed32::fromInt(-warren.adit.stepZ)));

    CharacterParams params{};
    CharacterIntent walk{};
    walk.forward = math::Fixed32::one();
    const math::Fixed32 dt = math::Fixed32::fromFloat(1.0f / 60.0f);

    core::u32 walkHash = kFnv1aOffsetBasis;
    core::i32 highestLevel = 0;
    core::i32 lowestLevel = 0;
    bool sawFloor = false;
    for (core::u32 tick = 0u; tick < kWalkTicks; ++tick)
    {
        body.step(params, walk, dt, space);
        walkHash = (walkHash ^ body.fold()) * kFnv1aPrime;
        out.enclosedTicks += body.isEnclosed() ? 1u : 0u;

        // How far DOWN the walk got, in voxel levels rather than metres: a level is the
        // unit a gallery is stacked in, so "descended two levels" is a statement about
        // the cave and "descended 2.8 metres" is one about this particular scale.
        if (!body.isEnclosed())
            continue;
        const core::i32 level = (body.y().raw() - warren.baseYFixed.raw()) / warren.levelHeightFixed.raw();
        if (!sawFloor)
        {
            highestLevel = level;
            lowestLevel = level;
            sawFloor = true;
        }
        if (level > highestLevel)
            highestLevel = level;
        if (level < lowestLevel)
            lowestLevel = level;
    }
    out.walkSignature = walkHash;
    out.descendedLevels = sawFloor ? static_cast<core::u32>(highestLevel - lowestLevel) : 0u;
    out.blocked = body.blockedCount();
    out.headBumps = body.headBumpCount();
    return out;
}

} // namespace

CaveFoldResult foldCaveParity() { return runCaveParity(false); }

CaveFoldResult foldSealedCaveParity() { return runCaveParity(true); }

} // namespace lpl::engine
