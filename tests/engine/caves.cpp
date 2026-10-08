#include <lpl/engine/CharacterController.hpp>
#include <lpl/math/Cordic.hpp>
#include <lpl/procgen/CaveWarren.hpp>
#include <lpl/procgen/EndlessPlan.hpp>
#include <lpl/procgen/WorldRecipe.hpp>
#include <lpl/testing/Test.hpp>

LPL_TEST_SUITE(caves);

namespace {

constexpr lpl::core::u32 kFnv1aOffsetBasis = 0x811C9DC5u;
constexpr lpl::core::u32 kFnv1aPrime = 0x01000193u;

/**
 * @brief Ticks the walk runs for: long enough to cross the trench at a jog and be inside.
 */
constexpr lpl::core::u32 kWalkTicks = 260u;

/**
 * @brief Cells outside the shelf the walk starts from, downhill of the mouth.
 */
constexpr lpl::core::i32 kStartBack = 4;

/**
 * @brief Landmark cells searched, either side of the origin, for a site that carries a cave.
 */
constexpr lpl::core::i32 kSearchHalf = 6;

/**
 * @struct CaveFoldResult
 * @brief What gate P19 caves records: the cave and the walker both.
 *
 * @details Every earlier gate folds a world that was generated; this one folds a world that was
 *          generated and then walked, because what a player has is not that two targets build the
 *          same cave but that a body entering it ends up in the same place on both. The chain is
 *          the whole of the feature: procgen::buildCaveWarren decides the cave,
 *          procgen::caveWarrenSpanAt turns a column into a floor and a ceiling, and
 *          engine::CharacterController decides where a body may stand between them. Each link is
 *          Fixed32 and can disagree between targets on its own; folding only the first would pass
 *          a run in which the walker fell through the floor in ring 0. Only authoritative state is
 *          folded, never how any of it is drawn.
 */
struct CaveFoldResult {
    lpl::core::u32 warrenSignature{0u}; /**< FNV-1a over the cave: cells, cover, volume, adit. */
    lpl::core::u32 walkSignature{0u};   /**< FNV-1a over the body's authoritative state, every tick. */
    lpl::core::u32 spanSignature{0u};   /**< FNV-1a over the floor and ceiling along the way in. */
    lpl::core::u32 coveredColumns{0u};  /**< Columns with rock over the gallery roof. */
    lpl::core::u32 openCells{0u};       /**< Hollow cells of the cave, over every layer. */
    lpl::core::u32 reachableCells{0u};  /**< Hollow cells the flood reaches from the mouth. */
    lpl::core::u32 apertureCells{0u};   /**< Bored columns of the doorway. */
    lpl::core::u32 pathLength{0u};      /**< Steps from the mouth to the deepest reachable cell. */
    lpl::core::u32 enclosedTicks{0u};   /**< Ticks the walker spent with rock over its head. */
    lpl::core::u32 descendedLevels{0u}; /**< Voxel levels between the highest and lowest floor stood on. */
    lpl::core::u32 blocked{0u};         /**< Moves refused because the rise ahead was a wall. */
    lpl::core::u32 headBumps{0u};       /**< Ticks the body's head met a ceiling. */
    lpl::core::u32 navigable{0u};       /**< Whether the deepest gallery can be reached from the mouth. */
    lpl::core::u32 kind{0u};            /**< The resolved procgen::CaveKind. */
};

/**
 * @brief The world the walk happens in, seeded once for the gate.
 */
[[nodiscard]] lpl::procgen::EndlessPlan cavePlan()
{
    lpl::procgen::WorldRecipe recipe = lpl::procgen::parityWorldRecipe();

    recipe.seed = 0x5EEDCA7Eu;
    recipe.terrain.seed = recipe.seed;
    return lpl::procgen::endlessPlanFromRecipe(recipe, 24u);
}

/**
 * @brief The first site of the world's own landmark lattice that carries a cave.
 *
 * @details Found rather than named: which cells carry a mouth is a property of the world, and a
 *          constant here would silently stop pointing at a cave the day the terrain moved, and
 *          the gate would fold an empty warren. A synthetic warren would fold just as
 *          deterministically and prove nothing about the world anybody plays.
 *
 * @param plan   The world.
 * @param warren Receives the cave.
 * @return true when one was found within the search.
 */
[[nodiscard]] bool findCaveWarren(const lpl::procgen::EndlessPlan &plan, lpl::procgen::CaveWarren &warren)
{
    const lpl::procgen::LandmarkParams mouths = plan.rule.caveMouths;

    for (lpl::core::i32 landmarkZ = -kSearchHalf; landmarkZ <= kSearchHalf; ++landmarkZ)
    {
        for (lpl::core::i32 landmarkX = -kSearchHalf; landmarkX <= kSearchHalf; ++landmarkX)
        {
            lpl::procgen::LandmarkSite site;

            if (!lpl::procgen::landmarkAt(plan.chunk, mouths, lpl::procgen::LandmarkKind::CaveMouth, plan.rule.seaLevel,
                                          landmarkX, landmarkZ, site))
                continue;

            lpl::procgen::CaveWarren candidate =
                lpl::procgen::buildCaveWarren(plan.chunk, site, plan.rule.warren, plan.rule.caveMouthDrop);

            if (!candidate.valid)
                continue;
            warren = static_cast<lpl::procgen::CaveWarren &&>(candidate);
            return true;
        }
    }
    return false;
}

/**
 * @brief The ground at one cell, carved exactly as a chunk carves it: the raw field, lowered by
 *        the mouth this warren belongs to.
 *
 * @details Not the streamer's field: a gate that needed a resident set would need a streaming
 *          schedule, and what it folded would then depend on the order chunks arrived in.
 */
[[nodiscard]] lpl::math::Fixed32 caveGround(const lpl::procgen::EndlessPlan &plan,
                                            const lpl::procgen::CaveWarren &warren, lpl::core::i32 worldX,
                                            lpl::core::i32 worldZ)
{
    const lpl::math::Fixed32 raw = lpl::procgen::sampleWorldHeight(plan.chunk, worldX, worldZ);
    lpl::core::f32 floor = 0.0f;

    if (!lpl::procgen::caveMouthFloorAt(warren.site, warren.adit, worldX, worldZ, floor))
        return raw;

    const lpl::math::Fixed32 cut = lpl::math::Fixed32::fromFloat(floor);

    return cut < raw ? cut : raw;
}

/**
 * @struct CaveSpace
 * @brief The floor and the ceiling a body meets at a column of the gate's world.
 */
struct CaveSpace {
    const lpl::procgen::EndlessPlan &plan;  /**< The world. */
    const lpl::procgen::CaveWarren &warren; /**< The cave in it. */

    [[nodiscard]] lpl::procgen::VerticalSpan operator()(lpl::core::i32 x, lpl::core::i32 z, lpl::math::Fixed32 y) const
    {
        return lpl::procgen::caveWarrenSpanAt(warren, x, z, y, caveGround(plan, warren, x, z));
    }
};

/**
 * @brief Fills every level of the doorway columns with rock, and nothing else.
 *
 * @details The control and the run must differ in one thing, or the control proves nothing about
 *          the doorway in particular; and every level, because a wall with a gap over it is a
 *          doorway.
 */
void sealDoorway(lpl::procgen::CaveWarren &warren)
{
    for (lpl::core::u32 aperture = 0u; aperture < warren.apertureCount; ++aperture)
    {
        const lpl::core::i32 localX = warren.apertureX[aperture] - warren.originX;
        const lpl::core::i32 localZ = warren.apertureZ[aperture] - warren.originZ;

        if (localX < 0 || localZ < 0 || static_cast<lpl::core::u32>(localX) >= warren.volume.width ||
            static_cast<lpl::core::u32>(localZ) >= warren.volume.depth)
            continue;
        for (lpl::core::u32 level = 0u; level < warren.volume.levels; ++level)
            warren.volume.at(static_cast<lpl::core::u32>(localX), level, static_cast<lpl::core::u32>(localZ)) = 1u;
    }
}

/**
 * @brief Folds the floor and the ceiling along the adit, at the height the trench was cut to,
 *        before anybody walks it.
 *
 * @details Apart from the walk because the two fail differently: a span signature that moves says
 *          the targets disagree about where the rock is, and a walk signature that moves on an
 *          unchanged span says they disagree about what a body does with it.
 */
[[nodiscard]] lpl::core::u32 foldSpanAlongAdit(const CaveSpace &space)
{
    const lpl::procgen::CaveWarren &warren = space.warren;
    const lpl::math::Fixed32 floorY = lpl::math::Fixed32::fromFloat(warren.adit.floorY);
    lpl::core::u32 hash = kFnv1aOffsetBasis;

    for (lpl::core::i32 step = -kStartBack; step <= static_cast<lpl::core::i32>(lpl::procgen::kMaxAditCells); ++step)
    {
        const lpl::core::i32 cellX = warren.site.cellX + warren.adit.stepX * step;
        const lpl::core::i32 cellZ = warren.site.cellZ + warren.adit.stepZ * step;
        const lpl::procgen::VerticalSpan span = space(cellX, cellZ, floorY);

        hash = (hash ^ static_cast<lpl::core::u32>(span.floor.raw())) * kFnv1aPrime;
        hash = (hash ^ static_cast<lpl::core::u32>(span.ceiling.raw())) * kFnv1aPrime;
        hash = (hash ^ (span.enclosed ? 1u : 0u)) * kFnv1aPrime;
    }
    return hash;
}

/**
 * @brief The heading that walks a body up the adit, derived through CORDIC.
 *
 * @details The body's convention is wish = (-forward * sin(yaw), -forward * cos(yaw)), so the
 *          heading along (stepX, stepZ) is the arctangent of their negations: derived rather than
 *          tabulated, or the gate would only work for a site whose adit runs north.
 */
[[nodiscard]] lpl::math::Fixed32 headingUpTheAdit(const lpl::procgen::CaveWarren &warren)
{
    return lpl::math::Cordic::atan2(lpl::math::Fixed32::fromInt(-warren.adit.stepX),
                                    lpl::math::Fixed32::fromInt(-warren.adit.stepZ));
}

/**
 * @brief The voxel level a body inside the cave stands on.
 *
 * @details In levels rather than metres: a level is the unit a gallery is stacked in, so a descent
 *          of two levels is a statement about the cave, and one of 2.8 metres about this scale.
 */
[[nodiscard]] lpl::core::i32 levelOf(const lpl::engine::CharacterController &body,
                                     const lpl::procgen::CaveWarren &warren)
{
    return (body.y().raw() - warren.baseYFixed.raw()) / warren.levelHeightFixed.raw();
}

/**
 * @brief Walks a body from below the mouth up the adit, and folds where it goes.
 *
 * @param space The world and its cave.
 * @param out   Receives the walk's signature and counters.
 */
void walkUpTheAdit(const CaveSpace &space, CaveFoldResult &out)
{
    const lpl::procgen::CaveWarren &warren = space.warren;
    const lpl::core::i32 startX = warren.site.cellX - warren.adit.stepX * kStartBack;
    const lpl::core::i32 startZ = warren.site.cellZ - warren.adit.stepZ * kStartBack;
    const lpl::math::Fixed32 tick = lpl::math::Fixed32::fromFloat(1.0f / 60.0f);
    lpl::engine::CharacterController body;
    lpl::engine::CharacterParams params{};
    lpl::engine::CharacterIntent walk{};
    lpl::core::u32 walkHash = kFnv1aOffsetBasis;
    lpl::core::i32 highestLevel = 0;
    lpl::core::i32 lowestLevel = 0;
    bool sawFloor = false;

    body.placeAt(lpl::math::Fixed32::fromInt(startX), lpl::math::Fixed32::fromInt(startZ),
                 caveGround(space.plan, warren, startX, startZ), space);
    body.setYaw(headingUpTheAdit(warren));
    walk.forward = lpl::math::Fixed32::one();
    for (lpl::core::u32 elapsed = 0u; elapsed < kWalkTicks; ++elapsed)
    {
        body.step(params, walk, tick, space);
        walkHash = (walkHash ^ body.fold()) * kFnv1aPrime;
        out.enclosedTicks += body.isEnclosed() ? 1u : 0u;
        if (!body.isEnclosed())
            continue;

        const lpl::core::i32 level = levelOf(body, warren);

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
    out.descendedLevels = sawFloor ? static_cast<lpl::core::u32>(highestLevel - lowestLevel) : 0u;
    out.blocked = body.blockedCount();
    out.headBumps = body.headBumpCount();
}

/**
 * @brief Builds the gate's cave, walks a body into it, and folds both.
 *
 * @param sealed When set, the doorway is filled with rock before the walk: the control, in which
 *               the body must not get inside.
 * @return The fold, all zero when the world carries no cave within the search.
 */
[[nodiscard]] CaveFoldResult foldCaveWalk(bool sealed)
{
    CaveFoldResult out;
    const lpl::procgen::EndlessPlan plan = cavePlan();
    lpl::procgen::CaveWarren warren;

    if (!findCaveWarren(plan, warren))
        return out;
    if (sealed)
        sealDoorway(warren);
    out.warrenSignature = lpl::procgen::foldCaveWarren(warren);
    out.coveredColumns = warren.coveredColumns;
    out.openCells = warren.openCells;
    out.reachableCells = warren.reachableCells;
    out.apertureCells = warren.apertureCount;
    out.pathLength = warren.pathLength;
    out.navigable = warren.navigable ? 1u : 0u;
    out.kind = static_cast<lpl::core::u32>(warren.kind);

    const CaveSpace space{plan, warren};

    out.spanSignature = foldSpanAlongAdit(space);
    walkUpTheAdit(space, out);
    return out;
}

} // namespace

/**
 * @brief Gate P19 caves: a body walking at a cave's mouth ends up under rock, and the same walk at
 *        a mouth filled with rock is stopped, with the same signatures on both targets.
 *
 * @details The sealed run is the control: a collider that let everything through would put a body
 *          inside the cave, and through a mountain as well. The enclosed ticks make the gate
 *          discriminating rather than merely stable: two targets would agree perfectly about a
 *          walker that never got in.
 */
LPL_TEST(a_body_walks_in_and_rock_stops_it)
{
    const CaveFoldResult open = foldCaveWalk(false);
    const CaveFoldResult sealed = foldCaveWalk(true);

    test.check(open.warrenSignature != 0u, "the gate finds a cave to walk into");
    test.check(open.enclosedTicks > 0u, "a body walking at the mouth ends up under rock");
    test.check(sealed.enclosedTicks == 0u, "and a mouth filled with rock lets nobody in");
    test.check(sealed.blocked > open.blocked, "the sealed walk is stopped, not merely slower");
    test.check(open.walkSignature != sealed.walkSignature, "the two walks are different runs");
    test.check(open.spanSignature != sealed.spanSignature, "and disagree about where the rock is");

    test.measureHexadecimal("warren_signature", open.warrenSignature);
    test.measureHexadecimal("walk_signature", open.walkSignature);
    test.measureHexadecimal("span_signature", open.spanSignature);
    test.measureHexadecimal("sealed_walk_signature", sealed.walkSignature);
    test.measure("covered_columns", open.coveredColumns);
    test.measure("open_cells", open.openCells);
    test.measure("reachable_cells", open.reachableCells);
    test.measure("aperture_cells", open.apertureCells);
    test.measure("path_length", open.pathLength);
    test.measure("enclosed_ticks", open.enclosedTicks);
    test.measure("descended_levels", open.descendedLevels);
    test.measure("blocked", open.blocked);
    test.measure("head_bumps", open.headBumps);
    test.measure("navigable", open.navigable);
    test.measure("kind", open.kind);
    test.measure("sealed_blocked", sealed.blocked);
}
