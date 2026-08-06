/**
 * @file CaveWarren.hpp
 * @brief A cave you can walk into, in a world that has no edges.
 *
 * @ref generateCaveSystem already produces everything a cave is: layers, shafts,
 * a reachability repair, a playability judgement in three dimensions, and a voxel
 * volume. What it cannot do is exist in a streamed world, and the reason is the
 * same one erosion and drainage have — it takes a BOUNDED heightfield and relaxes
 * over the whole of it. A world with no total has nothing to relax over.
 *
 * The answer is not a chunked cave generator. It is the one @ref planVillage
 * already uses, one concept up: **a cave system is a landmark**. A
 * @ref LandmarkSite is a pure function of its own landmark cell, so a bounded
 * world derived from a site is a pure function of that cell too — every chunk
 * that asks gets the same warren, and the global passes run over the warren
 * rather than over the world. Nothing here is a second cave generator.
 *
 * Four things in it are not obvious and each is the answer to a failure:
 *
 *  - **A heightfield cannot have a hole**, so the way in is HORIZONTAL. The site's
 *    shelf is extended into the hill as a trench — an adit — cut level with the
 *    gallery floor, and the mouth is the first cell along it with enough rock
 *    overhead. Cutting a vertical chimney instead would need the terrain patch to
 *    skip cells, which is renderer surgery for a worse result.
 *  - **The cave exists only where there is rock over it.** A column whose surface
 *    is not at least @ref CaveWarrenParams::coverMargin above the gallery roof is
 *    NOT part of the warren: the terrain answers there. Without that rule an open
 *    hillside is pitted with holes into a gallery, which is what a volume clipped
 *    to a footprint rather than to its cover actually looks like.
 *  - **Cover is measured on the CARVED surface, not the raw one.** The adit lowers
 *    the ground, so a trench cell reads as covered on the raw field and is bare
 *    sky in the world. Both this and @ref ChunkTerrain go through
 *    @ref caveMouthFloorAt, so the surface the warren reasons about is the surface
 *    that will exist.
 *  - **The floor and the ceiling are one query.** Under a hill there is ground
 *    below a body and rock above it, and a body can be between them; a heightfield
 *    answers only the first. @ref VerticalSpan is that pair, and it is what the
 *    walking body collides against.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-08-06
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_PROCGEN_CAVE_WARREN_HPP
#    define LPL_PROCGEN_CAVE_WARREN_HPP

#    include <lpl/procgen/CaveSystem.hpp>
#    include <lpl/procgen/Chunking.hpp>
#    include <lpl/procgen/Landmark.hpp>
#    include <lpl/procgen/VerticalSpan.hpp>

namespace lpl::procgen {

/// Cells an adit may be driven before the site is refused. Bounds the search and the array.
inline constexpr core::u32 kMaxAditCells = 32u;

/// Columns of a warren's doorway. The bore is a handful of cells wide and deep.
inline constexpr core::u32 kMaxApertureCells = 24u;

/**
 * @brief Whether anybody lives within reach of a site.
 *
 * The evidence @ref chooseCaveKind wants for "people dig rooms", asked of the SAME
 * settlement lattice the terrain sites its villages from — a second lattice would
 * credit a mine to a village that is not on the map. A pure function of the
 * coordinates, so two chunks asking about one warren agree about what kind of place
 * it is.
 *
 * Public because a readout and a test both want to see the evidence rather than only
 * the verdict: "this world has no dungeons" and "this rule never fires" are different
 * problems with the same symptom.
 *
 * @param params   World parameters.
 * @param villages The settlement lattice.
 * @param seaLevel Where the water is.
 * @param site     The cave site.
 * @param reach    World cells within which a village counts.
 * @return true when a settlement site falls within reach.
 */
[[nodiscard]] bool settledNearSite(const ChunkParams &params, const LandmarkParams &villages, core::f32 seaLevel,
                                   const LandmarkSite &site, core::u32 reach);

/**
 * @struct CaveWarrenParams
 * @brief How big a warren is, how deep it goes, and how much rock it needs over it.
 */
struct CaveWarrenParams {
    /**
     * @brief Cells from the site centre to the edge of the warren.
     *
     * Bounded by RESIDENCY, not by taste: a body inside the warren must have the
     * chunk that owns it loaded, and that chunk is at most
     * `generateRadius * chunkSize` cells away. Twenty against a radius of two and
     * a chunk of twenty-four leaves a factor of two in hand.
     */
    core::u32 halfSpan{20u};

    core::u32 layers{3u};          ///< Stacked galleries; deeper ones open out.
    core::u32 levelsPerLayer{2u};  ///< Voxel levels a gallery occupies.
    core::f32 levelHeight{1.4f};   ///< World units per voxel level.

    /**
     * @brief Solid levels ABOVE the shallowest gallery. The cave's roof.
     *
     * Not decoration, and the thing this design was missing: @ref caveVolume stops at
     * the top gallery, so its ceiling is where the array ends and a mesher emits no
     * face there at all. A body standing in the gallery would look up at nothing —
     * which in a scene with the sky suppressed is the background, and in a scene
     * without it is the sky. "You are underground" is a claim about what is OVER you,
     * so the rock over you has to exist.
     *
     * It is also what the mouth is a hole IN: the terrain quad across the opening is
     * skipped, and this is the lintel behind it.
     */
    core::u32 capLevels{1u};

    /**
     * @brief Rock a column needs over the gallery roof before the cave exists there.
     *
     * The whole reason a warren has a shape rather than a footprint. Too small and
     * the gallery breaks the surface in a rash of holes; too large and the cave
     * shrinks to nothing under any hill that is not a cliff.
     *
     * Measured over 441 landmark cells of the walked world — 68 sited mouths — against
     * the cap above it, since the two add up into one demand on the hillside:
     *
     *   | cap | margin | caves | reach the bottom | trench | open cells |
     *   |-----|--------|-------|------------------|--------|------------|
     *   |  1  |  0.4   |  47   |        27        |  7.1   |    321     |
     *   |  1  |  1.2   |  37   |        19        |  8.3   |    239     |
     *   |  2  |  0.8   |  29   |        16        |  9.2   |    190     |
     *   |  2  |  1.2   |  26   |        14        |  9.3   |    174     |
     *
     * The bottom row is what a two-level cap and a generous margin cost: a third of
     * the caves, and a trench two cells longer to reach each one — a nine-cell ravine
     * cut into a hillside reads as a quarry, not as a way in. One level of rock plus
     * four tenths of a metre of ground over it is 1.8 m of roof, which is a roof.
     */
    core::f32 coverMargin{0.4f};

    core::u32 aditReach{18u};      ///< Cells the adit may be driven before giving up.
    core::u32 aditHalfWidth{1u};   ///< Trench half-width, in cells.
    /**
     * @brief Cells the adit bores past the mouth before the plan takes over.
     *
     * The forced part of the gallery. Zero would leave the way in at the mercy of
     * whatever the automaton happened to put behind the mouth, which is a cave you
     * can see into and not enter — the failure this whole file exists to avoid.
     */
    core::u32 aditBore{4u};

    core::f32 topFill{0.46f};      ///< Fill probability of the shallowest gallery.
    core::f32 deepFill{0.38f};     ///< Fill probability of the deepest.
    core::u32 automatonSteps{5u};
    core::u32 minChamberSize{18u};
    core::u32 shaftsPerPair{3u};

    core::u8 rockMaterial{1u};     ///< Material id written for solid rock.

    // ── What kind of place this is ──────────────────────────────────────────

    /**
     * @brief What the document asked for, straight from @ref WorldRecipe::caveKind.
     *
     * @c Auto is the default and is the interesting case: a streamed world has
     * thousands of these, and naming one generator for all of them is how every cave
     * in a world ends up identical. It resolves through @ref chooseCaveKind, PER SITE,
     * from evidence the site itself carries — so two chunks asking about one warren
     * still get one answer, which is the property the whole landmark scheme rests on.
     */
    CaveKind kind{CaveKind::Auto};

    /**
     * @brief The settlement lattice, for the "were there people here" question.
     *
     * The same @ref LandmarkParams the terrain sites villages with, or the answer here
     * would be about a village that is not on the map. Read only when @ref kind is
     * @c Auto.
     */
    LandmarkParams villages{settlementDefaults()};
    core::f32 seaLevel{-1.0f};      ///< Where the water is, for the settlement query.
    /**
     * @brief How far a settlement counts as "here", in world cells.
     *
     * Generous on purpose: a mine is not under the market square, it is over the hill
     * from it. Small enough that a warren the far side of a valley is not credited to
     * a town it has nothing to do with.
     */
    core::u32 settlementReach{72u};
};

/** @brief The defaults a walked world uses. */
[[nodiscard]] constexpr CaveWarrenParams caveWarrenDefaults() noexcept { return CaveWarrenParams{}; }

/**
 * @brief How high the ground must stand over a trench floor before a cave fits under it.
 *
 * ONE statement of what "covered" means, because there are two places that have to
 * agree about it and they are not next to each other: @ref planCaveAdit walks uphill
 * looking for the first covered cell, and @ref buildCaveWarren masks every uncovered
 * column out of the plan. Written twice, they drifted the moment the rock cap was
 * added — the adit found a mouth by the old rule, the mask called that same cell bare,
 * and forty-eight of fifty-two caves silently built nothing. The symptom was a warren
 * that produced no geometry, which looks nothing like a disagreement about a threshold.
 *
 * @param warren Gallery geometry.
 * @param floorY The trench floor, i.e. the top gallery's floor.
 * @return The surface height at or above which a column is part of the cave.
 */
[[nodiscard]] constexpr core::f32 caveCoverThreshold(const CaveWarrenParams &warren, core::f32 floorY) noexcept
{
    const core::u32 perLayer = warren.levelsPerLayer == 0u ? 1u : warren.levelsPerLayer;
    const core::f32 gallery = static_cast<core::f32>(perLayer) * warren.levelHeight;
    const core::f32 cap = static_cast<core::f32>(warren.capLevels) * warren.levelHeight;
    return floorY + gallery + cap + warren.coverMargin;
}

/**
 * @struct CaveAdit
 * @brief The trench that leads into the hill, and where the rock closes over it.
 *
 * A pure function of the site, so the chunk that carves the ground and the warren
 * that reasons about cover derive the same one. Two derivations of a trench would be
 * two answers to where the mouth is, and the mouth is the only place the two halves
 * of this feature meet.
 */
struct CaveAdit {
    core::i32 cellX[kMaxAditCells]{}; ///< Trench cells, from the site centre outward.
    core::i32 cellZ[kMaxAditCells]{};
    core::u32 length{0u};             ///< Trench cells actually cut.
    core::i32 mouthX{0};              ///< First cell with rock over the gallery.
    core::i32 mouthZ{0};
    core::i32 stepX{0};               ///< Uphill step, one cell.
    core::i32 stepZ{0};
    core::i32 halfWidth{1};           ///< Trench half-width; travels with the plan so
                                      ///< the carve and the cover mask agree on it.
    core::f32 floorY{0.0f};           ///< The one level the trench is cut to.
    bool found{false};                ///< False when no cell within reach had cover.
};

/**
 * @brief Drives an adit uphill from a site until the ground can roof a gallery.
 *
 * Uphill is the opposite of @ref LandmarkSite::facing, which already names the way
 * down. The search is bounded and it is allowed to FAIL: a site with a qualifying
 * relief may still have no direction in which the ground rises far enough within
 * reach, and a warren forced onto it would be a gallery with daylight through its
 * roof. A refusal is a site with no cave, which is a world with fewer caves in it
 * and not a world with broken ones.
 *
 * @param params  World parameters.
 * @param site    The cave-mouth site.
 * @param warren  Gallery geometry; only the height and the cover margin are read.
 * @param drop    How far the shelf is cut below the site, @ref ChunkTerrainRule::caveMouthDrop.
 * @return The adit; @c found is false when the site cannot carry a cave.
 */
[[nodiscard]] CaveAdit planCaveAdit(const ChunkParams &params, const LandmarkSite &site,
                                    const CaveWarrenParams &warren, core::f32 drop);

/**
 * @brief The level a cave mouth cuts one world cell down to, if it cuts it at all.
 *
 * The single statement of what a cave mouth does to the ground, called once per cell
 * by the chunk that generates terrain and once per cell by the warren that measures
 * its own cover. It LOWERS and never raises, so a cell already below the floor is
 * left alone — setting it would build a plinth out into the valley.
 *
 * @param site     The site.
 * @param adit     Its adit, from @ref planCaveAdit.
 * @param worldX   Cell to test.
 * @param worldZ   Cell to test.
 * @param outFloor Receives the floor when the cell is cut.
 * @return true when this cell belongs to the shelf or the trench.
 */
[[nodiscard]] bool caveMouthFloorAt(const LandmarkSite &site, const CaveAdit &adit, core::i32 worldX, core::i32 worldZ,
                                    core::f32 &outFloor);

/**
 * @struct CaveWarren
 * @brief One bounded cave system, anchored under a site and placed in the world.
 *
 * The volume is indexed in ABSOLUTE world levels: level @c l spans world Y in
 * `[baseY + l * levelHeight, baseY + (l + 1) * levelHeight)`. A relative volume plus
 * a lift computed at each call site is the same information with one more chance of
 * disagreeing about where the floor is.
 */
struct CaveWarren {
    LandmarkSite site{};
    CaveAdit adit{};
    VoxelVolume volume{};      ///< Rock and air; layer 0 at the top, see @ref caveVolume.
    Grid<core::u8> covered{};  ///< 1 where the ground can roof the gallery, 0 where the terrain rules.

    core::i32 originX{0};      ///< World column of the volume's (0, *, 0) corner.
    core::i32 originZ{0};      ///< World row of it.
    core::f32 baseY{0.0f};     ///< World Y at the bottom of level 0, for the renderer.
    core::f32 levelHeight{1.4f};

    /**
     * @brief The same two numbers, quantised once, for the collider.
     *
     * A span decides where a body may stand, so it is authoritative and may not go
     * through a float. Quantising HERE rather than at every query is the point:
     * the rounding happens once per warren instead of once per step, so two targets
     * cannot round it differently on different ticks.
     */
    math::Fixed32 baseYFixed{};
    math::Fixed32 levelHeightFixed{};

    /**
     * @brief The doorway: columns where the gallery meets open air.
     *
     * The bored columns that touch an uncovered one, in WORLD cells. A renderer skips
     * the terrain quad at each of them, and that skip is the difference between a cave
     * you can see into and a cliff you walk through. Small and fixed-size: a doorway is
     * a doorway, and a warren that wanted a hundred of them would be a quarry.
     */
    core::i32 apertureX[kMaxApertureCells]{};
    core::i32 apertureZ[kMaxApertureCells]{};
    core::u32 apertureCount{0u};

    core::u32 layerCount{0u};
    core::u32 openCells{0u};       ///< Hollow cells of the system, over every layer.
    core::u32 coveredColumns{0u};  ///< Columns with rock over the gallery roof.
    core::u32 reachableCells{0u};  ///< Hollow cells the flood reaches from the mouth.
    core::u32 pathLength{0u};      ///< Steps from the mouth to the deepest reachable cell.
    core::u32 repairedCells{0u};
    /**
     * @brief What kind of place this turned out to be. Never @c Auto.
     *
     * Kept rather than recomputed because a readout, a renderer and a test all want to
     * know, and re-deriving it in three places is three chances to answer differently
     * from the generator that actually ran.
     */
    CaveKind kind{CaveKind::Cellular};
    /**
     * @brief Whether the deepest gallery can be reached from the mouth.
     *
     * @ref evaluateCaveSystem's verdict, kept because it is the one property a
     * player can feel: a warren that fails it is a mouth leading to one room.
     */
    bool navigable{false};
    bool valid{false};             ///< False when the site carried no adit, or no cave.

    [[nodiscard]] core::u32 levels() const noexcept { return volume.levels; }
    /** @brief World Y at the top of the volume: the top of the rock cap. */
    [[nodiscard]] core::f32 topY() const noexcept
    {
        return baseY + static_cast<core::f32>(volume.levels) * levelHeight;
    }

    /** @brief Whether a world cell is part of the doorway. */
    [[nodiscard]] bool isAperture(core::i32 worldX, core::i32 worldZ) const noexcept
    {
        for (core::u32 i = 0u; i < apertureCount; ++i)
            if (apertureX[i] == worldX && apertureZ[i] == worldZ)
                return true;
        return false;
    }

    /** @brief Whether a world cell falls inside the volume's footprint. */
    [[nodiscard]] bool containsCell(core::i32 worldX, core::i32 worldZ) const noexcept
    {
        if (volume.empty())
            return false;
        const core::i32 lx = worldX - originX;
        const core::i32 lz = worldZ - originZ;
        return lx >= 0 && lz >= 0 && static_cast<core::u32>(lx) < volume.width &&
               static_cast<core::u32>(lz) < volume.depth;
    }

    /** @brief Whether a world cell is part of the cave rather than of the terrain. */
    [[nodiscard]] bool isCavernous(core::i32 worldX, core::i32 worldZ) const noexcept
    {
        if (!containsCell(worldX, worldZ) || covered.empty())
            return false;
        return covered.at(static_cast<core::u32>(worldX - originX), static_cast<core::u32>(worldZ - originZ)) != 0u;
    }
};

/**
 * @brief Builds the warren under one cave-mouth site.
 *
 * The bounded passes @ref generateCaveSystem already owns, run over a world derived
 * from the site alone: layers, inter-layer shafts, the forced bore that guarantees a
 * way in, the reachability repair from that way in, and the judgement of whether the
 * bottom can be reached from the top.
 *
 * @warning Not cheap: a cellular automaton per layer plus a flood per repair round.
 *          It runs once, on the tick the chunk that OWNS the site is generated, and
 *          the chunk budget is one chunk per tick for exactly this reason.
 *
 * @param params World parameters.
 * @param site   The site, from @ref landmarkAt.
 * @param warren Gallery geometry and generator budget.
 * @param drop   The shelf depth the chunk carves with.
 * @return The warren; @c valid is false when the site cannot carry one.
 */
[[nodiscard]] CaveWarren buildCaveWarren(const ChunkParams &params, const LandmarkSite &site,
                                         const CaveWarrenParams &warren, core::f32 drop);

/**
 * @brief The gap a body stands in at one column of a warren.
 *
 * @param warren   The warren.
 * @param worldX   Column.
 * @param worldZ   Row.
 * @param y        Where the body is, world units.
 * @param terrain  The drawn ground height at that column — what answers wherever the
 *                 cave does not.
 * @return The span. Outside the cavernous part of the footprint this is exactly the
 *         heightfield's answer, so a caller may ask unconditionally.
 */
[[nodiscard]] VerticalSpan caveWarrenSpanAt(const CaveWarren &warren, core::i32 worldX, core::i32 worldZ,
                                            math::Fixed32 y, math::Fixed32 terrain);

/**
 * @brief FNV-1a over everything a warren decides, for determinism checks.
 *
 * The system, the cover mask, the volume and the adit. The adit is folded because it
 * is the only part that both this module and @ref ChunkTerrain act on, so a target
 * that derived a different mouth would carve different ground and still fold an
 * identical cave.
 *
 * @param warren The warren.
 * @return The 32-bit signature.
 */
[[nodiscard]] core::u32 foldCaveWarren(const CaveWarren &warren);

} // namespace lpl::procgen

#endif // LPL_PROCGEN_CAVE_WARREN_HPP
