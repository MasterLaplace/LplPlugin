/**
 * @file CaveWarren.cpp
 * @brief Implementation of the streamable cave system.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-08-06
 * @copyright MIT License
 */

#include <lpl/procgen/CaveWarren.hpp>

namespace lpl::procgen {

namespace {

constexpr core::u32 kFnv1aOffsetBasis = 0x811C9DC5u;
constexpr core::u32 kFnv1aPrime = 0x01000193u;

/**
 * @brief Floor division of two raw fixed-point words.
 *
 * Truncation would fold the world about its own datum: a body one raw unit under
 * the volume's base and one raw unit over it would both land in level 0, and a
 * gallery would gain a floor it does not have wherever the terrain crosses zero.
 *
 * @param value Numerator.
 * @param step  Denominator; must not be zero.
 * @return The largest integer not greater than the quotient.
 */
[[nodiscard]] core::i32 floorDivRaw(core::i32 value, core::i32 step) noexcept
{
    return value >= 0 ? value / step : -(((-value) + step - 1) / step);
}

/**
 * @brief The shelf a cave-mouth site cuts, on its own, before any adit.
 *
 * Split out of @ref caveMouthFloorAt because @ref planCaveAdit needs exactly this
 * half: it measures cover along a ray, and the ray starts inside the shelf. Testing
 * cover against the RAW ground there would find the hill the shelf was cut out of
 * and put the mouth inside the shelf, which is a cave mouth with no cave behind it.
 *
 * @param site     The site.
 * @param worldX   Cell to test.
 * @param worldZ   Cell to test.
 * @param drop     How far the shelf is cut below the site.
 * @param outFloor Receives the shelf floor when the cell is on it.
 * @return true when this cell is on the shelf.
 */
[[nodiscard]] bool shelfFloorAt(const LandmarkSite &site, core::i32 worldX, core::i32 worldZ, core::f32 drop,
                                core::f32 &outFloor) noexcept
{
    const core::i32 dx = worldX - site.cellX;
    const core::i32 dz = worldZ - site.cellZ;
    const core::i32 radius = static_cast<core::i32>(site.radius);
    if (dx * dx + dz * dz > radius * radius)
        return false;
    outFloor = site.height - drop;
    return true;
}

/**
 * @brief The ground a cave site actually leaves behind at one cell.
 *
 * The raw field lowered by whatever of the shelf and the trench reaches this cell.
 * Both this and @ref ChunkTerrain go through @ref caveMouthFloorAt, so there is one
 * statement of what a mouth does to the ground rather than one per consumer.
 */
[[nodiscard]] core::f32 carvedSurfaceAt(const ChunkParams &params, const LandmarkSite &site, const CaveAdit &adit,
                                        core::i32 worldX, core::i32 worldZ)
{
    core::f32 height = sampleWorldHeight(params, worldX, worldZ).toFloat();
    core::f32 floor = 0.0f;
    if (caveMouthFloorAt(site, adit, worldX, worldZ, floor) && floor < height)
        height = floor;
    return height;
}

/**
 * @brief Opens one column of the forced bore, and floors it.
 *
 * The layer BELOW is walled as well as the top layer opened, and that is the half
 * that is easy to leave out: without it the gallery's floor is whatever the second
 * automaton happened to put under the way in, so a walker steps through the mouth
 * and falls two storeys. Being able to enter and being able to stand are different
 * claims.
 *
 * @param system  The system being bored.
 * @param covered Cover mask, in the same local coordinates.
 * @param x       Local column.
 * @param z       Local row.
 * @return true when the column was covered and therefore bored.
 */
bool boreColumn(CaveSystem &system, const Grid<core::u8> &covered, core::i32 x, core::i32 z)
{
    if (!covered.contains(x, z))
        return false;
    const core::u32 lx = static_cast<core::u32>(x);
    const core::u32 lz = static_cast<core::u32>(z);
    if (covered.at(lx, lz) == 0u)
        return false;

    system.layer[0].at(lx, lz) = DungeonCell::Floor;
    if (system.layerCount > 1u)
        system.layer[1].at(lx, lz) = DungeonCell::Wall;
    return true;
}

/// @copydoc settledNearSite
[[nodiscard]] bool settledNearImpl(const ChunkParams &params, const LandmarkParams &villages, core::f32 seaLevel,
                                   const LandmarkSite &site, core::u32 reach)
{
    if (villages.cellSpan == 0u)
        return false;
    const core::i32 span = static_cast<core::i32>(villages.cellSpan);
    const core::i32 far = static_cast<core::i32>(reach);
    const core::i32 fromX = floorDivChunk(site.cellX - far, span);
    const core::i32 toX = floorDivChunk(site.cellX + far, span);
    const core::i32 fromZ = floorDivChunk(site.cellZ - far, span);
    const core::i32 toZ = floorDivChunk(site.cellZ + far, span);

    for (core::i32 lz = fromZ; lz <= toZ; ++lz)
        for (core::i32 lx = fromX; lx <= toX; ++lx)
        {
            LandmarkSite village;
            if (!landmarkAt(params, villages, LandmarkKind::Settlement, seaLevel, lx, lz, village))
                continue;
            const core::i32 dx = village.cellX - site.cellX;
            const core::i32 dz = village.cellZ - site.cellZ;
            if (dx * dx + dz * dz <= far * far)
                return true;
        }
    return false;
}

} // namespace

bool settledNearSite(const ChunkParams &params, const LandmarkParams &villages, core::f32 seaLevel,
                     const LandmarkSite &site, core::u32 reach)
{
    return settledNearImpl(params, villages, seaLevel, site, reach);
}

CaveAdit planCaveAdit(const ChunkParams &params, const LandmarkSite &site, const CaveWarrenParams &warren,
                      core::f32 drop)
{
    CaveAdit adit;
    adit.floorY = site.height - drop;
    adit.halfWidth = static_cast<core::i32>(warren.aditHalfWidth);

    // Uphill is the opposite of the way down, which the site already carries. Deriving
    // it again from the height field would be a second answer to which way the hill is.
    const core::u32 facing = site.facing & 7u;
    adit.stepX = -kNeighbor8X[facing];
    adit.stepZ = -kNeighbor8Z[facing];
    if (adit.stepX == 0 && adit.stepZ == 0)
        return adit;

    // What the ground has to reach before it can roof a gallery. The SAME rule
    // @ref buildCaveWarren masks its columns with — see @ref caveCoverThreshold for
    // what happened when it was written out twice.
    const core::f32 needed = caveCoverThreshold(warren, adit.floorY);

    // Bounded by the FOOTPRINT as well as by the array, and the first bound is the one
    // that matters: a mouth found further out than `halfSpan` is a mouth outside the
    // volume, so nothing can be bored at it and the warren silently builds nothing.
    // Measured — at halfSpan 10 that was five of the fifty-two sited caves.
    core::u32 reach = warren.aditReach < kMaxAditCells ? warren.aditReach : kMaxAditCells;
    if (reach > warren.halfSpan)
        reach = warren.halfSpan;
    for (core::u32 i = 1u; i <= reach; ++i)
    {
        const core::i32 cellX = site.cellX + adit.stepX * static_cast<core::i32>(i);
        const core::i32 cellZ = site.cellZ + adit.stepZ * static_cast<core::i32>(i);

        // The shelf, but NOT the trench: the trench is what this loop is deciding, and
        // every cell it has passed is one it will cut to `floorY` — which is below
        // `needed` by construction, so a cut cell can never read as covered anyway.
        core::f32 height = sampleWorldHeight(params, cellX, cellZ).toFloat();
        core::f32 shelf = 0.0f;
        if (shelfFloorAt(site, cellX, cellZ, drop, shelf) && shelf < height)
            height = shelf;

        if (height >= needed)
        {
            adit.mouthX = cellX;
            adit.mouthZ = cellZ;
            adit.found = true;
            return adit;
        }

        if (adit.length < kMaxAditCells)
        {
            adit.cellX[adit.length] = cellX;
            adit.cellZ[adit.length] = cellZ;
            ++adit.length;
        }
    }

    // No cover within reach. A site with a qualifying relief can still have no
    // direction in which the ground rises far enough, and a warren forced onto it
    // would be a gallery with daylight through its roof.
    adit.length = 0u;
    return adit;
}

bool caveMouthFloorAt(const LandmarkSite &site, const CaveAdit &adit, core::i32 worldX, core::i32 worldZ,
                      core::f32 &outFloor)
{
    if (shelfFloorAt(site, worldX, worldZ, site.height - adit.floorY, outFloor))
        return true;
    if (!adit.found)
        return false;

    // The cut stops AT the mouth, and this test is the whole of that rule. Without
    // it the trench's own width spills one cell past its last step and lowers the
    // mouth to the trench floor — which takes the rock off the very roof the mouth
    // is a hole in. Measured: it left every one of fifty-two sited caves with an
    // uncovered mouth and therefore no way in, and the symptom was a warren that
    // built nothing rather than one that built something wrong.
    const core::i32 aheadX = worldX - adit.mouthX;
    const core::i32 aheadZ = worldZ - adit.mouthZ;
    if (aheadX * adit.stepX + aheadZ * adit.stepZ >= 0)
        return false;

    // The trench, as a chain of small discs rather than a rasterised line: a diagonal
    // adit drawn cell by cell is one cell wide across its corners and pinches shut
    // there, and a passage a body cannot fit through is the same as no passage.
    const core::i32 half = adit.halfWidth;
    for (core::u32 i = 0u; i < adit.length; ++i)
    {
        const core::i32 dx = worldX - adit.cellX[i];
        const core::i32 dz = worldZ - adit.cellZ[i];
        const core::i32 spanX = dx < 0 ? -dx : dx;
        const core::i32 spanZ = dz < 0 ? -dz : dz;
        if (spanX <= half && spanZ <= half)
        {
            outFloor = adit.floorY;
            return true;
        }
    }
    return false;
}

CaveWarren buildCaveWarren(const ChunkParams &params, const LandmarkSite &site, const CaveWarrenParams &warren,
                           core::f32 drop)
{
    CaveWarren out;
    out.site = site;
    out.levelHeight = warren.levelHeight;
    out.levelHeightFixed = math::Fixed32::fromFloat(warren.levelHeight);
    out.adit = planCaveAdit(params, site, warren, drop);
    if (!out.adit.found || warren.halfSpan == 0u)
        return out;

    const core::u32 span = warren.halfSpan * 2u + 1u;
    const core::u32 layers =
        warren.layers == 0u ? 1u : (warren.layers > kMaxCaveLayers ? kMaxCaveLayers : warren.layers);
    const core::u32 perLayer = warren.levelsPerLayer == 0u ? 1u : warren.levelsPerLayer;
    const core::u32 galleryLevels = layers * perLayer;
    const core::u32 levels = galleryLevels + warren.capLevels;

    out.originX = site.cellX - static_cast<core::i32>(warren.halfSpan);
    out.originZ = site.cellZ - static_cast<core::i32>(warren.halfSpan);
    out.layerCount = layers;

    // The top gallery's FLOOR sits at the trench floor, so walking in is level. Every
    // other anchor follows from that one, which is why it is written once here rather
    // than being a lift each consumer adds.
    const core::f32 shelfFloor = out.adit.floorY;
    const core::f32 gallery = static_cast<core::f32>(perLayer) * warren.levelHeight;
    const core::f32 roofY = shelfFloor + gallery;
    const core::f32 topY = roofY + static_cast<core::f32>(warren.capLevels) * warren.levelHeight;
    out.baseY = topY - static_cast<core::f32>(levels) * warren.levelHeight;
    out.baseYFixed = math::Fixed32::fromFloat(out.baseY);

    // ── Cover: where there is rock enough over the roof for a cave to be a cave ──
    //
    // Measured against the top of the CAP, not the top of the gallery: the cap is rock
    // this warren puts there itself, so ground that only clears the gallery would have
    // the cave's own roof sticking out of the hillside.
    const core::f32 needed = caveCoverThreshold(warren, shelfFloor);
    out.covered = Grid<core::u8>{span, span, core::u8{0}};
    for (core::u32 z = 0u; z < span; ++z)
        for (core::u32 x = 0u; x < span; ++x)
        {
            const core::i32 worldX = out.originX + static_cast<core::i32>(x);
            const core::i32 worldZ = out.originZ + static_cast<core::i32>(z);
            if (carvedSurfaceAt(params, site, out.adit, worldX, worldZ) < needed)
                continue;
            out.covered.at(x, z) = 1u;
            ++out.coveredColumns;
        }

    CaveSystemParams cave;
    cave.width = span;
    cave.depth = span;
    cave.seed = site.seed ^ 0x0CA7E9u;
    cave.layers = layers;
    cave.levelsPerLayer = perLayer;
    cave.topFill = warren.topFill;
    cave.deepFill = warren.deepFill;
    cave.automatonSteps = warren.automatonSteps;
    cave.minChamberSize = warren.minChamberSize;
    cave.shaftsPerPair = warren.shaftsPerPair;
    // No vertical entrances, and an empty surface so none are attempted. A shaft that
    // pierces the sky is a hole in a heightfield, which is the one thing a heightfield
    // cannot have — the way in here is horizontal, and it is this file's business.
    cave.entrances = 0u;

    // ── What kind of place this is ──────────────────────────────────────────
    //
    // The document's word when it gave one, and otherwise the place's own evidence.
    // Both queries are pure functions of coordinates, so this is decided identically
    // by every chunk that touches the warren and on every target.
    cave.settled = settledNearImpl(params, warren.villages, warren.seaLevel, site, warren.settlementReach);
    cave.wetness = sampleWorldMoisture(params, site.cellX, site.cellZ).toFloat();
    out.kind = warren.kind;
    if (out.kind == CaveKind::Auto)
    {
        CaveContext context;
        context.settled = cave.settled;
        context.wetness = cave.wetness;
        context.layerCount = layers;
        out.kind = chooseCaveKind(context);
    }
    // Layered is the CONTAINER, not a filling: a warren is always a stack, so the word
    // means "give the floors different characters" rather than naming a generator.
    cave.mixLayerKinds = out.kind == CaveKind::Layered;
    cave.layerKind = cave.mixLayerKinds ? CaveKind::Cellular : out.kind;

    CaveSystem system = generateCaveSystem(cave, Heightfield{}, nullptr);
    if (system.layerCount == 0u || system.layer[0].empty())
        return out;

    // ── Uncovered columns are not cave ──────────────────────────────────────
    //
    // Before the repair, never after: the repair guarantees a route through every
    // hollow cell, and a column plugged afterwards would break exactly the route it
    // had just promised.
    for (core::u32 l = 0u; l < system.layerCount; ++l)
        for (core::u32 z = 0u; z < span; ++z)
            for (core::u32 x = 0u; x < span; ++x)
                if (out.covered.at(x, z) == 0u)
                    system.layer[l].at(x, z) = DungeonCell::Wall;

    // ── The bore: the one part of the gallery that is not left to the automaton ──
    const core::i32 localMouthX = out.adit.mouthX - out.originX;
    const core::i32 localMouthZ = out.adit.mouthZ - out.originZ;
    core::u32 bored = 0u;
    for (core::u32 b = 0u; b <= warren.aditBore; ++b)
    {
        const core::i32 cx = localMouthX + out.adit.stepX * static_cast<core::i32>(b);
        const core::i32 cz = localMouthZ + out.adit.stepZ * static_cast<core::i32>(b);
        if (!boreColumn(system, out.covered, cx, cz))
            break; // the hill dipped: stop where the roof stops, do not punch through it
        ++bored;
        // Widened across the bore rather than along it, so a diagonal passage keeps
        // its width at the corners the same way the trench above does.
        for (core::u32 w = 1u; w <= warren.aditHalfWidth; ++w)
        {
            (void) boreColumn(system, out.covered, cx - static_cast<core::i32>(w) * out.adit.stepZ,
                              cz + static_cast<core::i32>(w) * out.adit.stepX);
            (void) boreColumn(system, out.covered, cx + static_cast<core::i32>(w) * out.adit.stepZ,
                              cz - static_cast<core::i32>(w) * out.adit.stepX);
        }
    }
    if (bored == 0u)
        return out; // the mouth cell is not covered; there is no cave to enter

    CaveShaft entrance;
    entrance.x = static_cast<core::u32>(localMouthX);
    entrance.z = static_cast<core::u32>(localMouthZ);
    entrance.upperLayer = 0u;
    entrance.lowerLayer = 0u;
    entrance.surface = true;
    system.shafts.push_back(entrance);
    system.entranceCount = 1u;

    // The counts are stale after the masking and the bore, and the repair reads them
    // to decide when it is done. Recounting is the caller's job precisely because the
    // caller is what edited the layers.
    recountCaveReachability(system);
    out.repairedCells = repairCaveReachability(system, cave.seed ^ 0x3E9A18u);

    const LevelQuality quality = evaluateCaveSystem(system);
    out.openCells = quality.walkableCells;
    out.reachableCells = quality.reachableCells;
    out.pathLength = quality.pathLength;
    out.navigable = quality.goalReachable;

    // ── The volume, plus the rock over it ───────────────────────────────────
    const VoxelVolume galleries = caveVolume(system, cave, warren.rockMaterial);
    if (galleries.empty())
        return out;

    out.volume.width = galleries.width;
    out.volume.depth = galleries.depth;
    out.volume.levels = levels;
    out.volume.cells.resize(static_cast<core::usize>(span) * span * levels, core::u8{0});
    for (core::u32 y = 0u; y < galleryLevels; ++y)
        for (core::u32 z = 0u; z < span; ++z)
            for (core::u32 x = 0u; x < span; ++x)
                out.volume.at(x, y, z) = galleries.at(x, y, z);
    // The cap, on covered columns only. Uncovered columns are terrain and the span
    // query ignores their voxels entirely; filling them would be rock nobody can
    // reach standing over ground the renderer draws.
    for (core::u32 y = galleryLevels; y < levels; ++y)
        for (core::u32 z = 0u; z < span; ++z)
            for (core::u32 x = 0u; x < span; ++x)
                if (out.covered.at(x, z) != 0u)
                    out.volume.at(x, y, z) = warren.rockMaterial;

    // ── The doorway ─────────────────────────────────────────────────────────
    //
    // A bored column that touches an uncovered one: that touching IS the opening,
    // because an uncovered column is open air with the terrain drawn across it. Found
    // rather than assumed to be the mouth cell, since the bore is several cells wide
    // and the hill does not meet it squarely.
    for (core::u32 b = 0u; b < bored && out.apertureCount < kMaxApertureCells; ++b)
    {
        const core::i32 cx = localMouthX + out.adit.stepX * static_cast<core::i32>(b);
        const core::i32 cz = localMouthZ + out.adit.stepZ * static_cast<core::i32>(b);
        for (core::i32 w = -static_cast<core::i32>(warren.aditHalfWidth);
             w <= static_cast<core::i32>(warren.aditHalfWidth); ++w)
        {
            const core::i32 ax = cx - w * out.adit.stepZ;
            const core::i32 az = cz + w * out.adit.stepX;
            if (!out.covered.contains(ax, az) ||
                out.covered.at(static_cast<core::u32>(ax), static_cast<core::u32>(az)) == 0u)
                continue;
            bool touchesAir = false;
            for (core::u32 n = 0u; n < 4u && !touchesAir; ++n)
            {
                const core::i32 nx = ax + kNeighbor4X[n];
                const core::i32 nz = az + kNeighbor4Z[n];
                touchesAir = !out.covered.contains(nx, nz) ||
                             out.covered.at(static_cast<core::u32>(nx), static_cast<core::u32>(nz)) == 0u;
            }
            if (!touchesAir || out.apertureCount >= kMaxApertureCells)
                continue;
            out.apertureX[out.apertureCount] = out.originX + ax;
            out.apertureZ[out.apertureCount] = out.originZ + az;
            ++out.apertureCount;
        }
    }

    // A cave has to be bigger than the pockets this generator FILLS IN. `openCells != 0`
    // was the bar, and it let a single hollow cell through as a cave: the debug warp
    // dutifully teleported to the nearest one and reported "open 1", which is a hole you
    // can stand in and nothing else. `minChamberSize` is already the number that says
    // what is too small to be a room, so a whole warren under it is too small to be a
    // cave — a threshold the params already carry rather than one chosen here.
    const core::u32 leastCave = warren.minChamberSize == 0u ? 1u : warren.minChamberSize;
    out.valid = out.openCells >= leastCave && out.reachableCells != 0u && out.apertureCount != 0u;
    return out;
}

bool findNearestCaveWarren(const ChunkParams &params, const LandmarkParams &mouths, const CaveWarrenParams &warren,
                           core::f32 seaLevel, core::f32 drop, core::i32 fromCellX, core::i32 fromCellZ,
                           core::u32 maxRings, core::u32 maxBuilds, CaveWarren &out)
{
    if (mouths.cellSpan == 0u)
        return false;

    const core::i32 span = static_cast<core::i32>(mouths.cellSpan);
    const core::i32 homeX = floorDivChunk(fromCellX, span);
    const core::i32 homeZ = floorDivChunk(fromCellZ, span);

    core::u32 builds = 0u;
    long long bestDistance = -1;
    bool found = false;
    // Rings are searched one PAST the first hit, because the far corner of ring r is
    // further off than the near edge of ring r + 1 — stopping at the first ring that
    // contains anything would hand back something that is not the nearest.
    core::u32 grace = 0u;

    for (core::u32 ring = 0u; ring <= maxRings; ++ring)
    {
        const core::i32 radius = static_cast<core::i32>(ring);
        for (core::i32 lz = homeZ - radius; lz <= homeZ + radius; ++lz)
            for (core::i32 lx = homeX - radius; lx <= homeX + radius; ++lx)
            {
                // The ring only, not the filled square: the interior was searched at a
                // smaller radius, and re-testing it makes the search quadratic in the
                // ring for nothing.
                if (lx != homeX - radius && lx != homeX + radius && lz != homeZ - radius && lz != homeZ + radius)
                    continue;
                if (builds >= maxBuilds)
                    return found;

                LandmarkSite site;
                if (!landmarkAt(params, mouths, LandmarkKind::CaveMouth, seaLevel, lx, lz, site))
                    continue;
                // Cheap first. Planning the adit is a handful of noise samples and
                // rejects the quarter of sites with no cover within reach; building the
                // warren to find that out costs about 1.4 ms.
                if (!planCaveAdit(params, site, warren, drop).found)
                    continue;

                CaveWarren candidate = buildCaveWarren(params, site, warren, drop);
                ++builds;
                if (!candidate.valid)
                    continue;

                const long long dx = static_cast<long long>(site.cellX) - fromCellX;
                const long long dz = static_cast<long long>(site.cellZ) - fromCellZ;
                const long long distance = dx * dx + dz * dz;
                if (found && distance >= bestDistance)
                    continue;
                bestDistance = distance;
                out = static_cast<CaveWarren &&>(candidate);
                found = true;
            }

        if (!found)
            continue;
        if (grace != 0u)
            return true;
        grace = 1u;
    }
    return found;
}

VerticalSpan caveWarrenSpanAt(const CaveWarren &warren, core::i32 worldX, core::i32 worldZ, math::Fixed32 y,
                              math::Fixed32 terrain)
{
    VerticalSpan span;
    span.floor = terrain;
    span.ceiling = openSky();
    span.enclosed = false;

    if (!warren.isCavernous(worldX, worldZ) || warren.levelHeightFixed.raw() <= 0)
        return span;

    const core::u32 localX = static_cast<core::u32>(worldX - warren.originX);
    const core::u32 localZ = static_cast<core::u32>(worldZ - warren.originZ);
    const core::i32 levels = static_cast<core::i32>(warren.volume.levels);
    const core::i32 step = warren.levelHeightFixed.raw();
    const core::i32 base = warren.baseYFixed.raw();
    const auto levelY = [base, step](core::i32 level) { return math::Fixed32{base + level * step}; };

    core::i32 level = floorDivRaw(y.raw() - base, step);
    // Above the volume the hill answers, and a covered column IS hill: solid from the
    // roof up to the surface, sky over that. Below it, bedrock — clamped rather than
    // refused, because a body under the deepest gallery has to be put somewhere and
    // the lowest floor is the only place it can be.
    if (level >= levels)
        return span;
    if (level < 0)
        level = 0;

    const auto solid = [&warren, localX, localZ](core::i32 l) {
        return warren.volume.at(localX, static_cast<core::u32>(l), localZ) != 0u;
    };

    if (solid(level))
    {
        // Embedded in rock. Scan UP, the same direction a heightfield pushes a body
        // that is under its ground: a body can then always climb out of a mistake,
        // where scanning down could drop it into a sealed pocket it cannot leave.
        core::i32 escape = level + 1;
        while (escape < levels && solid(escape))
            ++escape;
        if (escape >= levels)
            return span; // solid all the way to the roof: the hill above is the way out
        level = escape;
    }

    core::i32 below = level - 1;
    while (below >= 0 && !solid(below))
        --below;
    span.floor = levelY(below + 1);

    core::i32 above = level + 1;
    while (above < levels && !solid(above))
        ++above;
    span.ceiling = levelY(above);
    span.enclosed = true;
    return span;
}

core::u32 foldCaveWarren(const CaveWarren &warren)
{
    core::u32 hash = kFnv1aOffsetBasis;
    const auto mix = [&hash](core::u32 word) { hash = (hash ^ word) * kFnv1aPrime; };

    mix(static_cast<core::u32>(warren.originX));
    mix(static_cast<core::u32>(warren.originZ));
    mix(static_cast<core::u32>(warren.baseYFixed.raw()));
    mix(static_cast<core::u32>(warren.levelHeightFixed.raw()));
    mix(warren.layerCount);
    mix(warren.openCells);
    mix(warren.coveredColumns);
    mix(warren.reachableCells);
    mix(warren.pathLength);
    mix(warren.repairedCells);
    mix(warren.navigable ? 1u : 0u);
    mix(warren.valid ? 1u : 0u);
    mix(static_cast<core::u32>(warren.kind));
    mix(warren.apertureCount);
    for (core::u32 i = 0u; i < warren.apertureCount; ++i)
    {
        mix(static_cast<core::u32>(warren.apertureX[i]));
        mix(static_cast<core::u32>(warren.apertureZ[i]));
    }

    // The adit, because it is the only part BOTH this module and the chunk terrain act
    // on. A target that derived a different mouth would carve different ground and
    // still fold an identical cave, which is the one disagreement a cave signature
    // alone cannot see.
    mix(warren.adit.found ? 1u : 0u);
    mix(warren.adit.length);
    mix(static_cast<core::u32>(warren.adit.mouthX));
    mix(static_cast<core::u32>(warren.adit.mouthZ));
    mix(static_cast<core::u32>(warren.adit.stepX));
    mix(static_cast<core::u32>(warren.adit.stepZ));
    mix(static_cast<core::u32>(math::Fixed32::fromFloat(warren.adit.floorY).raw()));

    for (core::u32 i = 0u; i < warren.covered.cellCount(); ++i)
        mix(warren.covered.data()[i]);
    hash ^= foldVolume(warren.volume);
    hash *= kFnv1aPrime;
    return hash;
}

} // namespace lpl::procgen
