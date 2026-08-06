/**
 * @file ChunkTerrain.hpp
 * @brief One chunk of an endless world: eroded relief, drainage, climate, biomes.
 *
 * The generation of a single chunk, as a function of its coordinates and nothing
 * else. That "nothing else" is the whole point: every value is sampled at ABSOLUTE
 * world coordinates, so two chunks generated at different times, on different
 * machines, in a different order, agree along their shared border by construction
 * rather than by a stitching pass.
 *
 * Three things in here are the answers to questions a chunked world asks and a
 * bounded one does not:
 *
 *  - erosion needs an APRON. Thermal relaxation propagates one cell per iteration,
 *    so a chunk generated with N iterations must be computed on a field N + 1 cells
 *    wider on every side and then cropped, or the chunked assembly stops being the
 *    unchunked computation. The host test measures that equality rather than
 *    asserting it.
 *  - drainage has no edges to flow off, so it is posed as a BOUNDED question: a cell
 *    carries water when enough terrain within a radius drains through it. The verdict
 *    depends only on the position, so neighbours agree.
 *  - temperature has to fall with latitude, and an endless world has no equator
 *    unless one is chosen. Here it is z = 0; without it, a world is one biome band
 *    repeated to infinity.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-28
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_PROCGEN_CHUNK_TERRAIN_HPP
#    define LPL_PROCGEN_CHUNK_TERRAIN_HPP

#    include <lpl/math/Random.hpp>
#    include <lpl/procgen/Biome.hpp>
#    include <lpl/procgen/CaveWarren.hpp>
#    include <lpl/procgen/Chunking.hpp>
#    include <lpl/procgen/Climate.hpp>
#    include <lpl/procgen/Heightfield.hpp>
#    include <lpl/procgen/Landmark.hpp>
#    include <lpl/procgen/ValueNoise.hpp>

namespace lpl::procgen {

/**
 * @struct ChunkTerrainRule
 * @brief Where the sea, the snow line and the rock line are, and how wet the world is.
 *
 * Altitude and water have the last word over the climate axes, exactly as they do in
 * the bounded classifier: no noise axis knows where the sea is.
 */
struct ChunkTerrainRule {
    core::u32 erosionIterations{6u}; ///< Thermal passes; also sets the apron.
    core::f32 seaLevel{-1.0f};
    core::f32 beachBand{0.8f};        ///< Height above the sea that is still shore.
    core::f32 rockLine{8.0f};         ///< Above this, bare rock.
    core::f32 snowLine{11.0f};        ///< Above this, snow.
    core::f32 latitudeScale{0.0016f}; ///< How fast it cools away from z = 0.
    core::f32 baseWarmth{0.72f};
    /**
     * @brief How fast it cools per metre ABOVE THE SEA.
     *
     * Above the sea, not above zero, and the difference is not pedantry: zero is an
     * arbitrary datum, so measuring from it makes the climate a function of where the
     * terrain happens to sit vertically. Measured — lowering a world six metres to give it
     * a coastline turned 97% forest into 0% forest and 71% grassland, because every cell
     * got 0.18 warmer without anything about the place having changed.
     *
     * It is also a rate per metre, so a world whose relief is scaled up needs it scaled
     * down by the same factor. See WalkScale.
     */
    core::f32 altitudeCooling{0.03f};
    core::u32 vegetationOneIn{3u}; ///< One wooded cell in N grows a plant.

    /**
     * @brief How far a river lowers its bed, and how deep the water stands in it.
     *
     * Zero depth means the mask is drawn and the ground is left alone, which is what this
     * path did: water lying on an untouched hillside, overflowing wherever the terrain
     * beside it dipped. The bounded path never had the problem — `carveRivers` takes its
     * field by non-const reference and lowers it.
     *
     * The fill is a FRACTION of the depth, and the two are separate for one reason: a
     * channel filled to its brim is a channel that was not worth cutting. Below one the
     * water sits inside its bed with banks standing over it, which is what a river looks
     * like; at one it is flush with the ground it was carved out of and reads as a wet
     * stripe painted on the hillside.
     */
    core::f32 riverDepth{0.0f};
    core::f32 riverFill{0.0f};

    // ── Landmarks ────────────────────────────────────────────────────────────
    //
    // Both default OFF. A landmark is a thing a world ASKS for: a chunk that carved
    // shelves and flattened terraces nobody requested would move every signature every
    // world already has.

    bool carveCaveMouths{false};
    LandmarkParams caveMouths{caveMouthDefaults()};

    /**
     * @brief Build the cave BEHIND each mouth this chunk owns.
     *
     * Separate from @ref carveCaveMouths because they cost three orders of magnitude
     * apart: carving a shelf is a disc of comparisons, and a warren is a cellular
     * automaton per floor plus a reachability flood — measured at 1.4 ms against
     * 0.28 ms for a whole chunk. A world that wants the shelves in its silhouette and
     * cannot afford the galleries says so here rather than getting both or neither.
     */
    bool buildWarrens{false};
    CaveWarrenParams warren{caveWarrenDefaults()};

    /**
     * @brief How far a cave mouth's shelf is cut below the ground at its centre.
     *
     * A heightfield cannot have an overhang, so a mouth is a SHELF cut into a hillside
     * with the hill still standing behind it — and the opening itself is a dark face the
     * renderer stands at the back of it. Carving a bowl instead would be a quarry.
     */
    core::f32 caveMouthDrop{2.6f};

    bool raiseVillages{false};
    LandmarkParams villages{settlementDefaults()};
};

/**
 * @struct ChunkTerrain
 * @brief What one chunk's terrain is.
 */
struct ChunkTerrain {
    Heightfield height{};
    BiomeMap biomes{};
    Grid<core::u8> rivers{};
    /**
     * @brief Which way the water runs, per river cell; @ref kNoFlow elsewhere.
     *
     * Baked here rather than derived at draw time because a coarse verdict costs nine
     * noise samples and a renderer wants one per cell per frame.
     */
    FlowDirection flow{};
    core::f32 lowest{0.0f}; ///< Lowest cell: whether this chunk has water at all.

    // ── Where the sea actually is, within the chunk ──────────────────────────
    //
    // `lowest` answers "is there water here at all", which is enough to skip a chunk
    // entirely. It is NOT enough to size the surface: a chunk with one drowned corner
    // was paying a per-pixel water pass over its whole extent, and the depth buffer
    // then threw almost all of it away. These bound the cells that are genuinely under
    // the sea, so the quad covers them and nothing else.
    core::u32 seaMinX{0u};
    core::u32 seaMaxX{0u};
    core::u32 seaMinZ{0u};
    core::u32 seaMaxZ{0u};
    bool hasSea{false};   ///< Whether any cell sits below the sea level.
    bool hasRiver{false}; ///< Whether any cell carries running water.

    /**
     * @brief Landmarks this chunk OWNS, i.e. the ones it is this chunk's job to draw.
     *
     * Only those whose centre falls inside it. Every chunk within reach carves a site's
     * ground — otherwise the terrain disagrees across a seam — but exactly one draws it,
     * or a village appears once per chunk that can see it.
     */
    lpl::pmr::vector<LandmarkSite> caveMouths{};
    lpl::pmr::vector<LandmarkBuilding> buildings{};
    /**
     * @brief The caves behind the mouths this chunk owns.
     *
     * Owned whole by one chunk rather than sliced across the chunks it overlaps, which
     * is the same decision @ref LandmarkBuilding makes and for a stronger reason: a
     * volume cut at a chunk border has a seam a body can fall through, and there is no
     * apron width that fixes it because a gallery is not a relaxation.
     */
    lpl::pmr::vector<CaveWarren> warrens{};
};

/**
 * @brief Generates one chunk's relief, drainage and biomes.
 *
 * @param emitPlant Called with (worldCellX, worldCellZ) for each plant grown. The
 *                  thinning is seeded from the chunk, so the same chunk grows the
 *                  same forest every time it streams in — which is what stops a
 *                  wood from rearranging itself when a walker turns around.
 */
template <typename EmitPlant>
[[nodiscard]] ChunkTerrain generateChunkTerrain(const ChunkParams &params, const EndlessRiverParams &riverParams,
                                                ChunkCoord coord, const ChunkTerrainRule &rule, EmitPlant &&emitPlant)
{
    const core::u32 size = params.size;
    ChunkTerrain out;
    out.height = generateErodedChunkTerrain(params, coord, rule.erosionIterations);
    out.biomes = BiomeMap{size, size, BiomeId::Grassland};
    out.rivers = markChunkRivers(params, riverParams, coord);

    // The bed, cut BEFORE `lowest` is measured — a carved channel is exactly how a chunk
    // comes to hold water it did not hold before, and a `lowest` taken beforehand would
    // report the chunk dry and let a renderer skip the water it just created.
    //
    // A pure function of (this cell's height, this cell's mask), so it needs no apron and
    // cannot disagree across a seam: both inputs are already chunk-independent. A depth that
    // looked at neighbours — a channel widening with its flow — would need the same apron
    // the erosion pass carries, and is the natural next step rather than this one.
    if (rule.riverDepth > 0.0f)
    {
        const math::Fixed32 depth = math::Fixed32::fromFloat(rule.riverDepth);
        for (core::u32 z = 0u; z < size; ++z)
            for (core::u32 x = 0u; x < size; ++x)
                if (out.rivers.at(x, z) != 0u)
                    out.height.at(x, z) = out.height.at(x, z) - depth;
    }

    // ── Landmarks: carved for every site in REACH, drawn only by the owner ───
    //
    // Caves before villages, so a village pad wins over a shelf: you do not build a
    // hamlet across a hole in the ground. Both before `lowest` and before the biome loop,
    // because a shelf is how a chunk comes to hold water it did not, and a terrace is how
    // a cell stops being a mountainside.
    if (rule.carveCaveMouths)
    {
        const core::i32 originX = coord.x * static_cast<core::i32>(size);
        const core::i32 originZ = coord.z * static_cast<core::i32>(size);
        forEachLandmarkNear(params, rule.caveMouths, LandmarkKind::CaveMouth, rule.seaLevel, coord,
                            [&](const LandmarkSite &site) {
                                // The shelf AND the trench that leads off it, from the one function the
                                // warren measures its own rock cover with. Two statements of where a
                                // mouth cuts the ground would be two answers to where the cave's roof
                                // starts, and the mouth is the only place those two halves meet.
                                const CaveAdit adit = planCaveAdit(params, site, rule.warren, rule.caveMouthDrop);
                                for (core::u32 z = 0u; z < size; ++z)
                                    for (core::u32 x = 0u; x < size; ++x)
                                    {
                                        core::f32 floor = 0.0f;
                                        if (!caveMouthFloorAt(site, adit, originX + static_cast<core::i32>(x),
                                                              originZ + static_cast<core::i32>(z), floor))
                                            continue;
                                        // LOWERED, never raised: the shelf is cut out of the hill, so
                                        // ground already below the floor is left alone. Setting it
                                        // would build a plinth out into the valley.
                                        const math::Fixed32 level = math::Fixed32::fromFloat(floor);
                                        if (out.height.at(x, z) > level)
                                            out.height.at(x, z) = level;
                                    }
                                if (!chunkOwnsLandmark(params, site, coord))
                                    return;
                                out.caveMouths.push_back(site);
                                // The warren itself is the EXPENSIVE half — a cellular automaton per
                                // floor plus a reachability flood — and only the owner needs it, for
                                // the same reason only the owner lays out a village. Every chunk in
                                // reach still carved the ground above, which is what keeps the seam
                                // exact.
                                if (rule.buildWarrens)
                                {
                                    CaveWarren warren = buildCaveWarren(params, site, rule.warren, rule.caveMouthDrop);
                                    if (warren.valid)
                                        out.warrens.push_back(static_cast<CaveWarren &&>(warren));
                                }
                            });
    }

    if (rule.raiseVillages)
    {
        const core::i32 originX = coord.x * static_cast<core::i32>(size);
        const core::i32 originZ = coord.z * static_cast<core::i32>(size);
        forEachLandmarkNear(
            params, rule.villages, LandmarkKind::Settlement, rule.seaLevel, coord, [&](const LandmarkSite &site) {
                const core::i32 radius = static_cast<core::i32>(site.radius);
                const core::f32 pad = site.height;
                // Graded over the outer cells rather than cut off at the edge: a square
                // terrace with a vertical rim is a mesa, and a village on a mesa reads as
                // an error. The blend is a function of the position alone, so two chunks
                // sharing the rim agree on it.
                const core::f32 feather = 3.0f;
                for (core::u32 z = 0u; z < size; ++z)
                    for (core::u32 x = 0u; x < size; ++x)
                    {
                        const core::i32 dx = originX + static_cast<core::i32>(x) - site.cellX;
                        const core::i32 dz = originZ + static_cast<core::i32>(z) - site.cellZ;
                        const core::i32 spanX = dx < 0 ? -dx : dx;
                        const core::i32 spanZ = dz < 0 ? -dz : dz;
                        const core::i32 furthest = spanX > spanZ ? spanX : spanZ;
                        if (furthest > radius)
                            continue;
                        const core::f32 inside = static_cast<core::f32>(radius - furthest);
                        core::f32 weight = inside / feather;
                        weight = weight > 1.0f ? 1.0f : weight;
                        const core::f32 natural = out.height.at(x, z).toFloat();
                        out.height.at(x, z) = math::Fixed32::fromFloat(natural + (pad - natural) * weight);
                    }

                // The layout is only needed by the chunk that will DRAW it, and it is the
                // expensive half — a whole bounded settlement solve. The flatten above
                // needs the site alone, which is what lets every chunk in reach agree
                // about the ground without any of them paying for the town.
                if (!chunkOwnsLandmark(params, site, coord))
                    return;
                const VillagePlan plan = planVillage(params, site);
                forEachVillageBuilding(plan,
                                       [&out](const LandmarkBuilding &building) { out.buildings.push_back(building); });
            });
    }

    out.flow = markChunkRiverFlow(params, riverParams, coord, out.rivers);

    out.lowest = out.height.empty() ? 0.0f : out.height.at(0u, 0u).toFloat();
    for (core::u32 z = 0u; z < size; ++z)
        for (core::u32 x = 0u; x < size; ++x)
        {
            const core::f32 h = out.height.at(x, z).toFloat();
            if (h < out.lowest)
                out.lowest = h;
            if (h < rule.seaLevel)
            {
                if (!out.hasSea)
                {
                    out.seaMinX = x;
                    out.seaMaxX = x;
                    out.seaMinZ = z;
                    out.seaMaxZ = z;
                    out.hasSea = true;
                }
                else
                {
                    if (x < out.seaMinX)
                        out.seaMinX = x;
                    if (x > out.seaMaxX)
                        out.seaMaxX = x;
                    if (z < out.seaMinZ)
                        out.seaMinZ = z;
                    if (z > out.seaMaxZ)
                        out.seaMaxZ = z;
                }
            }
            if (!out.rivers.empty() && out.rivers.at(x, z) != 0u)
                out.hasRiver = true;
        }

    const core::i32 originX = coord.x * static_cast<core::i32>(size);
    const core::i32 originZ = coord.z * static_cast<core::i32>(size);

    // Two extra noise layers, both at absolute coordinates: moisture at a little
    // under the terrain's frequency, continentalness at a quarter of it. Deriving
    // them from the terrain's own parameters is what keeps a wet world wet at every
    // scale instead of only at the one the layer was tuned at.
    // ⚠ Copied from the terrain layer for its FREQUENCY, and then stripped of everything
    // else it carries — because everything else it carries is about ground, not weather.
    //
    // Measured, and it is the reason this world had one biome in it: the copy inherited the
    // terrain's `baseHeight`, which endlessPlanFromRecipe sets to the recipe's base plus the
    // walk scale's lift. So `0.5 + sample * 0.5` came out at −0.54 to −0.28 for moisture and
    // −0.52 to −0.44 for continentalness, and clamp01 pinned BOTH to zero over the entire
    // world. Together with the three axes hard-coded below, five of the six were constant
    // and the classifier was deciding on temperature alone — which is why the whole map was
    // a single biome, and why lowering the world six metres flipped 97% forest to 71%
    // grassland: it was a knife-edge on one axis.
    //
    // The shaping terms go for the same reason. Plains flattening and a mountain gain are
    // instructions for a HEIGHTFIELD; applied to a moisture field they say nothing.
    NoiseParams climateLayer{};
    climateLayer.amplitude = 1.0f;
    climateLayer.octaves = 3u;

    NoiseParams moistureLayer = climateLayer;
    moistureLayer.seed = params.worldSeed ^ 0x3A15E7u;
    moistureLayer.frequency = params.noise.frequency * 0.6f;

    NoiseParams continentLayer = climateLayer;
    continentLayer.seed = params.worldSeed ^ 0xC0A57Du;
    continentLayer.frequency = params.noise.frequency * 0.25f;

    NoiseParams weirdnessLayer = climateLayer;
    weirdnessLayer.seed = params.worldSeed ^ 0x7E12D0u;
    weirdnessLayer.frequency = params.noise.frequency * 0.9f;

    math::Random thin{chunkSeed(params, coord)};
    const auto clamp01 = [](core::f32 v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); };

    for (core::u32 z = 0u; z < size; ++z)
        for (core::u32 x = 0u; x < size; ++x)
        {
            const core::i32 worldX = originX + static_cast<core::i32>(x);
            const core::i32 worldZ = originZ + static_cast<core::i32>(z);
            const core::f32 height = out.height.at(x, z).toFloat();

            ClimateVector climate{};
            const core::f32 latitude = static_cast<core::f32>(worldZ) * rule.latitudeScale;
            const core::f32 warmth = rule.baseWarmth - (latitude < 0.0f ? -latitude : latitude) -
                                     (height > rule.seaLevel ? (height - rule.seaLevel) * rule.altitudeCooling : 0.0f);
            climate[ClimateAxis::Temperature] = math::Fixed32::fromFloat(clamp01(warmth));

            const core::f32 wet = 0.5f + sampleNoiseAt(worldX, worldZ, moistureLayer).toFloat() * 0.5f;
            climate[ClimateAxis::Moisture] = math::Fixed32::fromFloat(clamp01(wet));

            const core::f32 land = 0.5f + sampleNoiseAt(worldX, worldZ, continentLayer).toFloat() * 0.5f;
            climate[ClimateAxis::Continentalness] = math::Fixed32::fromFloat(clamp01(land));
            // The other three axes, which were constants — so half the space the classifier
            // navigates was a single point and every cell of the world sat on it. Each one
            // now carries the quantity it is NAMED after, and each is already to hand:
            //
            //  - erosion is how worn the ground is here, which is its steepness;
            //  - depth is how far under the water it is, which is nothing on land;
            //  - weirdness is the axis whose whole job is to be arbitrary, so it is noise.
            const core::f32 east = out.height.at(x + 1u < size ? x + 1u : x, z).toFloat();
            const core::f32 south = out.height.at(x, z + 1u < size ? z + 1u : z).toFloat();
            const core::f32 slope =
                ((east > height ? east - height : height - east) + (south > height ? south - height : height - south)) *
                0.5f;
            // Half is flat and one is a cliff, on the scale the relief actually has: a
            // fixed metre threshold would read every world of a different amplitude wrong.
            climate[ClimateAxis::Erosion] =
                math::Fixed32::fromFloat(clamp01(0.5f + slope / (params.noise.amplitude * 0.25f)));
            climate[ClimateAxis::Depth] =
                math::Fixed32::fromFloat(height < rule.seaLevel ? clamp01((rule.seaLevel - height) / 16.0f) : 0.0f);
            climate[ClimateAxis::Weirdness] = math::Fixed32::fromFloat(
                clamp01(0.5f + sampleNoiseAt(worldX, worldZ, weirdnessLayer).toFloat() * 0.5f));

            math::Fixed32 distance{};
            BiomeId biome = nearestBiomeProfile(climate, distance);

            if (height < rule.seaLevel)
                biome = BiomeId::Ocean;
            else if (height < rule.seaLevel + rule.beachBand)
                biome = BiomeId::Beach;
            else if (height > rule.snowLine)
                biome = BiomeId::Snow;
            else if (height > rule.rockLine)
                biome = BiomeId::Rock;
            if (!out.rivers.empty() && out.rivers.at(x, z) != 0u && height >= rule.seaLevel)
                biome = BiomeId::Lake; // drawn as running water
            out.biomes.at(x, z) = biome;

            const bool wooded = biome == BiomeId::Forest || biome == BiomeId::Taiga || biome == BiomeId::Rainforest;
            if (wooded && (rule.vegetationOneIn <= 1u || thin.below(rule.vegetationOneIn) == 0u))
                emitPlant(worldX, worldZ);
        }

    return out;
}

} // namespace lpl::procgen

#endif // LPL_PROCGEN_CHUNK_TERRAIN_HPP
