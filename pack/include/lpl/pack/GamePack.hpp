/**
 * @file GamePack.hpp
 * @brief The baked game package: what a constrained target actually loads.
 *
 * A game is authored and versioned as a `.lplscene` JSON document — diffable,
 * hand-editable, and what the editor and the AI write. That document is the
 * source of truth, but it is not what every target should have to read: a JSON
 * parser needs a string heap and a float parser, neither of which belongs in
 * ring 0.
 *
 * So a host-side tool BAKES the document into this format: a flat, aligned,
 * little-endian byte image whose sections are plain old data. Reading it is
 * bounds-checking plus a memcpy — no parser, no allocation, no libc. Same game,
 * two encodings, one schema.
 *
 * The wire structs below are deliberately DECOUPLED from the in-memory engine
 * types they feed (lpl::procgen::WorldRecipe and friends). An engine struct is
 * free to grow a field, reorder for cache, or hold a bool; a wire struct may do
 * none of those without a version bump. Every field is explicitly sized, there
 * is no bool (booleans are packed into a flags word, since bool padding is not
 * something to bet a cross-target byte image on), and each layout is pinned by
 * a static_assert.
 *
 * Determinism: both targets are little-endian x86 with IEEE-754 floats, so a
 * memcpy of these structs reproduces identical bits. The authoritative state a
 * pack describes is a RECIPE, not a world — a few hundred bytes that the client
 * and the server each expand into the same world, exactly as sharing a Minecraft
 * seed reproduces the same map. That is what makes client/server agreement a
 * property of the format rather than a hope.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_PACK_GAMEPACK_HPP
#    define LPL_PACK_GAMEPACK_HPP

#    include <lpl/core/Types.hpp>

namespace lpl::pack {

/// Bytes of the magic identifier at the head of every pack.
inline constexpr core::u32 kMagicSize = 8u;

/// Current wire format version. Bump on ANY layout change below.
inline constexpr core::u32 kFormatVersion = 1u;

/**
 * @enum SectionType
 * @brief What a section carries. Unknown types are skipped, not an error:
 *        a client may ignore server-only sections and vice versa, and an older
 *        reader must survive a newer pack that carries sections it never heard
 *        of. Only the sections a target actually needs have to be understood.
 */
enum class SectionType : core::u32 {
    Unknown = 0u,
    WorldRecipe = 1u,  ///< A procgen recipe: seed + passes (see RecipeV1).
    LivingRecipe = 2u, ///< What lives on it: food web, herd, stigmergy (see LivingV1).
    ViewProfile = 3u,  ///< What it LOOKS like: sky, water, palette (see ViewV1).
    /// 4 is reserved for a baked entity export; see docs/PLAN_caine.md phase 4.2.
    Ecc = 5u, ///< Transversal Reed-Solomon parity over the rest of the pack (see EccV1).
};

/**
 * @struct Header
 * @brief Fixed 32-byte prologue of every pack.
 */
struct Header {
    char magic[kMagicSize];  ///< "LPLPAK\0\0", never NUL-terminated as a string.
    core::u32 formatVersion; ///< @ref kFormatVersion when written.
    core::u32 totalSize;     ///< Size of the whole image, header included.
    core::u32 sectionCount;  ///< Entries in the section table that follows.
    core::u32 contentHash;   ///< FNV-1a over every byte after this header.

    /**
     * Offset and size of the parity section, or 0 when there is none.
     *
     * Duplicated here, out of the section table, and that redundancy is the whole
     * reason the repair works. The parity protects everything from the end of this
     * header onward — which INCLUDES the section table — so a burst landing on the
     * table destroys the only thing that could have said where the parity is. That was
     * measured, not imagined: with the locator in the table alone, 30 of 31 whole-row
     * bursts repaired and the one that wiped row zero reported "no parity section" on
     * a pack that had one.
     *
     * The header is outside the protected span for exactly this reason: it is small,
     * it is the bootstrap, and it is the one thing a repair cannot afford to have lost.
     */
    core::u32 eccOffset;
    core::u32 eccSize;
};
static_assert(sizeof(Header) == 32u, "GamePack header layout is wire format");

/**
 * @struct SectionEntry
 * @brief One row of the section table, immediately after the header.
 */
struct SectionEntry {
    core::u32 type;     ///< A @ref SectionType value.
    core::u32 offset;   ///< Byte offset from the start of the pack.
    core::u32 size;     ///< Byte length of the section payload.
    core::u32 reserved; ///< Must be 0.
};
static_assert(sizeof(SectionEntry) == 16u, "GamePack section entry is wire format");

/// Bits of RecipeV1::flags — which passes the recipe asks for.
inline constexpr core::u32 kRecipeFlagNormalizeTerrain = 1u << 0;
inline constexpr core::u32 kRecipeFlagErodeTerrain = 1u << 1;
inline constexpr core::u32 kRecipeFlagCarveRivers = 1u << 2;
inline constexpr core::u32 kRecipeFlagClassifyBiomes = 1u << 3;
inline constexpr core::u32 kRecipeFlagCarveCaves = 1u << 4;
inline constexpr core::u32 kRecipeFlagPlaceSettlement = 1u << 5;
inline constexpr core::u32 kRecipeFlagMaterializeGround = 1u << 6;
inline constexpr core::u32 kRecipeFlagCheckPlayability = 1u << 7;
inline constexpr core::u32 kRecipeFlagGateRequireGoal = 1u << 8;
inline constexpr core::u32 kRecipeFlagGateRequireConnected = 1u << 9;
inline constexpr core::u32 kRecipeFlagGrowRoads = 1u << 10;
inline constexpr core::u32 kRecipeFlagRoadArterials = 1u << 11;
inline constexpr core::u32 kRecipeFlagPartitionRegions = 1u << 12;
inline constexpr core::u32 kRecipeFlagRaiseBuildings = 1u << 13;
inline constexpr core::u32 kRecipeFlagBuildingsHollow = 1u << 14;

/// Longest roadside grammar a cartridge carries, NUL included. Must match the engine.
inline constexpr core::u32 kWireRoadsidePattern = 64u;

/// Bits of ScatterV1::flags.
inline constexpr core::u32 kScatterFlagCollidable = 1u << 0;

/**
 * @brief Scatter rules a wire recipe carries; mirrors procgen::kMaxScatterRules.
 *
 * Eight, because one rule per biome is the natural way to write vegetation — a taiga
 * is conifers, a savanna is scrub, a marsh is reeds — and four ran out at the fifth
 * kind of plant. A world that wanted six had to be built by hand-written
 * `WorldBuilder` calls, which is a world that cannot be saved, baked, replayed in
 * ring 0 or asked for by an intelligence.
 */
inline constexpr core::u32 kWireScatterRules = 8u;

/**
 * @struct ScatterV1
 * @brief Wire form of one prop-placement rule.
 */
struct ScatterV1 {
    core::u32 biome;            ///< procgen::BiomeId value.
    core::f32 density;          ///< Share of the biome's area to cover.
    core::f32 halfExtent;       ///< Prop AABB half-size.
    core::f32 maxSlope;         ///< Steepest ground it stands on.
    core::f32 minMoisture;      ///< Driest ground it tolerates.
    core::f32 maxMoisture;      ///< Wettest ground it tolerates.
    core::f32 moistureAffinity; ///< How much wetter ground packs it closer.
    core::u32 tag;              ///< Caller-defined kind.
    core::f32 treeLine;         ///< Share of the height range above which it thins.
    core::f32 altitudeFalloff;  ///< How sharply it thins past the tree line.
    core::u32 maxRiverDistance; ///< Cells from running water; 0 disables the test.
    core::f32 endemicShare;     ///< Share of regions this species inhabits.
    core::u32 flags;            ///< kScatterFlag* bits.
};
static_assert(sizeof(ScatterV1) == 52u, "GamePack scatter rule is wire format");

/**
 * @struct RecipeV1
 * @brief Wire form of a procedural world recipe: the whole WorldBuilder pipeline.
 *
 * Flattened on purpose: a nested mirror of the engine structs would couple this
 * layout to their internal grouping, and the whole point is that the engine types
 * can move while the bytes on disk cannot. Every field is copied by name in
 * RecipeCodec.hpp, so a rename or a reordering on either side is a compile error
 * rather than a silently reinterpreted world.
 *
 * It describes the whole pipeline rather than a subset of it, and that is not a
 * feature list — it is the reason a single generator exists. A wire format that
 * could only express a raw heightfield would force anything richer to be built by
 * a second path the cartridge cannot carry, which is how a project ends up with
 * two generators and a parity gate that exercises the one nothing else runs.
 *
 * A kilobyte still buys a whole world — the point of a recipe is not that it is
 * small in the absolute, it is that it does not grow with the world it describes.
 *
 * It carries every pass, including the ones that were briefly split into a
 * `Morphology` section. That split existed for exactly one reason — `sizeof(RecipeV1)`
 * was treated as frozen, so growing it would have invalidated cartridges already
 * baked — and the reason does not apply: there is no released version of this format
 * and no reader in the wild. Two homes for recipe fields, justified by a constraint
 * that is not real, is a worse thing to explain than a longer struct. The size
 * assertion below stays, but it now means "this layout is deliberate", not "this
 * layout may never change": it catches padding a compiler slipped in, not growth an
 * author intended.
 */
struct RecipeV1 {
    // ── World ───────────────────────────────────────────────────────────────
    core::u32 seed;
    core::u32 width;
    core::u32 depth;
    core::f32 cellSize;

    // ── Terrain noise ───────────────────────────────────────────────────────
    core::u32 noiseSeed;
    core::f32 noiseFrequency;
    core::f32 noiseAmplitude;
    core::u32 noiseOctaves;
    core::f32 noiseBaseHeight;
    core::f32 noiseLacunarity;
    core::f32 noisePersistence;
    core::f32 noiseWarpStrength;
    core::u32 noiseKind; ///< procgen::NoiseKind value.
    core::f32 heightLow;
    core::f32 heightHigh;
    core::f32 groundClearance; ///< Lowest the finished ground may sit; 0 leaves it where it fell.
    core::u32 terraceSteps;    ///< 0 leaves the field smooth.

    // ── Thermal erosion ─────────────────────────────────────────────────────
    core::u32 thermalIterations;
    core::f32 thermalTalus;
    core::f32 thermalCarryFraction;

    // ── Hydraulic erosion ───────────────────────────────────────────────────
    core::u32 hydraulicIterations;
    core::f32 hydraulicRainAmount;
    core::f32 hydraulicSolubility;
    core::f32 hydraulicEvaporation;
    core::f32 hydraulicSedimentCapacity;
    core::f32 hydraulicDeposition;
    core::f32 hydraulicMinSlope;

    // ── Rivers ──────────────────────────────────────────────────────────────
    core::f32 riverDensity;
    core::f32 riverCarveDepth;
    core::u32 riverSmoothing;

    // ── Climate ─────────────────────────────────────────────────────────────
    core::f32 climateRainfallWeight;
    core::f32 climateFlowWeight;
    core::f32 climateAltitudeWeight;
    core::f32 climateCoastWeight;
    core::f32 climateSeaLevel;
    core::f32 climateCoastReach;
    core::f32 climateRainShadow;
    core::u32 climateWindDirection;
    core::u32 climateSmoothing;
    core::u32 climateRainfallSeed;
    core::f32 climateRainfallBelts;
    core::u32 climateRainfallOctaves;

    // ── Biomes ──────────────────────────────────────────────────────────────
    core::f32 biomeSeaLevel;
    core::f32 biomeBeachHeight;
    core::f32 biomeMountainHeight;
    core::f32 biomeSnowHeight;
    core::f32 biomeSnowlineWarmth;

    // ── Climate axes ────────────────────────────────────────────────────────
    //
    // These seven words used to carry the Whittaker thresholds (dry, wet, marsh,
    // polar, temperate...). Those thresholds no longer exist: the classification
    // is a nearest-profile lookup, and a profile table is code, not content. The
    // words are reused rather than appended because the format was never
    // published — there is no reader in the wild to keep happy.
    core::f32 axisColdLatitude;
    core::f32 axisLapseRate;
    core::f32 axisCoastReach;
    core::u32 axisWeirdnessSeed;
    core::f32 axisWeirdnessBelts;
    core::u32 axisWeirdnessOctaves;
    core::f32 axisSurfaceDepth;

    // ── Provinces ───────────────────────────────────────────────────────────
    core::u32 provinceWidth;
    core::u32 provinceDepth;
    core::u32 provinceSeed;
    core::u32 provinceCellSize;
    core::f32 provinceJitter;
    core::f32 provinceWarpStrength;
    core::u32 provinceMetric; ///< procgen::DistanceMetric value.

    // ── Underground ─────────────────────────────────────────────────────────
    //
    // Four generators, and the recipe names which one runs. Carrying all four sets of
    // parameters rather than a union is what lets a document switch generator without
    // losing the settings of the one it is leaving.
    core::u32 caveKind; ///< procgen::CaveKind value.

    // Cellular.
    core::u32 caveWidth;
    core::u32 caveDepth;
    core::u32 caveSeed;
    core::f32 caveFillProbability;
    core::u32 caveSteps;
    core::u32 caveBirthLimit;
    core::u32 caveSurvivalLimit;
    core::u32 caveMinRegionSize;

    // BSP rooms.
    core::u32 roomsWidth;
    core::u32 roomsDepth;
    core::u32 roomsSeed;
    core::u32 roomsMaxDepth;
    core::u32 roomsMinLeafSize;
    core::u32 roomsPadding;
    core::u32 roomsCorridorWidth;

    // Diffusion-limited aggregation.
    core::u32 dlaWidth;
    core::u32 dlaDepth;
    core::u32 dlaSeed;
    core::u32 dlaParticles;
    core::u32 dlaMaxStepsPerParticle;
    core::u32 dlaSpawnMargin;
    core::u32 dlaThickness;

    // The layered system. Every knob, not a chosen five: a document that can name
    // only half a generator sends its author back to hand-written builder calls for
    // the other half, which is the thing this format exists to stop.
    core::u32 systemWidth;
    core::u32 systemDepth;
    core::u32 systemSeed;
    core::u32 systemLayers;
    core::u32 systemLevelsPerLayer;
    core::f32 systemTopFill;
    core::f32 systemDeepFill;
    core::u32 systemAutomatonSteps;
    core::u32 systemMinChamberSize;
    core::u32 systemShaftsPerPair;
    core::u32 systemEntrances;
    core::f32 systemEntranceMaxSlope;

    // ── Settlement ──────────────────────────────────────────────────────────
    core::u32 settlementSeed;
    core::u32 settlementDistrictSize;
    core::u32 settlementRoadWidth;
    core::u32 settlementPlazaRadius;
    core::u32 settlementMinPlot;
    core::u32 settlementMaxPlot;
    core::f32 settlementPlotDensity;
    core::f32 settlementMaxSlope;
    core::f32 settlementMinHeight;

    // ── Roads ───────────────────────────────────────────────────────────────
    core::u32 roadSeed;
    core::u32 roadIterations;
    core::u32 roadStepLength;
    core::f32 roadConform;
    core::f32 roadMaxSlope;
    core::f32 roadMinHeight;
    core::u32 roadGridDistricts;

    // ── The shape grammar ───────────────────────────────────────────────────
    core::u32 buildingSeed;
    core::u32 buildingMinFloors;
    core::u32 buildingMaxFloors;
    core::u32 buildingBaseHeight;
    core::u32 buildingFloorHeight;
    core::u32 buildingRoofHeight;
    core::u32 buildingInset;
    core::f32 buildingRoofTaper;
    core::u32 buildingBaseMaterial;
    core::u32 buildingWallMaterial;
    core::u32 buildingRoofMaterial;

    core::u32 roadsideLevels;                   ///< 0 leaves the verges bare.
    char roadsidePattern[kWireRoadsidePattern]; ///< NUL-terminated L-system.

    // ── Playability gate ────────────────────────────────────────────────────
    core::u32 gateMinPathLength;
    core::u32 gateMinWalkableCells;
    core::u32 gateMaxDeadEndRatio;

    // ── Population ──────────────────────────────────────────────────────────
    ScatterV1 scatter[kWireScatterRules];
    core::u32 scatterCount;

    core::u32 flags; ///< kRecipeFlag* bits.
};
static_assert(sizeof(RecipeV1) == 996u, "GamePack recipe layout is wire format");

/**
 * @struct LivingSpeciesV1
 * @brief Wire form of one authored population.
 *
 * Fixed32 travels as its RAW Q16.16 word, never as a float: the value IS the
 * bits, and a decimal round trip would be a different number on the other side
 * of the gate.
 */
struct LivingSpeciesV1 {
    core::u32 level;      ///< ecology::TrophicLevel value.
    core::i32 growth;     ///< Raw Q16.16.
    core::i32 mortality;  ///< Raw Q16.16.
    core::i32 predation;  ///< Raw Q16.16.
    core::i32 conversion; ///< Raw Q16.16.
    core::i32 capacity;   ///< Raw Q16.16.
    core::i32 refuge;     ///< Raw Q16.16.
    core::i32 initial;    ///< Raw Q16.16 head count at tick 0.
    core::u32 preyIndex;  ///< Index into the table; 0xFFFFFFFF for a producer.
};
static_assert(sizeof(LivingSpeciesV1) == 36u, "GamePack living species layout is wire format");

/**
 * @struct LivingV1
 * @brief Wire form of a living recipe: what lives on the world the recipe built.
 *
 * A second section rather than more fields on @ref RecipeV1, because sections
 * ARE this format's extension mechanism: a pack without this one is a world with
 * no declared life, which the reader reports as absent rather than as zeroes. A
 * grown RecipeV1 would instead have made every previously baked cartridge the
 * wrong size.
 *
 * The world was authorable down to the erosion iteration count while what lived
 * on it was compiled into the host — so a `.lplscene` could describe a valley and
 * had no way to say what grazed in it. This closes that.
 */
struct LivingV1 {
    // ── Run ─────────────────────────────────────────────────────────────────
    core::u32 seed;
    core::u32 ticks;
    core::i32 stepSeconds; ///< Raw Q16.16 seconds per step.

    // ── Field ───────────────────────────────────────────────────────────────
    core::u32 width;
    core::u32 depth;
    core::u32 channels;

    // ── Populations ─────────────────────────────────────────────────────────
    core::u32 rooms;
    core::u32 creatures;
    core::u32 ants;
    core::u32 boids;
    core::u32 genomes;
    core::u32 packMembers;
    core::u32 regrowthTicks;
    core::u32 headPerBody;

    // ── Stigmergy ───────────────────────────────────────────────────────────
    core::f32 evaporation;
    core::f32 diffusion;
    core::f32 maximum;
    core::f32 floorValue;

    // ── Foraging ────────────────────────────────────────────────────────────
    core::u32 explore16;
    core::i32 depositQuality; ///< Raw Q16.16.
    core::u32 trailChannel;

    // ── Flock ───────────────────────────────────────────────────────────────
    core::i32 separationRadius; ///< Raw Q16.16.
    core::i32 neighbourRadius;  ///< Raw Q16.16.
    core::f32 separationWeight;
    core::f32 alignmentWeight;
    core::f32 cohesionWeight;
    core::f32 maxSpeed;

    // ── Realisation budget ──────────────────────────────────────────────────
    core::u32 maxRealisedRooms;
    core::u32 changeWeight;
    core::u32 adjacentBonus;
    core::u32 predictedBonus;
    core::u32 unlikelyPenalty;

    // ── Heredity ────────────────────────────────────────────────────────────
    core::u32 mutationChance16;
    core::f32 mutationAmplitude;
    core::u32 collapseShare16;
    core::u32 meltdownChance16;
    core::f32 meltdownAmplitude;
    core::f32 anomalySigma;

    // ── Packs ───────────────────────────────────────────────────────────────
    core::u32 packMaxSize;
    core::u32 packMinSize;
    core::u32 dissolutionChance16;
    core::u32 adoptStrays; ///< 0 or 1.

    // ── The food web ────────────────────────────────────────────────────────
    LivingSpeciesV1 species[4];
    core::u32 speciesCount;
};
static_assert(sizeof(LivingV1) == 316u, "GamePack living layout is wire format");

/// Biome colours a wire view profile carries; covers procgen::BiomeId's range.
inline constexpr core::u32 kWireBiomeColours = 16u;

/// Bits of ViewV1::flags.
inline constexpr core::u32 kViewFlagOverridePalette = 1u << 0; ///< Use the table below.


/**
 * @struct ViewV1
 * @brief Wire form of a view profile: what a world LOOKS like.
 *
 * The third section, and the one whose boundary is the interesting part. It would
 * have been easy to put the whole presentation state here — per-pixel shading,
 * shadow budgets, resident chunk ceilings — and it would have been wrong, because
 * those describe a MACHINE. A cartridge that carried them would tell a phone to
 * render like a workstation, and a browser build to keep fifty-six chunks resident
 * on a heap that has room for nine.
 *
 * What is here is the other half, and it was homeless: the colour of the sky, the
 * hour of the day, where the sea is, how the water is tinted, what a forest looks
 * like. Those describe a PLACE. They were compiled into the host, so a `.lplscene`
 * could specify erosion iteration counts and had no way to say the world was at
 * dusk — every world the format could express came out the same blue.
 *
 * The split is the same one @c TerrainSurfaceParams already draws, promoted to the
 * document: sea level and fog density are content, whether the fog is computed per
 * pixel is a budget. engine::HostProfile keeps the budgets.
 */
struct ViewV1 {
    // ── Sky ─────────────────────────────────────────────────────────────────
    core::f32 zenithR;
    core::f32 zenithG;
    core::f32 zenithB;
    core::f32 horizonR;
    core::f32 horizonG;
    core::f32 horizonB;
    core::f32 duskR;
    core::f32 duskG;
    core::f32 duskB;
    core::f32 groundR;
    core::f32 groundG;
    core::f32 groundB;
    core::f32 sunSize;
    core::f32 mieStrength;
    core::f32 mieSharpness;
    core::f32 nightFloor;

    /// Time of day at load, in [0, 1). Not a sun VECTOR: the vector is derived,
    /// and storing both invites a pack whose two halves disagree.
    core::f32 dayFraction;

    // ── Surface ─────────────────────────────────────────────────────────────
    core::f32 seaLevel;
    core::f32 fogDensity;
    core::f32 ambient;
    core::f32 grainTiles;
    core::u32 shadowSteps;

    // ── Water ───────────────────────────────────────────────────────────────
    core::u32 waterShallow; ///< Packed 0x00RRGGBB, as everything else here.
    core::u32 waterDeep;
    core::f32 rippleScale;
    core::f32 rippleAmplitude;
    core::f32 glintPower;
    core::f32 depthScale;

    // ── Creatures ───────────────────────────────────────────────────────────
    core::u32 grazerTint;
    core::u32 hunterTint;
    core::f32 bodyScale;

    // ── Biome palette ───────────────────────────────────────────────────────
    //
    // Indexed by procgen::BiomeId. Only consulted when kViewFlagOverridePalette
    // is set, so a pack that likes the built-in colours costs sixty-four zero
    // bytes and says nothing, rather than accidentally painting the world black.
    core::u32 biomeColour[kWireBiomeColours];
    core::u32 biomeColourCount;

    core::u32 flags; ///< kViewFlag* bits.
};
static_assert(sizeof(ViewV1) == 196u, "GamePack view profile layout is wire format");

/**
 * @brief FNV-1a over a byte range — the pack's integrity check.
 * @param bytes Start of the range (may be null when @p size is 0).
 * @param size  Length in bytes.
 * @return The 32-bit digest.
 */
[[nodiscard]] core::u32 hashBytes(const core::u8 *bytes, core::u32 size) noexcept;

/**
 * @class View
 * @brief Non-owning, bounds-checked reader over a pack image.
 *
 * Owns nothing and allocates nothing: a target maps or copies the bytes however
 * it likes (a GRUB module, a network buffer, a file) and hands them here. Every
 * accessor re-validates against the image length, so a truncated or corrupt
 * cartridge yields a clean failure instead of reading past the end.
 */
class View {
public:
    View() = default;

    /**
     * @brief Validates magic, version, size and content hash.
     * @param bytes Start of the pack image.
     * @param size  Bytes available at @p bytes.
     * @return true if the image is a well-formed pack this build can read.
     */
    [[nodiscard]] bool open(const core::u8 *bytes, core::u32 size) noexcept;

    /// @return true once @ref open has succeeded.
    [[nodiscard]] bool isOpen() const noexcept { return _bytes != nullptr; }

    /// @return Number of sections in the table (0 when not open).
    [[nodiscard]] core::u32 sectionCount() const noexcept;

    /**
     * @brief Locates a section payload by type.
     * @param type     Section to find.
     * @param outBytes Set to the payload start on success.
     * @param outSize  Set to the payload length on success.
     * @return false when absent, or when its extent escapes the image.
     */
    [[nodiscard]] bool findSection(SectionType type, const core::u8 *&outBytes, core::u32 &outSize) const noexcept;

    /**
     * @brief Reads the world recipe section, if present.
     * @param outRecipe Filled on success.
     * @return false when there is no recipe section or it is the wrong size.
     */
    [[nodiscard]] bool readRecipe(RecipeV1 &outRecipe) const noexcept;

    /**
     * @brief Reads the living section, when the pack carries one.
     *
     * Absence is legitimate and is NOT an error: a cartridge may describe a world
     * with nothing living on it, and the host then keeps its own defaults. It is
     * a wrong-sized section that must be refused.
     *
     * @param outLiving Receives the living recipe.
     * @return true when the section is present and exactly the right size.
     */
    [[nodiscard]] bool readLiving(LivingV1 &outLiving) const noexcept;

    /**
     * @brief Reads the view profile, when the pack carries one.
     *
     * Absent is legitimate, exactly as for the living section: a pack that says
     * nothing about how the world looks gets the host's own defaults. Only a
     * wrong-sized section is a fault.
     */
    [[nodiscard]] bool readView(ViewV1 &outView) const noexcept;


private:
    const core::u8 *_bytes{nullptr};
    core::u32 _size{0u};
};

} // namespace lpl::pack

#endif // LPL_PACK_GAMEPACK_HPP
