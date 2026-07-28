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
    WorldRecipe = 1u, ///< A procgen recipe: seed + passes (see RecipeV1).
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
    core::u32 reserved0;     ///< Must be 0.
    core::u32 reserved1;     ///< Must be 0.
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

/// Bits of ScatterV1::flags.
inline constexpr core::u32 kScatterFlagCollidable = 1u << 0;

/// Scatter rules a wire recipe carries; mirrors procgen::kMaxScatterRules.
inline constexpr core::u32 kWireScatterRules = 4u;

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
 * Four hundred-odd bytes still buys a whole world — the point of a recipe is not
 * that it is small in the absolute, it is that it does not grow with the world it
 * describes.
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

    // ── Underground ─────────────────────────────────────────────────────────
    core::u32 caveWidth;
    core::u32 caveDepth;
    core::u32 caveSeed;
    core::f32 caveFillProbability;
    core::u32 caveSteps;
    core::u32 caveBirthLimit;
    core::u32 caveSurvivalLimit;
    core::u32 caveMinRegionSize;

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

    // ── Playability gate ────────────────────────────────────────────────────
    core::u32 gateMinPathLength;
    core::u32 gateMinWalkableCells;
    core::u32 gateMaxDeadEndRatio;

    // ── Population ──────────────────────────────────────────────────────────
    ScatterV1 scatter[kWireScatterRules];
    core::u32 scatterCount;

    core::u32 flags; ///< kRecipeFlag* bits.
};
static_assert(sizeof(RecipeV1) == 532u, "GamePack recipe layout is wire format");

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

private:
    const core::u8 *_bytes{nullptr};
    core::u32 _size{0u};
};

} // namespace lpl::pack

#endif // LPL_PACK_GAMEPACK_HPP
