/**
 * @file Biome.hpp
 * @brief What kind of place each cell is.
 *
 * A world stops being a shape and starts being a place when its regions differ.
 * The classification is Whittaker's: temperature against moisture decides the
 * vegetation. Temperature has two sources, and both matter — latitude gives the
 * pole-to-equator gradient, and the **lapse rate** takes warmth away with
 * altitude. Without the second one, a tall mountain at the equator is classified
 * exactly like the swamp at its foot, and the elevation banding every real
 * mountain shows (forest, then conifer, then tundra, then bare rock) never
 * appears.
 *
 * The point of doing this on a grid, before any entity exists, is that the
 * biome map is then available to every later pass: scatter can place pines in
 * taiga and cacti in desert without either knowing how the terrain was made.
 *
 * @warning The height thresholds are absolute world units, so they are coupled
 *          to the terrain's range: a @c mountainHeight above the field's maximum
 *          simply never triggers, and half the biomes become unreachable without
 *          anything reporting it. The defaults here match @ref NoiseParams'
 *          defaults; a caller that changes the noise amplitude must either
 *          renormalise (see @ref normalizeHeights) or move these with it.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_PROCGEN_BIOME_HPP
#    define LPL_PROCGEN_BIOME_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/procgen/Climate.hpp>
#    include <lpl/procgen/Grid.hpp>
#    include <lpl/procgen/Heightfield.hpp>

namespace lpl::procgen {

/**
 * @enum BiomeId
 * @brief The biomes a cell can be classified as.
 *
 * Ordered roughly cold-to-hot within each moisture band, so a debug render that
 * maps the id to a colour ramp already reads sensibly.
 */
enum class BiomeId : core::u8 {
    Ocean = 0,  ///< Below sea level.
    Beach,      ///< Just above sea level.
    Snow,       ///< High and cold.
    Tundra,     ///< High, cold, dry.
    Taiga,      ///< Cold, wet: conifer forest.
    Rock,       ///< High and bare.
    Desert,     ///< Hot and dry.
    Savanna,    ///< Hot, moderately dry.
    Grassland,  ///< Temperate, moderate moisture.
    Forest,     ///< Temperate and wet.
    Rainforest, ///< Hot and very wet.
    Marsh,      ///< Low and saturated.
    Lake,       ///< Standing fresh water: a basin the drainage could not empty.
    Count       ///< Number of biomes; never a classification result.
};

/// Per-cell biome classification.
using BiomeMap = Grid<BiomeId>;

/**
 * @struct BiomeParams
 * @brief The thresholds that carve the Whittaker diagram into biomes.
 *
 * All heights are absolute world units, so a field must be normalised to a
 * known range before classification (see @ref normalizeHeights) or these
 * thresholds mean nothing.
 */
struct BiomeParams {
    core::f32 seaLevel{-4.0f};      ///< At or below: Ocean.
    core::f32 beachHeight{0.8f};    ///< Up to seaLevel + this: Beach.
    core::f32 mountainHeight{9.0f}; ///< Above: Rock or Snow rather than vegetation.
    core::f32 snowHeight{13.0f};    ///< Above: Snow.
    core::f32 snowlineWarmth{0.3f}; ///< High ground colder than this holds snow rather than bare rock.
};

/**
 * @struct BiomeProfile
 * @brief Where a biome lives in climate space, and which axes it cares about.
 *
 * The centre says what the biome's ideal climate is; the weights say how much
 * each axis matters to it. A weight of zero is the useful case: it lets a biome
 * declare an axis irrelevant instead of having to pick a plausible-looking value
 * for it, which is what an unweighted nearest-neighbour lookup would force. A
 * desert cares enormously about moisture and not at all about how eroded the
 * ground is, and the table can now say exactly that.
 *
 * Weights are also what shape the *hypercube* the report describes: a large
 * weight narrows the biome's reach along that axis, a small one stretches it.
 */
struct BiomeProfile {
    BiomeId id;           ///< Biome this profile selects.
    ClimateVector center; ///< Ideal climate.
    ClimateVector weight; ///< Per-axis importance; 0 means "indifferent".
};

/**
 * @brief The default profile table — the Whittaker diagram, restated.
 *
 * These are the same distinctions the old threshold cascade drew, expressed as
 * positions rather than as branches, which is what lets four more axes join
 * without touching a single test. Ocean, Beach and Lake are deliberately absent:
 * they are decided by the hydrology, which knows things no noise axis does.
 *
 * @param outCount Receives the number of profiles.
 * @return Pointer to a static table, never null.
 */
[[nodiscard]] const BiomeProfile *biomeProfiles(core::u32 &outCount) noexcept;

/**
 * @brief Classifies every cell by nearest profile in climate space.
 *
 * Three things are settled before the climate is consulted at all, and they are
 * settled by altitude and hydrology because that is where the truth about them
 * lives: what is under the sea, what is on its shore, and what is standing water
 * the drainage could not empty. No amount of temperature and rainfall can tell a
 * lake from a wet meadow — only the flow routing knows the basin had no outlet.
 *
 * Everything else is a weighted squared distance and an argmin.
 *
 * @param field   Terrain (absolute heights).
 * @param climate The six axes, each normalised to [0, 1].
 * @param params  Sea, shore and summit thresholds.
 * @param lakes   Optional standing-water mask (see @ref lakeMask).
 * @return The biome map (empty when the inputs are empty or mismatched).
 */
[[nodiscard]] BiomeMap classifyBiomes(const Heightfield &field, const ClimateField &climate, const BiomeParams &params,
                                      const Grid<core::u8> *lakes = nullptr);

/**
 * @brief Which profile a single climate vector is nearest to.
 *
 * Exposed because it is the honest way to answer "why is this cell a desert?" —
 * a debug view can show the runner-up and the margin, which a whole-map call
 * cannot.
 *
 * @param climate      The cell's climate.
 * @param outDistance  Receives the winning weighted squared distance.
 * @return The nearest biome.
 */
[[nodiscard]] BiomeId nearestBiomeProfile(const ClimateVector &climate, math::Fixed32 &outDistance) noexcept;

/**
 * @brief Human-readable biome name, for debugging and reports.
 * @param biome Biome to name.
 * @return A static string, never null.
 */
[[nodiscard]] const char *biomeName(BiomeId biome) noexcept;

/**
 * @brief The inverse of @ref biomeName, for documents that name a biome.
 *
 * A `.lplscene` says "forest", not 9: a scatter rule keyed by a raw enum value
 * silently changes meaning the day a biome is inserted into the enumeration,
 * and a document is meant to outlive the build that wrote it.
 *
 * @param name Lowercase biome name.
 * @return The biome, or @c BiomeId::Count when the name is not one.
 */
[[nodiscard]] BiomeId biomeIdByName(const char *name) noexcept;

/**
 * @brief Counts cells per biome.
 * @param map      Biome map.
 * @param outCounts Array of at least @c BiomeId::Count entries.
 */
void countBiomes(const BiomeMap &map, core::u32 *outCounts);

/**
 * @brief Is this biome under water?
 * @param biome Biome to test.
 */
[[nodiscard]] constexpr bool isWater(BiomeId biome) noexcept
{
    return biome == BiomeId::Ocean || biome == BiomeId::Lake;
}

/**
 * @brief Can vegetation and props be scattered here?
 *
 * Ocean and bare rock are excluded, which is what stops a scatter pass from
 * planting trees in the sea without every caller having to say so.
 *
 * @param biome Biome to test.
 */
[[nodiscard]] constexpr bool isHabitable(BiomeId biome) noexcept
{
    return biome != BiomeId::Ocean && biome != BiomeId::Lake && biome != BiomeId::Rock && biome != BiomeId::Snow;
}

/**
 * @brief FNV-1a fold of a biome map, for determinism checks.
 * @param map Map to fold.
 * @return The 32-bit signature.
 */
[[nodiscard]] core::u32 foldBiomeMap(const BiomeMap &map);

} // namespace lpl::procgen

#endif // LPL_PROCGEN_BIOME_HPP
