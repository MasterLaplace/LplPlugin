/**
 * @file RecipeCodec.hpp
 * @brief Translates between the wire recipe and the engine recipe.
 *
 * The one place that knows both layouts, kept apart from GamePack.hpp so the
 * container format stays usable without pulling procgen in. Every field is
 * copied by name: a reordering on either side is then a compile error rather
 * than a silently reinterpreted world.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_PACK_RECIPECODEC_HPP
#    define LPL_PACK_RECIPECODEC_HPP

#    include <lpl/ecology/LivingRecipe.hpp>
#    include <lpl/pack/GamePack.hpp>
#    include <lpl/procgen/WorldRecipe.hpp>

#    include <bit>
#    include <type_traits>

static_assert(lpl::pack::kWireScatterRules == lpl::procgen::kMaxScatterRules,
              "wire and engine must agree on how many scatter rules a recipe carries");

static_assert(lpl::pack::kWireRoadsidePattern == lpl::procgen::kMaxRoadsidePattern,
              "wire and engine must agree on how long a roadside grammar may be");

static_assert(sizeof(lpl::pack::LivingV1::species) / sizeof(lpl::pack::LivingSpeciesV1) ==
                  lpl::ecology::kMaxLivingSpecies,
              "wire and engine must agree on how many species a living recipe carries");

namespace lpl::pack {

/**
 * @struct WireRefusal
 * @brief Why a decoder refused a section: the field it could not read, and the value it held.
 */
struct WireRefusal {
    const char *field{nullptr}; /**< The field, spelled as its section struct spells it, or the expression of
                                     several fields a generator computes; null when none was refused. */
    core::u32 value{0u};        /**< The value the field held: a real's IEEE-754 bits, a Fixed32's raw word,
                                     an expression's result saturated to 32 bits. */
};

namespace detail {

/**
 * @name The values an enum of the wire may hold
 * @brief Whether a value is one this build knows, for each enum a section carries.
 *
 * A switch with no default, on purpose: an enumerator added to one of these enums and not
 * here is a -Wswitch warning, which the host build turns into an error. A bound written as the
 * enum's last value cannot say that, and went stale exactly that way: `CaveKind::Auto` was
 * added after the bound was spelled `Layered`, and every `auto` in a document decoded as
 * `cellular`.
 * @{
 */
[[nodiscard]] constexpr bool isKnown(procgen::NoiseKind kind) noexcept;
[[nodiscard]] constexpr bool isKnown(procgen::DistanceMetric metric) noexcept;
[[nodiscard]] constexpr bool isKnown(procgen::CaveKind kind) noexcept;
[[nodiscard]] constexpr bool isKnown(procgen::BiomeId biome) noexcept;
[[nodiscard]] constexpr bool isKnown(ecology::TrophicLevel level) noexcept;
/** @} */

/**
 * @brief Reads an enum from the value a section holds: the one conversion every enum of the
 *        wire goes through.
 *
 * Refused, never clamped: a cartridge naming a generator this build does not know describes
 * another world, and running a neighbouring one in its place would say nothing.
 *
 * @param field      The field, spelled as its section struct spells it; named by the refusal.
 * @param value      The value the field holds.
 * @param out        Receives the enum; untouched when refused.
 * @param outRefusal Receives @p field and @p value when refused; untouched otherwise.
 * @return false when @p value is not a value of @p Enum this build knows, a value that only
 *         fits once truncated to the enum's underlying type included.
 */
template <typename Enum>
requires std::is_scoped_enum_v<Enum>
[[nodiscard]] constexpr bool readEnum(const char *field, core::u32 value, Enum &out, WireRefusal &outRefusal) noexcept;

/**
 * @name What the generators can hold
 * @brief The bounds every decoded section keeps, so that no value a cartridge holds overflows a
 *        size, loops without end or reaches undefined behaviour in the generator it drives.
 *
 * A cartridge is input, not a promise: its content hash is a checksum, not a signature, and the
 * client runs its recipe in ring 0. The work bounds sit an order of magnitude above every recipe
 * the repository ships.
 * @{
 */

/** Longest side of any grid a recipe sizes. */
inline constexpr core::u32 kMaxGridSide = 4096u;

/** Most voxel levels a volume stacks: a building, the verges, the cave system. */
inline constexpr core::u32 kMaxVolumeLevels = 255u;

static_assert(static_cast<core::u64>(kMaxGridSide) * kMaxGridSide * kMaxVolumeLevels < (core::u64{1} << 32u),
              "a volume's cell count is computed in 32 bits on i686, and each of its indices on every target");

static_assert(static_cast<core::u64>(kMaxGridSide) < (core::u64{1} << 15u),
              "a grid coordinate becomes a Fixed32, which holds integers below 32768");

/** Largest magnitude of a real: every one becomes a Fixed32, which ends at 32767. */
inline constexpr core::f32 kMaxRealMagnitude = 32767.0f;

/** Most steps a diffusion-limited aggregation walks in all: its particles times their steps. */
inline constexpr core::u64 kMaxAggregationSteps = core::u64{1} << 26u;

/**
 * @struct CountBound
 * @brief The largest value an unsigned field of @p Section may hold.
 */
template <typename Section> struct CountBound {
    const char *field;          /**< The field, spelled as @p Section spells it. */
    core::u32 Section::*member; /**< The field. */
    core::u32 most;             /**< Largest legal value. */
};

/**
 * @struct RealBound
 * @brief The range a real field of @p Section may hold; NaN and the infinities never do.
 */
template <typename Section> struct RealBound {
    const char *field;          /**< The field, spelled as @p Section spells it. */
    core::f32 Section::*member; /**< The field. */
    core::f32 least;            /**< Smallest legal value. */
    core::f32 most;             /**< Largest legal value. */
};

/**
 * @struct RawBound
 * @brief The range of the raw Q16.16 word a Fixed32 field of @p Section may hold.
 */
template <typename Section> struct RawBound {
    const char *field;          /**< The field, spelled as @p Section spells it. */
    core::i32 Section::*member; /**< The field. */
    core::i32 least;            /**< Smallest legal raw word. */
    core::i32 most;             /**< Largest legal raw word. */
};

/**
 * @brief The raw word of the Fixed32 @p value, as a bound spells it.
 */
[[nodiscard]] constexpr core::i32 rawOf(core::i32 value) noexcept { return value * 65536; }

/**
 * @brief Whether the IEEE-754 single @p bits encode a finite number; read from the bits, so no
 *        compiler flag can assume it away.
 */
[[nodiscard]] constexpr bool isFinite(core::u32 bits) noexcept { return (bits & 0x7F800000u) != 0x7F800000u; }

/**
 * @brief @p value, or the largest 32-bit word when it does not fit in one: what a refusal reports.
 */
[[nodiscard]] constexpr core::u32 saturated(core::u64 value) noexcept
{
    return value > 0xFFFFFFFFu ? 0xFFFFFFFFu : static_cast<core::u32>(value);
}

/**
 * @brief Checks every count of @p section against @p bounds.
 * @param section    The section, as decoded.
 * @param bounds     Its count bounds.
 * @param outRefusal Receives the first field past its bound and the value it held.
 * @return false when a field holds more than its bound.
 */
template <typename Section, core::usize Count>
[[nodiscard]] constexpr bool withinCounts(const Section &section, const CountBound<Section> (&bounds)[Count],
                                          WireRefusal &outRefusal) noexcept;

/**
 * @brief Checks every real of @p section against @p bounds.
 * @param section    The section, as decoded.
 * @param bounds     Its real bounds.
 * @param outRefusal Receives the first field out of range and the bits it held.
 * @return false when a field is not finite or lies outside its range.
 */
template <typename Section, core::usize Count>
[[nodiscard]] constexpr bool withinReals(const Section &section, const RealBound<Section> (&bounds)[Count],
                                         WireRefusal &outRefusal) noexcept;

/**
 * @brief Checks every Fixed32 of @p section against @p bounds.
 * @param section    The section, as decoded.
 * @param bounds     Its raw bounds.
 * @param outRefusal Receives the first field out of range and the raw word it held.
 * @return false when a field lies outside its range.
 */
template <typename Section, core::usize Count>
[[nodiscard]] constexpr bool withinRaws(const Section &section, const RawBound<Section> (&bounds)[Count],
                                        WireRefusal &outRefusal) noexcept;

static_assert(sizeof(RecipeV1) == 996u, "a field added to RecipeV1 gets its bound in kRecipeCounts or kRecipeReals, "
                                        "or is a seed, an enum or a flag word");

/** @brief The counts of a world recipe: sizes, iterations, radii and levels. */
inline constexpr CountBound<RecipeV1> kRecipeCounts[] = {
    {"width",                  &RecipeV1::width,                  kMaxGridSide    },
    {"depth",                  &RecipeV1::depth,                  kMaxGridSide    },
    {"noiseOctaves",           &RecipeV1::noiseOctaves,           16u             },
    {"terraceSteps",           &RecipeV1::terraceSteps,           256u            },
    {"thermalIterations",      &RecipeV1::thermalIterations,      256u            },
    {"hydraulicIterations",    &RecipeV1::hydraulicIterations,    256u            },
    {"riverSmoothing",         &RecipeV1::riverSmoothing,         64u             },
    {"climateSmoothing",       &RecipeV1::climateSmoothing,       64u             },
    {"provinceCellSize",       &RecipeV1::provinceCellSize,       kMaxGridSide    },
    {"caveWidth",              &RecipeV1::caveWidth,              kMaxGridSide    },
    {"caveDepth",              &RecipeV1::caveDepth,              kMaxGridSide    },
    {"caveSteps",              &RecipeV1::caveSteps,              64u             },
    {"roomsWidth",             &RecipeV1::roomsWidth,             kMaxGridSide    },
    {"roomsDepth",             &RecipeV1::roomsDepth,             kMaxGridSide    },
    {"roomsMaxDepth",          &RecipeV1::roomsMaxDepth,          16u             },
    {"roomsMinLeafSize",       &RecipeV1::roomsMinLeafSize,       kMaxGridSide    },
    {"roomsPadding",           &RecipeV1::roomsPadding,           kMaxGridSide    },
    {"roomsCorridorWidth",     &RecipeV1::roomsCorridorWidth,     64u             },
    {"dlaWidth",               &RecipeV1::dlaWidth,               kMaxGridSide    },
    {"dlaDepth",               &RecipeV1::dlaDepth,               kMaxGridSide    },
    {"dlaParticles",           &RecipeV1::dlaParticles,           1u << 16u       },
    {"dlaMaxStepsPerParticle", &RecipeV1::dlaMaxStepsPerParticle, 1u << 16u       },
    {"dlaThickness",           &RecipeV1::dlaThickness,           64u             },
    {"systemLevelsPerLayer",   &RecipeV1::systemLevelsPerLayer,   kMaxVolumeLevels},
    {"systemAutomatonSteps",   &RecipeV1::systemAutomatonSteps,   64u             },
    {"systemShaftsPerPair",    &RecipeV1::systemShaftsPerPair,    64u             },
    {"systemEntrances",        &RecipeV1::systemEntrances,        64u             },
    {"settlementDistrictSize", &RecipeV1::settlementDistrictSize, kMaxGridSide    },
    {"settlementRoadWidth",    &RecipeV1::settlementRoadWidth,    64u             },
    {"settlementPlazaRadius",  &RecipeV1::settlementPlazaRadius,  256u            },
    {"settlementMinPlot",      &RecipeV1::settlementMinPlot,      kMaxGridSide    },
    {"settlementMaxPlot",      &RecipeV1::settlementMaxPlot,      kMaxGridSide    },
    {"buildingMinFloors",      &RecipeV1::buildingMinFloors,      kMaxVolumeLevels},
    {"buildingMaxFloors",      &RecipeV1::buildingMaxFloors,      kMaxVolumeLevels},
    {"buildingBaseHeight",     &RecipeV1::buildingBaseHeight,     kMaxVolumeLevels},
    {"buildingFloorHeight",    &RecipeV1::buildingFloorHeight,    kMaxVolumeLevels},
    {"buildingRoofHeight",     &RecipeV1::buildingRoofHeight,     kMaxVolumeLevels},
    {"roadsideLevels",         &RecipeV1::roadsideLevels,         kMaxVolumeLevels},
};

/** @brief The reals of a world recipe: each one becomes a Fixed32. */
inline constexpr RealBound<RecipeV1> kRecipeReals[] = {
    {"cellSize",                  &RecipeV1::cellSize,                  -kMaxRealMagnitude, kMaxRealMagnitude},
    {"noiseFrequency",            &RecipeV1::noiseFrequency,            -kMaxRealMagnitude, kMaxRealMagnitude},
    {"noiseAmplitude",            &RecipeV1::noiseAmplitude,            -kMaxRealMagnitude, kMaxRealMagnitude},
    {"noiseBaseHeight",           &RecipeV1::noiseBaseHeight,           -kMaxRealMagnitude, kMaxRealMagnitude},
    {"noiseLacunarity",           &RecipeV1::noiseLacunarity,           -kMaxRealMagnitude, kMaxRealMagnitude},
    {"noisePersistence",          &RecipeV1::noisePersistence,          -kMaxRealMagnitude, kMaxRealMagnitude},
    {"noiseWarpStrength",         &RecipeV1::noiseWarpStrength,         -kMaxRealMagnitude, kMaxRealMagnitude},
    {"heightLow",                 &RecipeV1::heightLow,                 -kMaxRealMagnitude, kMaxRealMagnitude},
    {"heightHigh",                &RecipeV1::heightHigh,                -kMaxRealMagnitude, kMaxRealMagnitude},
    {"groundClearance",           &RecipeV1::groundClearance,           -kMaxRealMagnitude, kMaxRealMagnitude},
    {"thermalTalus",              &RecipeV1::thermalTalus,              -kMaxRealMagnitude, kMaxRealMagnitude},
    {"thermalCarryFraction",      &RecipeV1::thermalCarryFraction,      -kMaxRealMagnitude, kMaxRealMagnitude},
    {"hydraulicRainAmount",       &RecipeV1::hydraulicRainAmount,       -kMaxRealMagnitude, kMaxRealMagnitude},
    {"hydraulicSolubility",       &RecipeV1::hydraulicSolubility,       -kMaxRealMagnitude, kMaxRealMagnitude},
    {"hydraulicEvaporation",      &RecipeV1::hydraulicEvaporation,      -kMaxRealMagnitude, kMaxRealMagnitude},
    {"hydraulicSedimentCapacity", &RecipeV1::hydraulicSedimentCapacity, -kMaxRealMagnitude, kMaxRealMagnitude},
    {"hydraulicDeposition",       &RecipeV1::hydraulicDeposition,       -kMaxRealMagnitude, kMaxRealMagnitude},
    {"hydraulicMinSlope",         &RecipeV1::hydraulicMinSlope,         -kMaxRealMagnitude, kMaxRealMagnitude},
    {"riverDensity",              &RecipeV1::riverDensity,              -kMaxRealMagnitude, kMaxRealMagnitude},
    {"riverCarveDepth",           &RecipeV1::riverCarveDepth,           -kMaxRealMagnitude, kMaxRealMagnitude},
    {"climateRainfallWeight",     &RecipeV1::climateRainfallWeight,     -kMaxRealMagnitude, kMaxRealMagnitude},
    {"climateFlowWeight",         &RecipeV1::climateFlowWeight,         -kMaxRealMagnitude, kMaxRealMagnitude},
    {"climateAltitudeWeight",     &RecipeV1::climateAltitudeWeight,     -kMaxRealMagnitude, kMaxRealMagnitude},
    {"climateCoastWeight",        &RecipeV1::climateCoastWeight,        -kMaxRealMagnitude, kMaxRealMagnitude},
    {"climateSeaLevel",           &RecipeV1::climateSeaLevel,           -kMaxRealMagnitude, kMaxRealMagnitude},
    {"climateCoastReach",         &RecipeV1::climateCoastReach,         -kMaxRealMagnitude, kMaxRealMagnitude},
    {"climateRainShadow",         &RecipeV1::climateRainShadow,         -kMaxRealMagnitude, kMaxRealMagnitude},
    {"climateRainfallBelts",      &RecipeV1::climateRainfallBelts,      -kMaxRealMagnitude, kMaxRealMagnitude},
    {"biomeSeaLevel",             &RecipeV1::biomeSeaLevel,             -kMaxRealMagnitude, kMaxRealMagnitude},
    {"biomeBeachHeight",          &RecipeV1::biomeBeachHeight,          -kMaxRealMagnitude, kMaxRealMagnitude},
    {"biomeMountainHeight",       &RecipeV1::biomeMountainHeight,       -kMaxRealMagnitude, kMaxRealMagnitude},
    {"biomeSnowHeight",           &RecipeV1::biomeSnowHeight,           -kMaxRealMagnitude, kMaxRealMagnitude},
    {"biomeSnowlineWarmth",       &RecipeV1::biomeSnowlineWarmth,       -kMaxRealMagnitude, kMaxRealMagnitude},
    {"axisColdLatitude",          &RecipeV1::axisColdLatitude,          -kMaxRealMagnitude, kMaxRealMagnitude},
    {"axisLapseRate",             &RecipeV1::axisLapseRate,             -kMaxRealMagnitude, kMaxRealMagnitude},
    {"axisCoastReach",            &RecipeV1::axisCoastReach,            -kMaxRealMagnitude, kMaxRealMagnitude},
    {"axisWeirdnessBelts",        &RecipeV1::axisWeirdnessBelts,        -kMaxRealMagnitude, kMaxRealMagnitude},
    {"axisSurfaceDepth",          &RecipeV1::axisSurfaceDepth,          -kMaxRealMagnitude, kMaxRealMagnitude},
    {"provinceJitter",            &RecipeV1::provinceJitter,            -kMaxRealMagnitude, kMaxRealMagnitude},
    {"provinceWarpStrength",      &RecipeV1::provinceWarpStrength,      -64.0f,             64.0f            },
    {"caveFillProbability",       &RecipeV1::caveFillProbability,       -kMaxRealMagnitude, kMaxRealMagnitude},
    {"systemTopFill",             &RecipeV1::systemTopFill,             -kMaxRealMagnitude, kMaxRealMagnitude},
    {"systemDeepFill",            &RecipeV1::systemDeepFill,            -kMaxRealMagnitude, kMaxRealMagnitude},
    {"systemEntranceMaxSlope",    &RecipeV1::systemEntranceMaxSlope,    -kMaxRealMagnitude, kMaxRealMagnitude},
    {"settlementPlotDensity",     &RecipeV1::settlementPlotDensity,     -kMaxRealMagnitude, kMaxRealMagnitude},
    {"settlementMaxSlope",        &RecipeV1::settlementMaxSlope,        -kMaxRealMagnitude, kMaxRealMagnitude},
    {"settlementMinHeight",       &RecipeV1::settlementMinHeight,       -kMaxRealMagnitude, kMaxRealMagnitude},
    {"roadConform",               &RecipeV1::roadConform,               -kMaxRealMagnitude, kMaxRealMagnitude},
    {"roadMaxSlope",              &RecipeV1::roadMaxSlope,              -kMaxRealMagnitude, kMaxRealMagnitude},
    {"roadMinHeight",             &RecipeV1::roadMinHeight,             -kMaxRealMagnitude, kMaxRealMagnitude},
    {"buildingRoofTaper",         &RecipeV1::buildingRoofTaper,         -kMaxRealMagnitude, kMaxRealMagnitude},
};

static_assert(sizeof(ScatterV1) == 52u, "a field added to ScatterV1 gets its bound in kScatterReals, or is a tag "
                                        "or a flag word");

/**
 * @brief The reals of a scatter rule.
 *
 * Its shares stay in [0, 1], and the factors that stretch its spacing stay small: a rule spaces
 * its props by a radius the stretch multiplies, and scans that radius around every candidate.
 */
inline constexpr RealBound<ScatterV1> kScatterReals[] = {
    {"scatter.density",          &ScatterV1::density,          0.0f,               1.0f             },
    {"scatter.halfExtent",       &ScatterV1::halfExtent,       -kMaxRealMagnitude, kMaxRealMagnitude},
    {"scatter.maxSlope",         &ScatterV1::maxSlope,         -kMaxRealMagnitude, kMaxRealMagnitude},
    {"scatter.minMoisture",      &ScatterV1::minMoisture,      -kMaxRealMagnitude, kMaxRealMagnitude},
    {"scatter.maxMoisture",      &ScatterV1::maxMoisture,      -kMaxRealMagnitude, kMaxRealMagnitude},
    {"scatter.moistureAffinity", &ScatterV1::moistureAffinity, 0.0f,               1.0f             },
    {"scatter.treeLine",         &ScatterV1::treeLine,         0.0f,               1.0f             },
    {"scatter.altitudeFalloff",  &ScatterV1::altitudeFalloff,  0.0f,               64.0f            },
    {"scatter.endemicShare",     &ScatterV1::endemicShare,     0.0f,               1.0f             },
};

static_assert(sizeof(LivingV1) == 316u, "a field added to LivingV1 gets its bound in kLivingCounts or "
                                        "kLivingReals, or is a seed or a harmless word");

/** @brief The counts of a living recipe: sizes, populations and shares out of 16. */
inline constexpr CountBound<LivingV1> kLivingCounts[] = {
    {"width",               &LivingV1::width,               kMaxGridSide},
    {"depth",               &LivingV1::depth,               kMaxGridSide},
    {"ticks",               &LivingV1::ticks,               1u << 20u   },
    {"rooms",               &LivingV1::rooms,               4096u       },
    {"creatures",           &LivingV1::creatures,           4096u       },
    {"ants",                &LivingV1::ants,                4096u       },
    {"boids",               &LivingV1::boids,               4096u       },
    {"genomes",             &LivingV1::genomes,             4096u       },
    {"packMembers",         &LivingV1::packMembers,         256u        },
    {"explore16",           &LivingV1::explore16,           16u         },
    {"changeWeight",        &LivingV1::changeWeight,        1u << 16u   },
    {"adjacentBonus",       &LivingV1::adjacentBonus,       1u << 16u   },
    {"predictedBonus",      &LivingV1::predictedBonus,      1u << 16u   },
    {"unlikelyPenalty",     &LivingV1::unlikelyPenalty,     1u << 16u   },
    {"mutationChance16",    &LivingV1::mutationChance16,    16u         },
    {"collapseShare16",     &LivingV1::collapseShare16,     16u         },
    {"meltdownChance16",    &LivingV1::meltdownChance16,    16u         },
    {"dissolutionChance16", &LivingV1::dissolutionChance16, 16u         },
};

/** @brief The reals of a living recipe: each one becomes a Fixed32. */
inline constexpr RealBound<LivingV1> kLivingReals[] = {
    {"evaporation",       &LivingV1::evaporation,       -kMaxRealMagnitude, kMaxRealMagnitude},
    {"diffusion",         &LivingV1::diffusion,         -kMaxRealMagnitude, kMaxRealMagnitude},
    {"maximum",           &LivingV1::maximum,           -kMaxRealMagnitude, kMaxRealMagnitude},
    {"floorValue",        &LivingV1::floorValue,        -kMaxRealMagnitude, kMaxRealMagnitude},
    {"separationWeight",  &LivingV1::separationWeight,  -kMaxRealMagnitude, kMaxRealMagnitude},
    {"alignmentWeight",   &LivingV1::alignmentWeight,   -kMaxRealMagnitude, kMaxRealMagnitude},
    {"cohesionWeight",    &LivingV1::cohesionWeight,    -kMaxRealMagnitude, kMaxRealMagnitude},
    {"maxSpeed",          &LivingV1::maxSpeed,          -kMaxRealMagnitude, kMaxRealMagnitude},
    {"mutationAmplitude", &LivingV1::mutationAmplitude, -kMaxRealMagnitude, kMaxRealMagnitude},
    {"meltdownAmplitude", &LivingV1::meltdownAmplitude, -kMaxRealMagnitude, kMaxRealMagnitude},
    {"anomalySigma",      &LivingV1::anomalySigma,      -kMaxRealMagnitude, kMaxRealMagnitude},
};

static_assert(sizeof(LivingSpeciesV1) == 36u, "a field added to LivingSpeciesV1 gets its bound in kSpeciesRaws");

/**
 * @brief The Fixed32 words of a species: rates up to 16 a step, head counts up to 16383, so that
 *        the sum of two never overflows a Q16.16 word.
 */
inline constexpr RawBound<LivingSpeciesV1> kSpeciesRaws[] = {
    {"species.growth",     &LivingSpeciesV1::growth,     0, rawOf(16)   },
    {"species.mortality",  &LivingSpeciesV1::mortality,  0, rawOf(16)   },
    {"species.predation",  &LivingSpeciesV1::predation,  0, rawOf(16)   },
    {"species.conversion", &LivingSpeciesV1::conversion, 0, rawOf(16)   },
    {"species.capacity",   &LivingSpeciesV1::capacity,   0, rawOf(16383)},
    {"species.refuge",     &LivingSpeciesV1::refuge,     0, rawOf(16383)},
    {"species.initial",    &LivingSpeciesV1::initial,    0, rawOf(16383)},
};

/**
 * @brief Checks the bounds that span several fields of a world recipe: the levels of the volumes
 *        it stacks, and the steps of its aggregation.
 * @param wire       The recipe, its single fields already within their bounds.
 * @param outRefusal Receives the expression past its bound and what it came to.
 * @return false when one of them exceeds its bound.
 */
[[nodiscard]] constexpr bool withinProducts(const RecipeV1 &wire, WireRefusal &outRefusal) noexcept;
/** @} */

/**
 * @brief Reads one wire scatter rule, or refuses it.
 *
 * A free function because a rule is converted in both directions and a field-by-field
 * copy written twice is a field silently dropped in one of them the day a rule grows a
 * knob.
 *
 * @param from       The wire rule.
 * @param to         Receives the engine rule; untouched when refused.
 * @param outRefusal Receives the field refused and the value it held.
 * @return false when the rule names a biome this build does not know, or holds a real past
 *         @ref kScatterReals.
 */
[[nodiscard]] inline bool readScatterRule(const ScatterV1 &from, procgen::ScatterRule &to,
                                          WireRefusal &outRefusal) noexcept;

/**
 * @brief Flattens one scatter rule into its wire form.
 *
 * See @ref readScatterRule.
 */
inline void writeScatterRule(const procgen::ScatterRule &from, ScatterV1 &to) noexcept;

/**
 * @brief Reads a species' prey index, or refuses it.
 *
 * Refused, never clamped, exactly as an unknown enum is: the trophic web subscripts its species
 * vector by this index when it is stepped, so an index past the table a cartridge declares reads
 * and writes out of bounds — in ring 0, where the ecosystem runs, past no allocator guard. The
 * content hash is a checksum and not a signature, so a crafted image reaches here.
 *
 * @param value        The index the wire holds.
 * @param speciesCount Number of species the recipe declares.
 * @param out          Receives the index; untouched when refused.
 * @param outRefusal   Receives the field and the value it held when refused; untouched otherwise.
 * @return false when @p value is neither Species::kNoPrey nor a species the table holds.
 */
[[nodiscard]] inline bool readPreyIndex(core::u32 value, core::u32 speciesCount, core::u32 &out,
                                        WireRefusal &outRefusal) noexcept;

} // namespace detail

/**
 * @brief Expands a wire recipe into the engine's in-memory recipe, or refuses it.
 *
 * Refused when an enum field holds a value this build does not know (see @ref detail::readEnum),
 * or when a field holds more than its generator can hold (see @ref detail::kRecipeCounts,
 * @ref detail::kRecipeReals and @ref detail::withinProducts).
 *
 * A roadside pattern that arrived unterminated is a truncated grammar, not a shorter one: its
 * last byte becomes the terminator. The scatter count, which indexes an array, is clamped rather
 * than trusted: a cartridge is input, not a promise.
 *
 * @param wire       The decoded section payload.
 * @param outRecipe  Receives the recipe procgen::bakeWorld consumes; untouched when refused.
 * @param outRefusal Receives the first field refused and the value it held; untouched otherwise.
 * @return false when the recipe is refused.
 */
[[nodiscard]] inline bool toEngineRecipe(const RecipeV1 &wire, procgen::WorldRecipe &outRecipe,
                                         WireRefusal &outRefusal) noexcept;

/**
 * @brief Flattens an engine recipe into its wire form, for baking.
 * @param recipe The in-memory recipe.
 * @return The payload to write into a WorldRecipe section.
 */
[[nodiscard]] inline RecipeV1 toWireRecipe(const procgen::WorldRecipe &recipe) noexcept;

/**
 * @brief Wire living recipe to engine living recipe.
 *
 * Field by field and by NAME, like its world counterpart: a rename on either
 * side is then a compile error rather than a silently reinterpreted ecosystem.
 * Fixed32 crosses as its raw Q16.16 word — the value is the bits. Refused when a species
 * sits at a trophic level this build does not know (see @ref detail::readEnum), or when a field
 * holds more than its system can hold (see @ref detail::kLivingCounts, @ref detail::kLivingReals
 * and @ref detail::kSpeciesRaws).
 *
 * @param wire       Decoded living section.
 * @param outLiving  Receives the engine recipe it describes; untouched when refused.
 * @param outRefusal Receives the first field refused and the value it held; untouched otherwise.
 * @return false when the living recipe is refused.
 */
[[nodiscard]] inline bool toEngineLiving(const LivingV1 &wire, ecology::LivingRecipe &outLiving,
                                         WireRefusal &outRefusal) noexcept;

/**
 * @brief Engine living recipe to wire living recipe.
 * @param recipe Engine recipe.
 * @return Its wire form.
 */
[[nodiscard]] inline LivingV1 toWireLiving(const ecology::LivingRecipe &recipe) noexcept;

} // namespace lpl::pack

#    include "RecipeCodec.inl"

#endif // LPL_PACK_RECIPECODEC_HPP
