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
[[nodiscard]] constexpr bool isKnown(procgen::NoiseKind kind) noexcept
{
    switch (kind)
    {
    case procgen::NoiseKind::Fbm:
    case procgen::NoiseKind::Ridged:
    case procgen::NoiseKind::Billow: return true;
    }
    return false;
}

[[nodiscard]] constexpr bool isKnown(procgen::DistanceMetric metric) noexcept
{
    switch (metric)
    {
    case procgen::DistanceMetric::Euclidean:
    case procgen::DistanceMetric::Manhattan:
    case procgen::DistanceMetric::Chebyshev: return true;
    }
    return false;
}

[[nodiscard]] constexpr bool isKnown(procgen::CaveKind kind) noexcept
{
    switch (kind)
    {
    case procgen::CaveKind::Cellular:
    case procgen::CaveKind::Bsp:
    case procgen::CaveKind::Dla:
    case procgen::CaveKind::Layered:
    case procgen::CaveKind::Auto: return true;
    }
    return false;
}

[[nodiscard]] constexpr bool isKnown(procgen::BiomeId biome) noexcept
{
    switch (biome)
    {
    case procgen::BiomeId::Ocean:
    case procgen::BiomeId::Beach:
    case procgen::BiomeId::Snow:
    case procgen::BiomeId::Tundra:
    case procgen::BiomeId::Taiga:
    case procgen::BiomeId::Rock:
    case procgen::BiomeId::Desert:
    case procgen::BiomeId::Savanna:
    case procgen::BiomeId::Grassland:
    case procgen::BiomeId::Forest:
    case procgen::BiomeId::Rainforest:
    case procgen::BiomeId::Marsh:
    case procgen::BiomeId::Lake: return true;
    case procgen::BiomeId::Count: return false;
    }
    return false;
}

[[nodiscard]] constexpr bool isKnown(ecology::TrophicLevel level) noexcept
{
    switch (level)
    {
    case ecology::TrophicLevel::Producer:
    case ecology::TrophicLevel::Primary:
    case ecology::TrophicLevel::Secondary:
    case ecology::TrophicLevel::Apex: return true;
    }
    return false;
}
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
[[nodiscard]] constexpr bool readEnum(const char *field, core::u32 value, Enum &out, WireRefusal &outRefusal) noexcept
{
    const Enum candidate = static_cast<Enum>(value);

    if (static_cast<core::u32>(candidate) != value || !isKnown(candidate))
    {
        outRefusal = WireRefusal{field, value};
        return false;
    }
    out = candidate;
    return true;
}

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

/** @brief The raw word of the Fixed32 @p value, as a bound spells it. */
[[nodiscard]] constexpr core::i32 rawOf(core::i32 value) noexcept { return value * 65536; }

/** @brief Whether the IEEE-754 single @p bits encode a finite number; read from the bits, so no
 *         compiler flag can assume it away. */
[[nodiscard]] constexpr bool isFinite(core::u32 bits) noexcept { return (bits & 0x7F800000u) != 0x7F800000u; }

/** @brief @p value, or the largest 32-bit word when it does not fit in one: what a refusal reports. */
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
                                          WireRefusal &outRefusal) noexcept
{
    for (const CountBound<Section> &bound : bounds)
    {
        if (section.*bound.member > bound.most)
        {
            outRefusal = WireRefusal{bound.field, section.*bound.member};
            return false;
        }
    }
    return true;
}

/**
 * @brief Checks every real of @p section against @p bounds.
 * @param section    The section, as decoded.
 * @param bounds     Its real bounds.
 * @param outRefusal Receives the first field out of range and the bits it held.
 * @return false when a field is not finite or lies outside its range.
 */
template <typename Section, core::usize Count>
[[nodiscard]] constexpr bool withinReals(const Section &section, const RealBound<Section> (&bounds)[Count],
                                         WireRefusal &outRefusal) noexcept
{
    for (const RealBound<Section> &bound : bounds)
    {
        const core::f32 value = section.*bound.member;
        const core::u32 bits = std::bit_cast<core::u32>(value);

        if (!isFinite(bits) || value < bound.least || value > bound.most)
        {
            outRefusal = WireRefusal{bound.field, bits};
            return false;
        }
    }
    return true;
}

/**
 * @brief Checks every Fixed32 of @p section against @p bounds.
 * @param section    The section, as decoded.
 * @param bounds     Its raw bounds.
 * @param outRefusal Receives the first field out of range and the raw word it held.
 * @return false when a field lies outside its range.
 */
template <typename Section, core::usize Count>
[[nodiscard]] constexpr bool withinRaws(const Section &section, const RawBound<Section> (&bounds)[Count],
                                        WireRefusal &outRefusal) noexcept
{
    for (const RawBound<Section> &bound : bounds)
    {
        const core::i32 raw = section.*bound.member;

        if (raw < bound.least || raw > bound.most)
        {
            outRefusal = WireRefusal{bound.field, static_cast<core::u32>(raw)};
            return false;
        }
    }
    return true;
}

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
[[nodiscard]] constexpr bool withinProducts(const RecipeV1 &wire, WireRefusal &outRefusal) noexcept
{
    const core::u64 layers = wire.systemLayers < procgen::kMaxCaveLayers ? wire.systemLayers : procgen::kMaxCaveLayers;
    const core::u64 caveLevels = layers * wire.systemLevelsPerLayer;
    const core::u32 floors =
        wire.buildingMaxFloors > wire.buildingMinFloors ? wire.buildingMaxFloors : wire.buildingMinFloors;
    const core::u64 storeys = floors > 1u ? floors : 1u;
    const core::u64 storeyHeight = wire.buildingFloorHeight > 1u ? wire.buildingFloorHeight : 1u;
    const core::u64 buildingLevels =
        static_cast<core::u64>(wire.buildingBaseHeight) + storeys * storeyHeight + wire.buildingRoofHeight;
    const core::u64 aggregationSteps = static_cast<core::u64>(wire.dlaParticles) * wire.dlaMaxStepsPerParticle;

    if (caveLevels > kMaxVolumeLevels)
    {
        outRefusal = WireRefusal{"systemLayers * systemLevelsPerLayer", saturated(caveLevels)};
        return false;
    }
    if (buildingLevels > kMaxVolumeLevels)
    {
        outRefusal = WireRefusal{"buildingBaseHeight + buildingMaxFloors * buildingFloorHeight + buildingRoofHeight",
                                 saturated(buildingLevels)};
        return false;
    }
    if (aggregationSteps > kMaxAggregationSteps)
    {
        outRefusal = WireRefusal{"dlaParticles * dlaMaxStepsPerParticle", saturated(aggregationSteps)};
        return false;
    }
    return true;
}
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
                                          WireRefusal &outRefusal) noexcept
{
    if (!withinReals(from, kScatterReals, outRefusal) || !readEnum("scatter.biome", from.biome, to.biome, outRefusal))
        return false;
    to.density = from.density;
    to.halfExtent = from.halfExtent;
    to.maxSlope = from.maxSlope;
    to.minMoisture = from.minMoisture;
    to.maxMoisture = from.maxMoisture;
    to.moistureAffinity = from.moistureAffinity;
    to.tag = from.tag;
    to.treeLine = from.treeLine;
    to.altitudeFalloff = from.altitudeFalloff;
    to.maxRiverDistance = from.maxRiverDistance;
    to.endemicShare = from.endemicShare;
    to.collidable = (from.flags & kScatterFlagCollidable) != 0u;
    return true;
}

/// Flattens one scatter rule into its wire form. See @ref readScatterRule.
inline void writeScatterRule(const procgen::ScatterRule &from, ScatterV1 &to) noexcept
{
    to.biome = static_cast<core::u32>(from.biome);
    to.density = from.density;
    to.halfExtent = from.halfExtent;
    to.maxSlope = from.maxSlope;
    to.minMoisture = from.minMoisture;
    to.maxMoisture = from.maxMoisture;
    to.moistureAffinity = from.moistureAffinity;
    to.tag = from.tag;
    to.treeLine = from.treeLine;
    to.altitudeFalloff = from.altitudeFalloff;
    to.maxRiverDistance = from.maxRiverDistance;
    to.endemicShare = from.endemicShare;
    to.flags = from.collidable ? kScatterFlagCollidable : 0u;
}

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
                                        WireRefusal &outRefusal) noexcept
{
    if (value != ecology::Species::kNoPrey && value >= speciesCount)
    {
        outRefusal = WireRefusal{"species.preyIndex", value};
        return false;
    }
    out = value;
    return true;
}

} // namespace detail

/**
 * @brief Expands a wire recipe into the engine's in-memory recipe, or refuses it.
 *
 * Refused when an enum field holds a value this build does not know (see @ref detail::readEnum),
 * or when a field holds more than its generator can hold (see @ref detail::kRecipeCounts,
 * @ref detail::kRecipeReals and @ref detail::withinProducts).
 *
 * @param wire       The decoded section payload.
 * @param outRecipe  Receives the recipe procgen::bakeWorld consumes; untouched when refused.
 * @param outRefusal Receives the first field refused and the value it held; untouched otherwise.
 * @return false when the recipe is refused.
 */
[[nodiscard]] inline bool toEngineRecipe(const RecipeV1 &wire, procgen::WorldRecipe &outRecipe,
                                         WireRefusal &outRefusal) noexcept
{
    procgen::WorldRecipe recipe{};

    if (!detail::withinCounts(wire, detail::kRecipeCounts, outRefusal) ||
        !detail::withinReals(wire, detail::kRecipeReals, outRefusal) || !detail::withinProducts(wire, outRefusal))
        return false;

    recipe.seed = wire.seed;
    recipe.width = wire.width;
    recipe.depth = wire.depth;
    recipe.cellSize = wire.cellSize;

    recipe.terrain.seed = wire.noiseSeed;
    recipe.terrain.frequency = wire.noiseFrequency;
    recipe.terrain.amplitude = wire.noiseAmplitude;
    recipe.terrain.octaves = wire.noiseOctaves;
    recipe.terrain.baseHeight = wire.noiseBaseHeight;
    recipe.terrain.lacunarity = wire.noiseLacunarity;
    recipe.terrain.persistence = wire.noisePersistence;
    recipe.terrain.warpStrength = wire.noiseWarpStrength;
    if (!detail::readEnum("noiseKind", wire.noiseKind, recipe.terrain.kind, outRefusal))
        return false;
    recipe.heightLow = wire.heightLow;
    recipe.heightHigh = wire.heightHigh;
    recipe.groundClearance = wire.groundClearance;

    recipe.thermal.iterations = wire.thermalIterations;
    recipe.thermal.talus = wire.thermalTalus;
    recipe.thermal.carryFraction = wire.thermalCarryFraction;

    recipe.hydraulic.iterations = wire.hydraulicIterations;
    recipe.hydraulic.rainAmount = wire.hydraulicRainAmount;
    recipe.hydraulic.solubility = wire.hydraulicSolubility;
    recipe.hydraulic.evaporation = wire.hydraulicEvaporation;
    recipe.hydraulic.sedimentCapacity = wire.hydraulicSedimentCapacity;
    recipe.hydraulic.deposition = wire.hydraulicDeposition;
    recipe.hydraulic.minSlope = wire.hydraulicMinSlope;

    recipe.rivers.density = wire.riverDensity;
    recipe.rivers.carveDepth = wire.riverCarveDepth;
    recipe.rivers.smoothing = wire.riverSmoothing;

    recipe.climate.rainfallWeight = wire.climateRainfallWeight;
    recipe.climate.flowWeight = wire.climateFlowWeight;
    recipe.climate.altitudeWeight = wire.climateAltitudeWeight;
    recipe.climate.coastWeight = wire.climateCoastWeight;
    recipe.climate.seaLevel = wire.climateSeaLevel;
    recipe.climate.coastReach = wire.climateCoastReach;
    recipe.climate.rainShadow = wire.climateRainShadow;
    recipe.climate.windDirection = wire.climateWindDirection;
    recipe.climate.smoothing = wire.climateSmoothing;
    recipe.climate.rainfallSeed = wire.climateRainfallSeed;
    recipe.climate.rainfallBelts = wire.climateRainfallBelts;
    recipe.climate.rainfallOctaves = wire.climateRainfallOctaves;

    recipe.biomes.seaLevel = wire.biomeSeaLevel;
    recipe.biomes.beachHeight = wire.biomeBeachHeight;
    recipe.biomes.mountainHeight = wire.biomeMountainHeight;
    recipe.biomes.snowHeight = wire.biomeSnowHeight;
    recipe.biomes.snowlineWarmth = wire.biomeSnowlineWarmth;

    recipe.axes.coldLatitude = wire.axisColdLatitude;
    recipe.axes.lapseRate = wire.axisLapseRate;
    recipe.axes.coastReach = wire.axisCoastReach;
    recipe.axes.weirdnessSeed = wire.axisWeirdnessSeed;
    recipe.axes.weirdnessBelts = wire.axisWeirdnessBelts;
    recipe.axes.weirdnessOctaves = wire.axisWeirdnessOctaves;
    recipe.axes.surfaceDepth = wire.axisSurfaceDepth;

    recipe.provinces.width = wire.provinceWidth;
    recipe.provinces.depth = wire.provinceDepth;
    recipe.provinces.seed = wire.provinceSeed;
    recipe.provinces.cellSize = wire.provinceCellSize;
    recipe.provinces.jitter = wire.provinceJitter;
    recipe.provinces.warpStrength = wire.provinceWarpStrength;
    if (!detail::readEnum("provinceMetric", wire.provinceMetric, recipe.provinces.metric, outRefusal))
        return false;

    recipe.terraceSteps = wire.terraceSteps;
    if (!detail::readEnum("caveKind", wire.caveKind, recipe.caveKind, outRefusal))
        return false;

    recipe.rooms.width = wire.roomsWidth;
    recipe.rooms.depth = wire.roomsDepth;
    recipe.rooms.seed = wire.roomsSeed;
    recipe.rooms.maxDepth = wire.roomsMaxDepth;
    recipe.rooms.minLeafSize = wire.roomsMinLeafSize;
    recipe.rooms.roomPadding = wire.roomsPadding;
    recipe.rooms.corridorWidth = wire.roomsCorridorWidth;

    recipe.aggregation.width = wire.dlaWidth;
    recipe.aggregation.depth = wire.dlaDepth;
    recipe.aggregation.seed = wire.dlaSeed;
    recipe.aggregation.particles = wire.dlaParticles;
    recipe.aggregation.maxStepsPerParticle = wire.dlaMaxStepsPerParticle;
    recipe.aggregation.spawnMargin = wire.dlaSpawnMargin;
    recipe.aggregation.thickness = wire.dlaThickness;

    recipe.caveSystem.width = wire.systemWidth;
    recipe.caveSystem.depth = wire.systemDepth;
    recipe.caveSystem.seed = wire.systemSeed;
    recipe.caveSystem.layers = wire.systemLayers;
    recipe.caveSystem.levelsPerLayer = wire.systemLevelsPerLayer;
    recipe.caveSystem.topFill = wire.systemTopFill;
    recipe.caveSystem.deepFill = wire.systemDeepFill;
    recipe.caveSystem.automatonSteps = wire.systemAutomatonSteps;
    recipe.caveSystem.minChamberSize = wire.systemMinChamberSize;
    recipe.caveSystem.shaftsPerPair = wire.systemShaftsPerPair;
    recipe.caveSystem.entrances = wire.systemEntrances;
    recipe.caveSystem.entranceMaxSlope = wire.systemEntranceMaxSlope;

    recipe.buildings.seed = wire.buildingSeed;
    recipe.buildings.minFloors = wire.buildingMinFloors;
    recipe.buildings.maxFloors = wire.buildingMaxFloors;
    recipe.buildings.baseHeight = wire.buildingBaseHeight;
    recipe.buildings.floorHeight = wire.buildingFloorHeight;
    recipe.buildings.roofHeight = wire.buildingRoofHeight;
    recipe.buildings.inset = wire.buildingInset;
    recipe.buildings.roofTaper = wire.buildingRoofTaper;
    recipe.buildings.baseMaterial = static_cast<core::u8>(wire.buildingBaseMaterial);
    recipe.buildings.wallMaterial = static_cast<core::u8>(wire.buildingWallMaterial);
    recipe.buildings.roofMaterial = static_cast<core::u8>(wire.buildingRoofMaterial);
    recipe.buildings.hollow = (wire.flags & kRecipeFlagBuildingsHollow) != 0u;

    recipe.roadsideLevels = wire.roadsideLevels;
    for (core::u32 i = 0u; i < kWireRoadsidePattern; ++i)
        recipe.roadsidePattern[i] = wire.roadsidePattern[i];
    // A pattern that arrived unterminated is a truncated grammar, not a shorter one.
    recipe.roadsidePattern[kWireRoadsidePattern - 1u] = '\0';

    recipe.partitionRegions = (wire.flags & kRecipeFlagPartitionRegions) != 0u;
    recipe.raiseBuildings = (wire.flags & kRecipeFlagRaiseBuildings) != 0u;

    recipe.caves.width = wire.caveWidth;
    recipe.caves.depth = wire.caveDepth;
    recipe.caves.seed = wire.caveSeed;
    recipe.caves.fillProbability = wire.caveFillProbability;
    recipe.caves.steps = wire.caveSteps;
    recipe.caves.birthLimit = wire.caveBirthLimit;
    recipe.caves.survivalLimit = wire.caveSurvivalLimit;
    recipe.caves.minRegionSize = wire.caveMinRegionSize;

    recipe.settlement.seed = wire.settlementSeed;
    recipe.settlement.districtSize = wire.settlementDistrictSize;
    recipe.settlement.roadWidth = wire.settlementRoadWidth;
    recipe.settlement.plazaRadius = wire.settlementPlazaRadius;
    recipe.settlement.minPlot = wire.settlementMinPlot;
    recipe.settlement.maxPlot = wire.settlementMaxPlot;
    recipe.settlement.plotDensity = wire.settlementPlotDensity;
    recipe.settlement.maxSlope = wire.settlementMaxSlope;
    recipe.settlement.minHeight = wire.settlementMinHeight;

    recipe.roads.seed = wire.roadSeed;
    recipe.roads.iterations = wire.roadIterations;
    recipe.roads.stepLength = wire.roadStepLength;
    recipe.roads.conform = wire.roadConform;
    recipe.roads.maxSlope = wire.roadMaxSlope;
    recipe.roads.minHeight = wire.roadMinHeight;
    recipe.roads.gridDistricts = wire.roadGridDistricts;
    recipe.roads.arterials = (wire.flags & kRecipeFlagRoadArterials) != 0u;

    recipe.gate.minPathLength = wire.gateMinPathLength;
    recipe.gate.minWalkableCells = wire.gateMinWalkableCells;
    recipe.gate.maxDeadEndRatio = wire.gateMaxDeadEndRatio;
    recipe.gate.requireGoalReachable = (wire.flags & kRecipeFlagGateRequireGoal) != 0u;
    recipe.gate.requireFullyConnected = (wire.flags & kRecipeFlagGateRequireConnected) != 0u;

    // A count from untrusted bytes indexes an array, so it is clamped here
    // rather than trusted: a cartridge is input, not a promise.
    recipe.scatterCount = wire.scatterCount < kWireScatterRules ? wire.scatterCount : kWireScatterRules;
    for (core::u32 i = 0u; i < recipe.scatterCount; ++i)
        if (!detail::readScatterRule(wire.scatter[i], recipe.scatter[i], outRefusal))
            return false;

    recipe.normalizeTerrain = (wire.flags & kRecipeFlagNormalizeTerrain) != 0u;
    recipe.erodeTerrain = (wire.flags & kRecipeFlagErodeTerrain) != 0u;
    recipe.carveRivers = (wire.flags & kRecipeFlagCarveRivers) != 0u;
    recipe.classifyBiomes = (wire.flags & kRecipeFlagClassifyBiomes) != 0u;
    recipe.carveCaves = (wire.flags & kRecipeFlagCarveCaves) != 0u;
    recipe.placeSettlement = (wire.flags & kRecipeFlagPlaceSettlement) != 0u;
    recipe.materializeGround = (wire.flags & kRecipeFlagMaterializeGround) != 0u;
    recipe.growRoads = (wire.flags & kRecipeFlagGrowRoads) != 0u;
    recipe.checkPlayability = (wire.flags & kRecipeFlagCheckPlayability) != 0u;

    outRecipe = recipe;
    return true;
}

/**
 * @brief Flattens an engine recipe into its wire form, for baking.
 * @param recipe The in-memory recipe.
 * @return The payload to write into a WorldRecipe section.
 */
[[nodiscard]] inline RecipeV1 toWireRecipe(const procgen::WorldRecipe &recipe) noexcept
{
    RecipeV1 wire{};

    wire.seed = recipe.seed;
    wire.width = recipe.width;
    wire.depth = recipe.depth;
    wire.cellSize = recipe.cellSize;

    wire.noiseSeed = recipe.terrain.seed;
    wire.noiseFrequency = recipe.terrain.frequency;
    wire.noiseAmplitude = recipe.terrain.amplitude;
    wire.noiseOctaves = recipe.terrain.octaves;
    wire.noiseBaseHeight = recipe.terrain.baseHeight;
    wire.noiseLacunarity = recipe.terrain.lacunarity;
    wire.noisePersistence = recipe.terrain.persistence;
    wire.noiseWarpStrength = recipe.terrain.warpStrength;
    wire.noiseKind = static_cast<core::u32>(recipe.terrain.kind);
    wire.heightLow = recipe.heightLow;
    wire.heightHigh = recipe.heightHigh;
    wire.groundClearance = recipe.groundClearance;

    wire.thermalIterations = recipe.thermal.iterations;
    wire.thermalTalus = recipe.thermal.talus;
    wire.thermalCarryFraction = recipe.thermal.carryFraction;

    wire.hydraulicIterations = recipe.hydraulic.iterations;
    wire.hydraulicRainAmount = recipe.hydraulic.rainAmount;
    wire.hydraulicSolubility = recipe.hydraulic.solubility;
    wire.hydraulicEvaporation = recipe.hydraulic.evaporation;
    wire.hydraulicSedimentCapacity = recipe.hydraulic.sedimentCapacity;
    wire.hydraulicDeposition = recipe.hydraulic.deposition;
    wire.hydraulicMinSlope = recipe.hydraulic.minSlope;

    wire.riverDensity = recipe.rivers.density;
    wire.riverCarveDepth = recipe.rivers.carveDepth;
    wire.riverSmoothing = recipe.rivers.smoothing;

    wire.climateRainfallWeight = recipe.climate.rainfallWeight;
    wire.climateFlowWeight = recipe.climate.flowWeight;
    wire.climateAltitudeWeight = recipe.climate.altitudeWeight;
    wire.climateCoastWeight = recipe.climate.coastWeight;
    wire.climateSeaLevel = recipe.climate.seaLevel;
    wire.climateCoastReach = recipe.climate.coastReach;
    wire.climateRainShadow = recipe.climate.rainShadow;
    wire.climateWindDirection = recipe.climate.windDirection;
    wire.climateSmoothing = recipe.climate.smoothing;
    wire.climateRainfallSeed = recipe.climate.rainfallSeed;
    wire.climateRainfallBelts = recipe.climate.rainfallBelts;
    wire.climateRainfallOctaves = recipe.climate.rainfallOctaves;

    wire.biomeSeaLevel = recipe.biomes.seaLevel;
    wire.biomeBeachHeight = recipe.biomes.beachHeight;
    wire.biomeMountainHeight = recipe.biomes.mountainHeight;
    wire.biomeSnowHeight = recipe.biomes.snowHeight;
    wire.axisColdLatitude = recipe.axes.coldLatitude;
    wire.axisLapseRate = recipe.axes.lapseRate;
    wire.axisCoastReach = recipe.axes.coastReach;
    wire.axisWeirdnessSeed = recipe.axes.weirdnessSeed;
    wire.axisWeirdnessBelts = recipe.axes.weirdnessBelts;
    wire.axisWeirdnessOctaves = recipe.axes.weirdnessOctaves;
    wire.axisSurfaceDepth = recipe.axes.surfaceDepth;
    wire.biomeSnowlineWarmth = recipe.biomes.snowlineWarmth;

    wire.provinceWidth = recipe.provinces.width;
    wire.provinceDepth = recipe.provinces.depth;
    wire.provinceSeed = recipe.provinces.seed;
    wire.provinceCellSize = recipe.provinces.cellSize;
    wire.provinceJitter = recipe.provinces.jitter;
    wire.provinceWarpStrength = recipe.provinces.warpStrength;
    wire.provinceMetric = static_cast<core::u32>(recipe.provinces.metric);

    wire.terraceSteps = recipe.terraceSteps;
    wire.caveKind = static_cast<core::u32>(recipe.caveKind);

    wire.roomsWidth = recipe.rooms.width;
    wire.roomsDepth = recipe.rooms.depth;
    wire.roomsSeed = recipe.rooms.seed;
    wire.roomsMaxDepth = recipe.rooms.maxDepth;
    wire.roomsMinLeafSize = recipe.rooms.minLeafSize;
    wire.roomsPadding = recipe.rooms.roomPadding;
    wire.roomsCorridorWidth = recipe.rooms.corridorWidth;

    wire.dlaWidth = recipe.aggregation.width;
    wire.dlaDepth = recipe.aggregation.depth;
    wire.dlaSeed = recipe.aggregation.seed;
    wire.dlaParticles = recipe.aggregation.particles;
    wire.dlaMaxStepsPerParticle = recipe.aggregation.maxStepsPerParticle;
    wire.dlaSpawnMargin = recipe.aggregation.spawnMargin;
    wire.dlaThickness = recipe.aggregation.thickness;

    wire.systemWidth = recipe.caveSystem.width;
    wire.systemDepth = recipe.caveSystem.depth;
    wire.systemSeed = recipe.caveSystem.seed;
    wire.systemLayers = recipe.caveSystem.layers;
    wire.systemLevelsPerLayer = recipe.caveSystem.levelsPerLayer;
    wire.systemTopFill = recipe.caveSystem.topFill;
    wire.systemDeepFill = recipe.caveSystem.deepFill;
    wire.systemAutomatonSteps = recipe.caveSystem.automatonSteps;
    wire.systemMinChamberSize = recipe.caveSystem.minChamberSize;
    wire.systemShaftsPerPair = recipe.caveSystem.shaftsPerPair;
    wire.systemEntrances = recipe.caveSystem.entrances;
    wire.systemEntranceMaxSlope = recipe.caveSystem.entranceMaxSlope;

    wire.buildingSeed = recipe.buildings.seed;
    wire.buildingMinFloors = recipe.buildings.minFloors;
    wire.buildingMaxFloors = recipe.buildings.maxFloors;
    wire.buildingBaseHeight = recipe.buildings.baseHeight;
    wire.buildingFloorHeight = recipe.buildings.floorHeight;
    wire.buildingRoofHeight = recipe.buildings.roofHeight;
    wire.buildingInset = recipe.buildings.inset;
    wire.buildingRoofTaper = recipe.buildings.roofTaper;
    wire.buildingBaseMaterial = recipe.buildings.baseMaterial;
    wire.buildingWallMaterial = recipe.buildings.wallMaterial;
    wire.buildingRoofMaterial = recipe.buildings.roofMaterial;

    wire.roadsideLevels = recipe.roadsideLevels;
    for (core::u32 i = 0u; i < kWireRoadsidePattern; ++i)
        wire.roadsidePattern[i] = recipe.roadsidePattern[i];
    wire.roadsidePattern[kWireRoadsidePattern - 1u] = '\0';

    wire.caveWidth = recipe.caves.width;
    wire.caveDepth = recipe.caves.depth;
    wire.caveSeed = recipe.caves.seed;
    wire.caveFillProbability = recipe.caves.fillProbability;
    wire.caveSteps = recipe.caves.steps;
    wire.caveBirthLimit = recipe.caves.birthLimit;
    wire.caveSurvivalLimit = recipe.caves.survivalLimit;
    wire.caveMinRegionSize = recipe.caves.minRegionSize;

    wire.settlementSeed = recipe.settlement.seed;
    wire.settlementDistrictSize = recipe.settlement.districtSize;
    wire.settlementRoadWidth = recipe.settlement.roadWidth;
    wire.settlementPlazaRadius = recipe.settlement.plazaRadius;
    wire.settlementMinPlot = recipe.settlement.minPlot;
    wire.settlementMaxPlot = recipe.settlement.maxPlot;
    wire.settlementPlotDensity = recipe.settlement.plotDensity;
    wire.settlementMaxSlope = recipe.settlement.maxSlope;
    wire.settlementMinHeight = recipe.settlement.minHeight;

    wire.roadSeed = recipe.roads.seed;
    wire.roadIterations = recipe.roads.iterations;
    wire.roadStepLength = recipe.roads.stepLength;
    wire.roadConform = recipe.roads.conform;
    wire.roadMaxSlope = recipe.roads.maxSlope;
    wire.roadMinHeight = recipe.roads.minHeight;
    wire.roadGridDistricts = recipe.roads.gridDistricts;

    wire.gateMinPathLength = recipe.gate.minPathLength;
    wire.gateMinWalkableCells = recipe.gate.minWalkableCells;
    wire.gateMaxDeadEndRatio = recipe.gate.maxDeadEndRatio;

    wire.scatterCount = recipe.scatterCount < kWireScatterRules ? recipe.scatterCount : kWireScatterRules;
    for (core::u32 i = 0u; i < wire.scatterCount; ++i)
    {
        const procgen::ScatterRule &from = recipe.scatter[i];
        ScatterV1 &to = wire.scatter[i];
        to.biome = static_cast<core::u32>(from.biome);
        to.density = from.density;
        to.halfExtent = from.halfExtent;
        to.maxSlope = from.maxSlope;
        to.minMoisture = from.minMoisture;
        to.maxMoisture = from.maxMoisture;
        to.moistureAffinity = from.moistureAffinity;
        to.tag = from.tag;
        to.treeLine = from.treeLine;
        to.altitudeFalloff = from.altitudeFalloff;
        to.maxRiverDistance = from.maxRiverDistance;
        to.endemicShare = from.endemicShare;
        to.flags = from.collidable ? kScatterFlagCollidable : 0u;
    }

    wire.flags = 0u;
    if (recipe.normalizeTerrain)
        wire.flags |= kRecipeFlagNormalizeTerrain;
    if (recipe.erodeTerrain)
        wire.flags |= kRecipeFlagErodeTerrain;
    if (recipe.carveRivers)
        wire.flags |= kRecipeFlagCarveRivers;
    if (recipe.classifyBiomes)
        wire.flags |= kRecipeFlagClassifyBiomes;
    if (recipe.carveCaves)
        wire.flags |= kRecipeFlagCarveCaves;
    if (recipe.placeSettlement)
        wire.flags |= kRecipeFlagPlaceSettlement;
    if (recipe.materializeGround)
        wire.flags |= kRecipeFlagMaterializeGround;
    if (recipe.growRoads)
        wire.flags |= kRecipeFlagGrowRoads;
    if (recipe.roads.arterials)
        wire.flags |= kRecipeFlagRoadArterials;
    if (recipe.checkPlayability)
        wire.flags |= kRecipeFlagCheckPlayability;
    if (recipe.gate.requireGoalReachable)
        wire.flags |= kRecipeFlagGateRequireGoal;
    if (recipe.gate.requireFullyConnected)
        wire.flags |= kRecipeFlagGateRequireConnected;
    if (recipe.partitionRegions)
        wire.flags |= kRecipeFlagPartitionRegions;
    if (recipe.raiseBuildings)
        wire.flags |= kRecipeFlagRaiseBuildings;
    if (recipe.buildings.hollow)
        wire.flags |= kRecipeFlagBuildingsHollow;

    return wire;
}

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
                                         WireRefusal &outRefusal) noexcept
{
    ecology::LivingRecipe recipe{};

    if (!detail::withinCounts(wire, detail::kLivingCounts, outRefusal) ||
        !detail::withinReals(wire, detail::kLivingReals, outRefusal))
        return false;

    recipe.seed = wire.seed;
    recipe.ticks = wire.ticks;
    recipe.stepSeconds = math::Fixed32::fromRaw(wire.stepSeconds);

    recipe.width = wire.width;
    recipe.depth = wire.depth;
    recipe.channels = wire.channels;

    recipe.rooms = wire.rooms;
    recipe.creatures = wire.creatures;
    recipe.ants = wire.ants;
    recipe.boids = wire.boids;
    recipe.genomes = wire.genomes;
    recipe.packMembers = wire.packMembers;
    recipe.regrowthTicks = wire.regrowthTicks;
    recipe.headPerBody = wire.headPerBody;

    recipe.stigmergy.evaporation = wire.evaporation;
    recipe.stigmergy.diffusion = wire.diffusion;
    recipe.stigmergy.maximum = wire.maximum;
    recipe.stigmergy.floor = wire.floorValue;

    recipe.foraging.explore16 = wire.explore16;
    recipe.foraging.depositQuality = math::Fixed32::fromRaw(wire.depositQuality);
    recipe.foraging.channel = wire.trailChannel;

    recipe.flock.separationRadius = math::Fixed32::fromRaw(wire.separationRadius);
    recipe.flock.neighbourRadius = math::Fixed32::fromRaw(wire.neighbourRadius);
    recipe.flock.separationWeight = wire.separationWeight;
    recipe.flock.alignmentWeight = wire.alignmentWeight;
    recipe.flock.cohesionWeight = wire.cohesionWeight;
    recipe.flock.maxSpeed = wire.maxSpeed;

    recipe.budget.maxRealisedRooms = wire.maxRealisedRooms;
    recipe.budget.changeWeight = wire.changeWeight;
    recipe.budget.adjacentBonus = wire.adjacentBonus;
    recipe.budget.predictedBonus = wire.predictedBonus;
    recipe.budget.unlikelyPenalty = wire.unlikelyPenalty;

    recipe.heredity.mutationChance16 = wire.mutationChance16;
    recipe.heredity.mutationAmplitude = wire.mutationAmplitude;
    recipe.heredity.collapseShare16 = wire.collapseShare16;
    recipe.heredity.meltdownChance16 = wire.meltdownChance16;
    recipe.heredity.meltdownAmplitude = wire.meltdownAmplitude;
    recipe.heredity.anomalySigma = wire.anomalySigma;

    recipe.packs.maxSize = wire.packMaxSize;
    recipe.packs.minSize = wire.packMinSize;
    recipe.packs.dissolutionChance16 = wire.dissolutionChance16;
    recipe.packs.adoptStrays = wire.adoptStrays != 0u;

    const core::u32 count =
        wire.speciesCount < ecology::kMaxLivingSpecies ? wire.speciesCount : ecology::kMaxLivingSpecies;
    for (core::u32 i = 0u; i < count; ++i)
    {
        const LivingSpeciesV1 &from = wire.species[i];
        ecology::LivingSpecies &to = recipe.species[i];
        if (!detail::withinRaws(from, detail::kSpeciesRaws, outRefusal) ||
            !detail::readEnum("species.level", from.level, to.params.level, outRefusal))
            return false;
        to.params.growth = math::Fixed32::fromRaw(from.growth);
        to.params.mortality = math::Fixed32::fromRaw(from.mortality);
        to.params.predation = math::Fixed32::fromRaw(from.predation);
        to.params.conversion = math::Fixed32::fromRaw(from.conversion);
        to.params.capacity = math::Fixed32::fromRaw(from.capacity);
        to.params.refuge = math::Fixed32::fromRaw(from.refuge);
        to.initial = math::Fixed32::fromRaw(from.initial);
        if (!detail::readPreyIndex(from.preyIndex, count, to.preyIndex, outRefusal))
            return false;
    }
    recipe.speciesCount = count;

    outLiving = recipe;
    return true;
}

/**
 * @brief Engine living recipe to wire living recipe.
 * @param recipe Engine recipe.
 * @return Its wire form.
 */
[[nodiscard]] inline LivingV1 toWireLiving(const ecology::LivingRecipe &recipe) noexcept
{
    LivingV1 wire{};

    wire.seed = recipe.seed;
    wire.ticks = recipe.ticks;
    wire.stepSeconds = recipe.stepSeconds.raw();

    wire.width = recipe.width;
    wire.depth = recipe.depth;
    wire.channels = recipe.channels;

    wire.rooms = recipe.rooms;
    wire.creatures = recipe.creatures;
    wire.ants = recipe.ants;
    wire.boids = recipe.boids;
    wire.genomes = recipe.genomes;
    wire.packMembers = recipe.packMembers;
    wire.regrowthTicks = recipe.regrowthTicks;
    wire.headPerBody = recipe.headPerBody;

    wire.evaporation = recipe.stigmergy.evaporation;
    wire.diffusion = recipe.stigmergy.diffusion;
    wire.maximum = recipe.stigmergy.maximum;
    wire.floorValue = recipe.stigmergy.floor;

    wire.explore16 = recipe.foraging.explore16;
    wire.depositQuality = recipe.foraging.depositQuality.raw();
    wire.trailChannel = recipe.foraging.channel;

    wire.separationRadius = recipe.flock.separationRadius.raw();
    wire.neighbourRadius = recipe.flock.neighbourRadius.raw();
    wire.separationWeight = recipe.flock.separationWeight;
    wire.alignmentWeight = recipe.flock.alignmentWeight;
    wire.cohesionWeight = recipe.flock.cohesionWeight;
    wire.maxSpeed = recipe.flock.maxSpeed;

    wire.maxRealisedRooms = recipe.budget.maxRealisedRooms;
    wire.changeWeight = recipe.budget.changeWeight;
    wire.adjacentBonus = recipe.budget.adjacentBonus;
    wire.predictedBonus = recipe.budget.predictedBonus;
    wire.unlikelyPenalty = recipe.budget.unlikelyPenalty;

    wire.mutationChance16 = recipe.heredity.mutationChance16;
    wire.mutationAmplitude = recipe.heredity.mutationAmplitude;
    wire.collapseShare16 = recipe.heredity.collapseShare16;
    wire.meltdownChance16 = recipe.heredity.meltdownChance16;
    wire.meltdownAmplitude = recipe.heredity.meltdownAmplitude;
    wire.anomalySigma = recipe.heredity.anomalySigma;

    wire.packMaxSize = recipe.packs.maxSize;
    wire.packMinSize = recipe.packs.minSize;
    wire.dissolutionChance16 = recipe.packs.dissolutionChance16;
    wire.adoptStrays = recipe.packs.adoptStrays ? 1u : 0u;

    const core::u32 count =
        recipe.speciesCount < ecology::kMaxLivingSpecies ? recipe.speciesCount : ecology::kMaxLivingSpecies;
    for (core::u32 i = 0u; i < count; ++i)
    {
        const ecology::LivingSpecies &from = recipe.species[i];
        LivingSpeciesV1 &to = wire.species[i];
        to.level = static_cast<core::u32>(from.params.level);
        to.growth = from.params.growth.raw();
        to.mortality = from.params.mortality.raw();
        to.predation = from.params.predation.raw();
        to.conversion = from.params.conversion.raw();
        to.capacity = from.params.capacity.raw();
        to.refuge = from.params.refuge.raw();
        to.initial = from.initial.raw();
        to.preyIndex = from.preyIndex;
    }
    wire.speciesCount = count;

    return wire;
}

} // namespace lpl::pack

#endif // LPL_PACK_RECIPECODEC_HPP
