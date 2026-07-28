/**
 * @file Climate.hpp
 * @brief The climate a cell sits in, as a point in a six-dimensional space.
 *
 * Two axes — warmth against moisture — draw Whittaker's diagram, and Whittaker's
 * diagram is a fine model of *vegetation*. It is a poor model of a *world*: it
 * cannot say that this plain is deep inland while that one is coastal, that this
 * ridge is jagged while that one is worn flat, or that here the ordinary rules
 * bend. Those distinctions are what stop a map from reading as contour bands with
 * colours on them.
 *
 * So the classification is not a cascade of thresholds on two numbers, it is a
 * nearest-neighbour lookup in a six-dimensional space — the multi-noise
 * architecture Minecraft 1.18 popularised. Each biome declares where it lives
 * (its centre) and which axes it actually cares about (its weights), and a cell
 * becomes whichever biome it is closest to. Adding an axis then costs one column
 * in a table instead of a new branch in every test, and — more importantly — a
 * biome that cares about nothing but temperature simply zeroes the other five
 * rather than pretending to have an opinion.
 *
 * The continuity guarantee comes for free and is the real prize: every axis is a
 * continuous field, so moving from one biome's neighbourhood to another's has to
 * cross the space between them. A glacier cannot abut a dune sea, because the
 * temperature axis has to pass through every value in between, and something else
 * is nearest along the way.
 *
 * @warning Every axis MUST be normalised to [0, 1] over the map before it is
 *          compared against a profile, and the two heavy-tailed ones (erosion,
 *          drainage) must be **rank**-normalised rather than min-max scaled. An
 *          axis left in absolute units is a threshold measured against a
 *          distribution that moves with the map's size — the same defect that
 *          made six biomes of twelve unreachable, and that flooded a small map
 *          with rivers. Twice was enough.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_PROCGEN_CLIMATE_HPP
#    define LPL_PROCGEN_CLIMATE_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/procgen/Grid.hpp>
#    include <lpl/procgen/Heightfield.hpp>
#    include <lpl/procgen/Hydrology.hpp>

namespace lpl::procgen {

/**
 * @enum ClimateAxis
 * @brief The six independent things that decide what grows somewhere.
 *
 * The order is wire-visible: it indexes @ref ClimateVector and @ref ClimateField,
 * and a profile table is written against it. Inserting an axis in the middle
 * silently re-reads every profile, so new axes go before @c Count and nowhere
 * else.
 */
enum class ClimateAxis : core::u8 {
    Temperature = 0, ///< Latitude gradient minus the altitude lapse rate.
    Moisture,        ///< Rainfall, drainage, coast and rain shadow (see @ref computeMoisture).
    Continentalness, ///< How far inland: 0 at the shore, 1 in the deep interior.
    Erosion,         ///< How worn the ground is: 0 for jagged peaks, 1 for a floodplain.
    Depth,           ///< Vertical position: 0 at the surface, 1 deep underground.
    Weirdness,       ///< Controlled anomaly — an independent field that shifts rare biomes in.
    Count            ///< Number of axes; never an index.
};

/// Number of climate axes, as a plain count for array sizes.
inline constexpr core::u32 kClimateAxisCount = static_cast<core::u32>(ClimateAxis::Count);

/**
 * @struct ClimateVector
 * @brief One cell's climate, or one biome's ideal, in the same units.
 *
 * The same type serves both on purpose: a profile's centre is literally a place
 * in the space a cell could occupy, and comparing them is then a subtraction
 * rather than a conversion.
 */
struct ClimateVector {
    math::Fixed32 axis[kClimateAxisCount]{};

    [[nodiscard]] math::Fixed32 &operator[](ClimateAxis a) noexcept { return axis[static_cast<core::u32>(a)]; }
    [[nodiscard]] const math::Fixed32 &operator[](ClimateAxis a) const noexcept
    {
        return axis[static_cast<core::u32>(a)];
    }
};

/**
 * @struct ClimateField
 * @brief The six axes sampled over the whole map.
 *
 * Six parallel grids rather than one grid of six-vectors. A pass almost always
 * wants one axis over many cells (normalise the erosion axis, fold the
 * temperature axis, draw the moisture axis), and that access pattern walks
 * contiguous memory here and strides by six the other way. It also means each
 * axis can be normalised on its own without touching the others.
 */
struct ClimateField {
    Heightfield axes[kClimateAxisCount];

    [[nodiscard]] Heightfield &operator[](ClimateAxis a) noexcept { return axes[static_cast<core::u32>(a)]; }
    [[nodiscard]] const Heightfield &operator[](ClimateAxis a) const noexcept
    {
        return axes[static_cast<core::u32>(a)];
    }

    [[nodiscard]] core::u32 width() const noexcept { return axes[0].width(); }
    [[nodiscard]] core::u32 depth() const noexcept { return axes[0].depth(); }
    [[nodiscard]] bool empty() const noexcept { return axes[0].empty(); }

    /// @brief Reads every axis at one cell.
    [[nodiscard]] ClimateVector at(core::u32 x, core::u32 z) const noexcept
    {
        ClimateVector v{};
        for (core::u32 i = 0u; i < kClimateAxisCount; ++i)
            v.axis[i] = axes[i].at(x, z);
        return v;
    }
};

/**
 * @struct ClimateParams
 * @brief What shapes each axis before a biome is ever named.
 */
struct ClimateParams {
    core::f32 coldLatitude{0.2f};      ///< Share of the map at each edge that is fully polar.
    core::f32 lapseRate{0.45f};        ///< Warmth lost between the lowest and highest ground, in [0, 1].
    core::f32 seaLevel{-4.0f};         ///< Height at or below which a cell is sea.
    core::f32 coastReach{0.25f};       ///< Share of the long axis over which continentalness saturates.
    core::u32 weirdnessSeed{0x7E12Du}; ///< Seed of the weirdness field.
    core::f32 weirdnessBelts{1.5f};    ///< Weirdness features across the map's longer axis.
    core::u32 weirdnessOctaves{2u};    ///< fBm octaves of the weirdness field.
    core::f32 surfaceDepth{0.0f};      ///< Depth value for the surface layer, in [0, 1].
};

/**
 * @brief Builds the six-axis climate field.
 *
 * Temperature and moisture are not recomputed here — moisture arrives already
 * built by @ref computeMoisture, and the latitude-plus-lapse-rate construction is
 * the one the biome classification used to keep to itself. There is one
 * definition of each axis in the module, which is the only way the map a debug
 * view draws and the map the classifier reads can be the same map.
 *
 * @param field    Terrain (absolute heights).
 * @param moisture Moisture in [0, 1], same dimensions as @p field.
 * @param network  Drainage, for the erosion axis' flow term.
 * @param params   Axis shaping.
 * @return All six axes, each normalised to [0, 1] (empty when inputs mismatch).
 */
[[nodiscard]] ClimateField computeClimate(const Heightfield &field, const Heightfield &moisture,
                                          const DrainageNetwork &network, const ClimateParams &params);

/**
 * @brief Rescales a field so its values span exactly [0, 1].
 *
 * For axes whose distribution is roughly even. A constant field becomes all
 * halves rather than dividing by zero.
 *
 * @param field Field to rescale in place.
 */
void normalizeUnit(Heightfield &field);

/**
 * @brief Replaces each value by its rank in the field's own distribution.
 *
 * For heavy-tailed axes, where min-max scaling is a trap: a handful of extreme
 * cells set the maximum and crush everything else into the bottom few percent, so
 * a threshold at "0.5" selects nothing and the axis carries no information. Rank
 * normalisation guarantees a flat distribution whatever the input's shape — the
 * median lands at 0.5 by construction, on a 24-cell map and a 1024-cell one
 * alike.
 *
 * Ranks are computed through a fixed-resolution histogram, so the result depends
 * only on the values present and never on the order they are visited.
 *
 * @param field Field to rank-normalise in place.
 */
void rankNormalize(Heightfield &field);

/**
 * @brief FNV-1a fold of every axis, in axis then storage order.
 * @param climate Field to fold.
 * @return The 32-bit signature.
 */
[[nodiscard]] core::u32 foldClimateField(const ClimateField &climate);

} // namespace lpl::procgen

#endif // LPL_PROCGEN_CLIMATE_HPP
