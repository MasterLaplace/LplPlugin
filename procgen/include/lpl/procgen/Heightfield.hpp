/**
 * @file Heightfield.hpp
 * @brief The terrain grid, and the passes that shape it before anything else runs.
 *
 * A heightfield is the first thing a world becomes and the last thing to be
 * turned into entities. Everything between — erosion, rivers, biomes — reads and
 * writes this grid.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_PROCGEN_HEIGHTFIELD_HPP
#    define LPL_PROCGEN_HEIGHTFIELD_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/math/FixedPoint.hpp>
#    include <lpl/procgen/Grid.hpp>

namespace lpl::procgen {

/// Terrain elevation per cell, in world units.
using Heightfield = Grid<math::Fixed32>;

/**
 * @enum NoiseKind
 * @brief Which fractal construction shapes a layer.
 */
enum class NoiseKind : core::u8 {
    Fbm = 0, ///< Symmetric fractal sum: rolling hills.
    Ridged,  ///< Crests along the noise's zero set: mountain ranges.
    Billow   ///< Rectified: rounded bulges with creases, for dunes and clouds.
};

/**
 * @struct NoiseParams
 * @brief How the base terrain is sampled before any pass reshapes it.
 */
struct NoiseParams {
    core::u32 seed{1337u};          ///< Determinism anchor.
    core::f32 frequency{0.05f};     ///< Cells-to-noise scale; smaller is smoother.
    core::f32 amplitude{16.0f};     ///< Peak-to-trough height.
    core::u32 octaves{5u};          ///< Octaves summed.
    core::f32 baseHeight{0.0f};     ///< Constant added to every cell (sea level offset).
    core::f32 lacunarity{2.0f};     ///< Frequency multiplier per octave.
    core::f32 persistence{0.5f};    ///< Amplitude multiplier per octave.
    core::f32 warpStrength{0.0f};   ///< Domain warp, in cells; 0 leaves the lattice unwarped.
    NoiseKind kind{NoiseKind::Fbm}; ///< Fractal construction.

    /**
     * @brief Flattens the middle of the range into plains. 0 leaves noise as noise.
     *
     * Raw fractal noise makes terrain nobody wants to walk on. Its values are
     * distributed around the mean, so MOST of the world sits on a slope — and with
     * an amplitude big enough for real mountains, most of the world is a slope you
     * cannot climb. What comes out is exactly what this world came out as: a ridge,
     * a drowned valley, another ridge, and no ground between them. Raising the
     * amplitude makes the mountains right and the rest worse; lowering it makes the
     * walking right and flattens the mountains. There is no amplitude that gives
     * both, because the problem is the DISTRIBUTION, not the scale.
     *
     * So the noise is reshaped before it becomes a height: the middle of the range
     * is compressed towards a plain, and only the top of it is allowed to rise. That
     * is the trick every open-world terrain uses (Minecraft calls the input
     * continentalness and the curve a spline); the parameters here are the smallest
     * set that expresses it.
     *
     * At 1 the mid band is dead flat, which reads as a table. Around 0.8 leaves
     * enough undulation to be a landscape and still be walkable.
     */
    core::f32 plainsFlatten{0.0f};
    /**
     * @brief Share of the noise range treated as lowland, in [0, 1).
     *
     * Wide is the point: at 0.55 more than half of everything generated is ground a
     * player can walk across, which is what "an open world" means as a measurable
     * property rather than an impression.
     */
    core::f32 plainsWidth{0.55f};
    /**
     * @brief Where mountains begin, as a share of the range above the plains.
     *
     * Rare, because a mountain that is everywhere is not a mountain, it is the
     * ground. Everything between the plains and here is foothill.
     */
    core::f32 mountainThreshold{0.62f};
    /**
     * @brief How much taller a mountain is than the raw amplitude would make it.
     *
     * This is what buys real SCALE without ruining the rest: the flattened band and
     * the raised peaks are two ends of the same curve, so the mountains can be four
     * times the amplitude while the plains get calmer rather than steeper. Walking
     * at eye height, that is the difference between a lawn with bumps and a valley
     * with a mountain at the end of it.
     */
    core::f32 mountainGain{1.0f};
};

/**
 * @brief Reshapes a normalised noise value into a terrain profile.
 *
 * Input and output are both roughly [-1, 1]; what changes is how the range is SPENT.
 * Three bands, and the boundaries are continuous so no seam appears where one ends:
 *
 *  - below the plains band: ocean, left proportional so a coast still shelves.
 *  - the plains band: compressed towards its own middle by @c plainsFlatten.
 *  - above the mountain threshold: expanded by @c mountainGain.
 *
 * Pure Fixed32, no transcendentals, and a function of the value alone — so it is
 * seamless by construction, and a chunk boundary cannot see it.
 *
 * @param value Normalised noise, nominally [-1, 1].
 * @param params The layer whose shaping parameters apply.
 * @return The reshaped value.
 */
[[nodiscard]] math::Fixed32 shapeTerrainValue(math::Fixed32 value, const NoiseParams &params);

/**
 * @brief What @p params says the elevation is at one world coordinate.
 *
 * The single definition of a noise layer, and deliberately the only one. Both a
 * whole-map generator and a chunked one need to answer this question, and if each
 * spelled the answer out itself they would drift apart the moment a parameter was
 * added — a chunk would then disagree with the map it is part of, which is
 * exactly the failure a seamless scheme exists to prevent.
 *
 * Coordinates are absolute world cells, not offsets into a grid: that is what
 * makes two chunks agree about the ground they share.
 *
 * @param worldX Absolute cell abscissa.
 * @param worldZ Absolute cell ordinate.
 * @param params Layer description.
 * @return The layer's contribution, including its base height.
 */
[[nodiscard]] math::Fixed32 sampleNoiseAt(core::i32 worldX, core::i32 worldZ, const NoiseParams &params);

/**
 * @brief Builds a heightfield by sampling a noise layer.
 * @param width  Cells along X.
 * @param depth  Cells along Z.
 * @param params Sampling parameters.
 * @return The generated field (empty when width or depth is 0).
 */
[[nodiscard]] Heightfield generateNoiseHeightfield(core::u32 width, core::u32 depth, const NoiseParams &params);

/**
 * @brief Adds a second noise layer on top of an existing field.
 *
 * Layering is how a terrain stops looking like one noise function: a low
 * frequency lays down continents, a higher one roughens them.
 *
 * @param field  Field to modify in place.
 * @param params Layer parameters.
 */
void addNoiseLayer(Heightfield &field, const NoiseParams &params);

/**
 * @brief Lowest and highest cell of @p field.
 * @param field   Field to measure.
 * @param outMin  Receives the minimum.
 * @param outMax  Receives the maximum.
 * @return false when the field is empty (outputs untouched).
 */
bool heightRange(const Heightfield &field, math::Fixed32 &outMin, math::Fixed32 &outMax);

/**
 * @brief Rescales every cell so the field spans exactly [@p low, @p high].
 *
 * Erosion and layering both change a field's range unpredictably. Renormalising
 * afterwards is what lets later passes use absolute thresholds — "below 0.3 is
 * ocean" only means something once the range is known.
 *
 * @param field Field to rescale in place.
 * @param low   Target minimum.
 * @param high  Target maximum.
 */
void normalizeHeights(Heightfield &field, math::Fixed32 low, math::Fixed32 high);

/**
 * @brief Raises every cell below @p level up to it, flattening the sea floor.
 * @param field Field to modify.
 * @param level Water level.
 */
void clampToSeaLevel(Heightfield &field, math::Fixed32 level);

/**
 * @brief Applies a terracing (stepped plateau) effect.
 * @param field Field to modify.
 * @param steps Number of terraces; 0 or 1 leaves the field untouched.
 */
void terrace(Heightfield &field, core::u32 steps);

/**
 * @brief Averages each cell with its 8 neighbours, @p iterations times.
 * @param field      Field to smooth in place.
 * @param iterations Passes to run.
 */
void smoothHeights(Heightfield &field, core::u32 iterations);

/**
 * @brief Steepness at a cell: the largest absolute drop to a 4-neighbour.
 * @param field Field to measure.
 * @param x     Column.
 * @param z     Row.
 * @return The slope magnitude in world units per cell.
 */
[[nodiscard]] math::Fixed32 slopeAt(const Heightfield &field, core::u32 x, core::u32 z);

/**
 * @brief FNV-1a fold of every cell's raw Q16.16 word, in storage order.
 *
 * The determinism gate for a pass: run it twice, or on two targets, and compare.
 *
 * @param field Field to fold.
 * @return The 32-bit signature.
 */
[[nodiscard]] core::u32 foldHeightfield(const Heightfield &field);

} // namespace lpl::procgen

#endif // LPL_PROCGEN_HEIGHTFIELD_HPP
