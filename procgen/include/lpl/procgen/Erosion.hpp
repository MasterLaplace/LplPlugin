/**
 * @file Erosion.hpp
 * @brief The passes that make terrain look like it has a history.
 *
 * Raw fBm reads as noise because nothing has ever happened to it: slopes are
 * uniformly rough at every scale and ridges have no direction. Erosion is what
 * removes that signature — it carries material downhill, so valleys widen,
 * ridges sharpen, and debris collects at the foot of slopes.
 *
 * Two complementary models, both grid-based:
 *
 *  - **Thermal** (Musgrave's talus model): material above a critical slope
 *    slides to lower neighbours. Cheap, and the one that produces scree slopes
 *    and flattens impossible cliffs.
 *  - **Hydraulic**: rain falls, water flows downhill, and how much material it
 *    can hold depends on how fast it is going. This is what carves drainage
 *    networks.
 *
 * The hydraulic model turns on one equation, the sediment transport capacity
 * @f$C = K_c \cdot |slope| \cdot water@f$. Water below its capacity picks
 * material up; water above it puts material down. Every reference states it this
 * way, and the slope term is the whole point: without it, water dissolves ground
 * at the same rate on a flat plain as in a gorge, so the pass lowers the terrain
 * uniformly and carves nothing. That is not a subtle difference — measured on a
 * 64x64 field, a slope-blind pass moved marginally *more* material on flat cells
 * than on steep ones.
 *
 * Both are grid relaxations rather than particle simulations: a droplet model
 * would need a random walk per drop and thousands of drops, and its result would
 * depend on the order drops were drawn. A grid pass visits every cell in a fixed
 * order, every iteration, which is both cheaper and reproducible by
 * construction.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_PROCGEN_EROSION_HPP
#    define LPL_PROCGEN_EROSION_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/procgen/Heightfield.hpp>

namespace lpl::procgen {

/**
 * @struct ThermalErosionParams
 * @brief Talus-angle relaxation.
 */
struct ThermalErosionParams {
    core::u32 iterations{16u};     ///< Relaxation passes.
    core::f32 talus{0.6f};         ///< Height difference per cell a slope holds before sliding.
    core::f32 carryFraction{0.5f}; ///< Share of the excess moved per pass, in [0, 1].
};

/**
 * @struct HydraulicErosionParams
 * @brief Rain, dissolve, transport, evaporate.
 */
struct HydraulicErosionParams {
    core::u32 iterations{24u};        ///< Rain/flow/evaporate cycles.
    core::f32 rainAmount{0.02f};      ///< Water added per cell per cycle.
    core::f32 solubility{0.35f};      ///< Share of the capacity deficit dissolved per cycle.
    core::f32 evaporation{0.35f};     ///< Share of water lost per cycle, in [0, 1].
    core::f32 sedimentCapacity{6.0f}; ///< Sediment a unit of water holds per unit of slope.
    core::f32 deposition{0.3f};       ///< Share of the excess deposited per cycle, in [0, 1].
    core::f32 minSlope{0.02f};        ///< Slope floor, so still water still carries a little.
};

/**
 * @brief Slides material off slopes steeper than the talus angle.
 * @param field  Terrain to erode in place.
 * @param params Talus and iteration count.
 * @return Total material moved, in world units (0 when nothing was unstable).
 */
math::Fixed32 thermalErode(Heightfield &field, const ThermalErosionParams &params);

/**
 * @brief Runs rainfall-driven erosion and deposition.
 * @param field  Terrain to erode in place.
 * @param params Rain, solubility, evaporation and capacity.
 * @return Total material displaced, in world units.
 */
math::Fixed32 hydraulicErode(Heightfield &field, const HydraulicErosionParams &params);

} // namespace lpl::procgen

#endif // LPL_PROCGEN_EROSION_HPP
