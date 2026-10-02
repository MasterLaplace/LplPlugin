/**
 * @file VoxelBench.hpp
 * @brief A volume the raymarcher can be measured against, without a network or a corpus.
 *
 * @warning **A benchmark that needs a download is a benchmark nobody runs.** The fixture here is
 * built from an integer formula and looks like what the marcher actually meets: bright sheets a
 * few samples thick, spaced tens of samples apart, floating in a medium that is dark and NOT
 * empty. That last property is the one that matters for timing -- a field of solid against vacuum
 * lets every ray terminate or escape immediately and reports a speed the real thing never sees.
 *
 * @warning **The counts come out with the timing, and that is not decoration.** A frame that got
 * faster because rays stopped entering the volume is faster and wrong; samples, gradients and
 * saturated rays are what tell the two apart.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_BENCH_VOXELBENCH_HPP
#    define LPL_BENCH_VOXELBENCH_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/voxel/Raymarch.hpp>
#    include <lpl/voxel/Volume.hpp>

#    include <vector>

namespace lpl::bench {

/**
 * @struct VoxelScene
 * @brief A resident set of synthetic bricks, plus everything a march needs.
 */
struct VoxelScene final {
    std::vector<std::vector<core::u8>> storage;
    std::vector<std::vector<core::u8>> cells; ///< Occupancy, one grid per brick.
    voxel::BrickMosaic mosaic;
    voxel::VolumeGeometry geometry{};
    voxel::DensityProfile profile{};
    voxel::TransferFunction transfer{};
    voxel::Eye eye{};
};

/**
 * @brief Builds a cube of @p bricksPerAxis bricks of sheets, at the given pyramid levels.
 *
 * @param bricksPerAxis  Bricks along each axis at the finest level. Three is a small working set
 *                       that stays in cache; five or more starts to look like a real resident set,
 *                       and the difference between those two numbers IS the memory behaviour.
 * @param levels         How many pyramid levels to populate around the centre. Levels overlap, so
 *                       this exercises the lookup's level walk rather than just its hit path.
 */
[[nodiscard]] VoxelScene makeSheetScene(core::u32 bricksPerAxis, core::u32 levels);

/// @brief Runs every voxel benchmark and prints its rows.
void runVoxelBenchmarks();

} // namespace lpl::bench

#endif // LPL_BENCH_VOXELBENCH_HPP
