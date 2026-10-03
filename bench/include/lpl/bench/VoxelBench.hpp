/**
 * @file VoxelBench.hpp
 * @brief The voxel section of lpl-benchmark: the raymarcher, the mosaic lookup and the sheet tracer, timed on a
 * volume built in memory.
 *
 * @warning **A benchmark that needs a download is a benchmark nobody runs.** The fixture here is
 * built from an integer formula and looks like what the marcher actually meets: bright sheets a
 * few samples thick, spaced tens of samples apart, floating in a medium that is dark and NOT
 * empty. That last property is the one that matters for timing -- a field of solid against vacuum
 * lets every ray terminate or escape immediately and reports a speed the real thing never sees.
 *
 * @warning **The counts come out with the timing, and that is not decoration.** A frame that got
 * faster because rays stopped entering the volume is faster and wrong; samples, gradients and
 * saturated rays are what tell the two apart. The same holds for the tracer: a patch that came
 * back short does not time a whole patch, and its summary line says so.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_BENCH_VOXELBENCH_HPP
#    define LPL_BENCH_VOXELBENCH_HPP

namespace lpl::bench {

/// @brief Runs every voxel benchmark and prints its summary lines.
void runVoxelBenchmarks();

} // namespace lpl::bench

#endif // LPL_BENCH_VOXELBENCH_HPP
