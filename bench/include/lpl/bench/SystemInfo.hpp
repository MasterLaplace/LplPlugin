/**
 * @file SystemInfo.hpp
 * @brief Host hardware and toolchain introspection for self-documenting
 *        benchmark runs.
 *
 * A benchmark number is only meaningful next to the machine, compiler, and
 * build configuration it was produced on. @ref collectSystemInfo gathers that
 * context (CPU, cores, hypervisor, RAM, OS, compiler, build config, frequency governor,
 * commit) and @ref printSystemInfo emits it as a header block before any measurement runs.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-06
 * @copyright MIT License
 */
#pragma once

#ifndef LPL_BENCH_SYSTEM_INFO_HPP
#    define LPL_BENCH_SYSTEM_INFO_HPP

#    include <lpl/core/Types.hpp>

#    include <string>

namespace lpl::bench {

/**
 * @brief Snapshot of the host context a benchmark result must be read against.
 */
struct SystemInfo {
    std::string os;             ///< OS name, kernel/build release, and architecture.
    std::string platform;       ///< OS and architecture the build targets, no release: "Linux x86_64".
    std::string cpu;            ///< CPU brand string, or "Unknown" off x86.
    core::u32 logicalCores = 0; ///< std::thread::hardware_concurrency().
    std::string hypervisor;     ///< Vendor CPUID names, "none" on bare metal, "unknown" off x86.
    core::u64 ramBytes = 0;     ///< Total physical RAM, or 0 if undetermined.
    std::string compiler;       ///< Compiler name and version.
    std::string buildConfig;    ///< "Debug" / "Release" / "Profile" / "Unknown".
    std::string cpuGovernor;    ///< Linux frequency governor, else "unknown".
    std::string commit;         ///< Short commit built from, "-dirty" if modified, else "unknown".
};

/**
 * @brief The key that groups the runs made on one kind of machine.
 *
 * @details Made of what sets the speed of the same code on two machines, and only that: the
 *          platform, the processor brand string, the logical core count, and the hypervisor
 *          the run sits under or "bare metal", e.g.
 *          "Linux x86_64, Intel(R) Core(TM) Ultra 7 165H, 22 logical cores, hypervisor Microsoft Hv".
 *          The host name, the kernel release, the RAM, the compiler, the build, the governor and
 *          the commit are left out: they change between runs of one machine, or are recorded
 *          beside it.
 *
 * @param info The host context.
 * @return The class, one line.
 */
[[nodiscard]] std::string machineClass(const SystemInfo &info);

/**
 * @brief Gathers the host context in a portable way (Linux / macOS / Windows,
 *        GCC / Clang / MSVC on x86/x64).
 * @return Filled @ref SystemInfo. Fields that cannot be determined on the
 *         current platform fall back to "Unknown" / 0.
 */
[[nodiscard]] SystemInfo collectSystemInfo();

/**
 * @brief Prints @p info as a labelled block, prefixed with the local date and
 *        followed by a warning when the CPU governor is not "performance"
 *        (a real source of timing noise).
 * @param info Context to print.
 */
void printSystemInfo(const SystemInfo &info);

/**
 * @brief Convenience overload: @ref collectSystemInfo then @ref printSystemInfo.
 */
void printSystemInfo();

} // namespace lpl::bench

#endif // LPL_BENCH_SYSTEM_INFO_HPP
