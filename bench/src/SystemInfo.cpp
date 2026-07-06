/**
 * @file SystemInfo.cpp
 * @brief Portable host hardware/toolchain introspection (Linux / macOS /
 *        Windows, GCC / Clang / MSVC on x86/x64).
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-06
 * @copyright MIT License
 */

#include <lpl/bench/SystemInfo.hpp>

#include <lpl/core/Platform.hpp>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <thread>

#if defined(LPL_OS_LINUX)
#    include <unistd.h>
#elif defined(LPL_OS_MACOS)
#    include <sys/sysctl.h>
#    include <unistd.h>
#endif

#if defined(LPL_OS_LINUX) || defined(LPL_OS_MACOS)
#    include <sys/utsname.h>
#elif defined(LPL_OS_WINDOWS)
#    define WIN32_LEAN_AND_MEAN
#    include <windows.h>
#endif

#if (defined(LPL_ARCH_X64) || defined(LPL_ARCH_X86)) && (defined(LPL_COMPILER_GCC) || defined(LPL_COMPILER_CLANG))
#    include <cpuid.h>
#elif (defined(LPL_ARCH_X64) || defined(LPL_ARCH_X86)) && defined(LPL_COMPILER_MSVC)
#    include <intrin.h>
#endif

namespace lpl::bench {
namespace {

/// Reads the x86 CPUID brand string (leaves 0x80000002..4). Returns an empty
/// string off x86, where no portable equivalent exists without going
/// OS-specific.
std::string cpuBrandString()
{
#if (defined(LPL_ARCH_X64) || defined(LPL_ARCH_X86)) &&                                                                 \
    (defined(LPL_COMPILER_GCC) || defined(LPL_COMPILER_CLANG) || defined(LPL_COMPILER_MSVC))
    core::u32 regs[12] = {};
#    if defined(LPL_COMPILER_MSVC)
    __cpuid(reinterpret_cast<int *>(regs + 0), static_cast<int>(0x80000002));
    __cpuid(reinterpret_cast<int *>(regs + 4), static_cast<int>(0x80000003));
    __cpuid(reinterpret_cast<int *>(regs + 8), static_cast<int>(0x80000004));
#    else
    __get_cpuid(0x80000002, regs + 0, regs + 1, regs + 2, regs + 3);
    __get_cpuid(0x80000003, regs + 4, regs + 5, regs + 6, regs + 7);
    __get_cpuid(0x80000004, regs + 8, regs + 9, regs + 10, regs + 11);
#    endif
    char brand[49] = {};
    std::memcpy(brand, regs, sizeof(regs));
    std::string s{brand};
    const auto first = s.find_first_not_of(' ');
    return first == std::string::npos ? std::string{} : s.substr(first);
#else
    return {};
#endif
}

/// Returns "<OS> (<kernel/build release>) <arch>".
std::string osDescription()
{
#if defined(LPL_OS_LINUX) || defined(LPL_OS_MACOS)
    struct utsname uts {};
    if (uname(&uts) == 0)
        return std::string(uts.sysname) + " (" + uts.release + ") " + uts.machine;
    return "Unix (unknown release)";
#elif defined(LPL_OS_WINDOWS)
    // RtlGetVersion (resolved dynamically) avoids the deprecated GetVersionEx
    // and the version lie it tells without an application manifest.
    using RtlGetVersionFn = LONG(WINAPI *)(OSVERSIONINFOEXW *);
    std::string desc = "Windows";
    if (HMODULE ntdll = GetModuleHandleW(L"ntdll.dll"))
    {
        auto fn = reinterpret_cast<RtlGetVersionFn>(GetProcAddress(ntdll, "RtlGetVersion"));
        OSVERSIONINFOEXW info{};
        info.dwOSVersionInfoSize = sizeof(info);
        if (fn && fn(&info) == 0)
        {
            desc += " " + std::to_string(info.dwMajorVersion) + "." + std::to_string(info.dwMinorVersion) +
                    " (build " + std::to_string(info.dwBuildNumber) + ")";
        }
    }
    return desc;
#else
    return "Unknown OS";
#endif
}

/// Returns total physical RAM in bytes, or 0 if it could not be determined.
core::u64 totalPhysicalRamBytes()
{
#if defined(LPL_OS_LINUX)
    const long pages = sysconf(_SC_PHYS_PAGES);
    const long pageSize = sysconf(_SC_PAGESIZE);
    return (pages > 0 && pageSize > 0) ? static_cast<core::u64>(pages) * static_cast<core::u64>(pageSize) : 0;
#elif defined(LPL_OS_MACOS)
    core::u64 memBytes = 0;
    std::size_t size = sizeof(memBytes);
    sysctlbyname("hw.memsize", &memBytes, &size, nullptr, 0);
    return memBytes;
#elif defined(LPL_OS_WINDOWS)
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    return GlobalMemoryStatusEx(&status) ? static_cast<core::u64>(status.ullTotalPhys) : 0;
#else
    return 0;
#endif
}

/// Best-effort read of the Linux CPU frequency-scaling governor. A governor
/// other than "performance" means the OS may throttle or boost cores mid-run,
/// a real source of timing noise. Returns "unknown" elsewhere or when
/// unreadable (e.g. inside a VM or WSL2, where cpufreq is not exposed).
std::string cpuGovernor()
{
#if defined(LPL_OS_LINUX)
    std::ifstream f("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor");
    std::string governor;
    if (f >> governor)
        return governor;
#endif
    return "unknown";
}

/// Returns "<Compiler name> <version>".
std::string compilerDescription()
{
#if defined(LPL_COMPILER_CLANG)
    return std::string("Clang ") + __clang_version__;
#elif defined(LPL_COMPILER_GCC)
    return std::string("GCC ") + __VERSION__;
#elif defined(LPL_COMPILER_MSVC)
    return "MSVC " + std::to_string(_MSC_FULL_VER);
#else
    return "Unknown compiler";
#endif
}

/// Build configuration selected by the xmake mode (see root xmake.lua).
const char *buildConfig()
{
#if defined(LPL_RELEASE)
    return "Release";
#elif defined(LPL_PROFILE)
    return "Profile";
#elif defined(LPL_DEBUG)
    return "Debug";
#else
    return "Unknown";
#endif
}

} // namespace

SystemInfo collectSystemInfo()
{
    SystemInfo info;
    info.os = osDescription();
    const std::string cpu = cpuBrandString();
    info.cpu = cpu.empty() ? "Unknown" : cpu;
    info.logicalCores = std::thread::hardware_concurrency();
    info.ramBytes = totalPhysicalRamBytes();
    info.compiler = compilerDescription();
    info.buildConfig = buildConfig();
    info.cpuGovernor = cpuGovernor();
    return info;
}

void printSystemInfo(const SystemInfo &info)
{
    const auto now = std::chrono::system_clock::now();
    const std::time_t nowC = std::chrono::system_clock::to_time_t(now);
    char timeBuf[32] = {};
    std::strftime(timeBuf, sizeof(timeBuf), "%Y-%m-%d %H:%M:%S", std::localtime(&nowC));

    const core::f64 ramGiB = static_cast<core::f64>(info.ramBytes) / (1024.0 * 1024.0 * 1024.0);

    std::printf("=== System Info ===\n");
    std::printf("  Date            : %s (local)\n", timeBuf);
    std::printf("  OS              : %s\n", info.os.c_str());
    std::printf("  CPU             : %s\n", info.cpu.c_str());
    std::printf("  Logical cores   : %u\n", info.logicalCores);
    std::printf("  RAM             : %.2f GiB\n", ramGiB);
    std::printf("  Compiler        : %s\n", info.compiler.c_str());
    std::printf("  Build config    : %s\n", info.buildConfig.c_str());
#if defined(LPL_OS_LINUX)
    std::printf("  CPU governor    : %s%s\n", info.cpuGovernor.c_str(),
                info.cpuGovernor == "performance" ? "" : "  [warning: may add timing noise]");
#endif
    std::printf("\n");
}

void printSystemInfo()
{
    printSystemInfo(collectSystemInfo());
}

} // namespace lpl::bench
