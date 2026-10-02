/**
 * @file EnergyMeter.cpp
 * @brief Locating and reading the package energy counter through Linux powercap.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-09-24
 * @copyright MIT License
 */

#include <lpl/bench/EnergyMeter.hpp>

#include <cerrno>
#include <cstdio>
#include <cstring>

namespace lpl::bench {

namespace {

/** Powercap zones are numbered; the package is conventionally the first, but is checked by name. */
constexpr int kMaximumZonesScanned = 16;

/**
 * @brief Reads one unsigned decimal value from a sysfs file.
 * @param path File to read.
 * @param out Receives the value.
 * @return The errno of the open, or 0 on success.
 */
int readSysfsValue(const std::string &path, core::u64 &out)
{
    std::FILE *file = std::fopen(path.c_str(), "r");
    if (file == nullptr)
        return errno;

    unsigned long long value = 0;
    const int matched = std::fscanf(file, "%llu", &value);
    std::fclose(file);
    if (matched != 1)
        return EIO;

    out = static_cast<core::u64>(value);
    return 0;
}

/**
 * @brief Reads the first line of a sysfs file.
 * @param path File to read.
 * @return The line, or an empty string.
 */
std::string readSysfsLine(const std::string &path)
{
    std::FILE *file = std::fopen(path.c_str(), "r");
    if (file == nullptr)
        return {};

    char line[128] = {};
    if (std::fgets(line, sizeof(line), file) == nullptr)
        line[0] = '\0';
    std::fclose(file);

    std::string text(line);
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r'))
        text.pop_back();
    return text;
}

/**
 * @brief Whether this Linux runs as a WSL2 guest.
 * @return true when the kernel release names Microsoft.
 */
bool runningUnderWindowsSubsystem()
{
    const std::string release = readSysfsLine("/proc/sys/kernel/osrelease");
    return release.find("microsoft") != std::string::npos || release.find("Microsoft") != std::string::npos;
}

} // namespace

std::optional<core::u64> energyDeltaMicrojoules(core::u64 before, core::u64 after, core::u64 rangeMicrojoules) noexcept
{
    if (after >= before)
        return after - before;

    if (rangeMicrojoules == 0u || before > rangeMicrojoules)
        return std::nullopt;

    return (rangeMicrojoules - before) + after;
}

std::string formatEnergy(core::f64 microjoules)
{
    const char *unit = "uJ";
    core::f64 value = microjoules;
    if (microjoules >= 1e6)
    {
        value = microjoules / 1e6;
        unit = "J";
    }
    else if (microjoules >= 1e3)
    {
        value = microjoules / 1e3;
        unit = "mJ";
    }

    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.4g %s", value, unit);
    return std::string(buffer);
}

EnergyMeter EnergyMeter::probe()
{
    EnergyMeter meter;
    std::string deniedPath;

    for (int zone = 0; zone < kMaximumZonesScanned; ++zone)
    {
        const std::string base = "/sys/class/powercap/intel-rapl:" + std::to_string(zone);
        const std::string name = readSysfsLine(base + "/name");
        if (name.rfind("package", 0) != 0)
            continue;

        core::u64 probeValue = 0;
        const int status = readSysfsValue(base + "/energy_uj", probeValue);
        if (status == EACCES || status == EPERM)
        {
            deniedPath = base + "/energy_uj";
            continue;
        }
        if (status != 0)
            continue;

        core::u64 range = 0;
        if (readSysfsValue(base + "/max_energy_range_uj", range) != 0 || range == 0u)
            continue;

        meter._availability = EnergyAvailability::Measured;
        meter._counterPath = base + "/energy_uj";
        meter._range = range;
        meter._description = name + " (intel-rapl:" + std::to_string(zone) + "), whole package";
        return meter;
    }

    if (!deniedPath.empty())
    {
        meter._availability = EnergyAvailability::Denied;
        meter._description =
            "denied: " + deniedPath +
            " is readable by root only since the PLATYPUS side channel; run as root or grant read access";
        return meter;
    }

    meter._availability = EnergyAvailability::Absent;
    meter._description = runningUnderWindowsSubsystem() ?
                             "absent: WSL2 does not pass RAPL through to the Linux guest; boot Linux natively or "
                             "measure on the target hardware" :
                             "absent: no powercap package zone under /sys/class/powercap";
    return meter;
}

bool EnergyMeter::read(core::u64 &outMicrojoules) const
{
    if (_availability != EnergyAvailability::Measured)
        return false;

    return readSysfsValue(_counterPath, outMicrojoules) == 0;
}

const EnergyMeter &energyMeter()
{
    static const EnergyMeter meter = EnergyMeter::probe();
    return meter;
}

} // namespace lpl::bench
