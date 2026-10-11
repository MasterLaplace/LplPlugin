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
#include <utility>

namespace lpl::bench {

namespace {

/** Powercap zones are numbered; the package is conventionally the first, but is checked by name. */
constexpr int kMaximumZonesScanned = 16;

/**
 * @brief Reads one unsigned decimal value from a sysfs file.
 * @param path File to read.
 * @param out Receives the value.
 * @return 0 on success, the errno of the open when it fails, or EIO when the read fails or
 *         the file does not start with a decimal value.
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

/**
 * @brief Why a meter gives no energy, when it is not measuring.
 * @param availability What the meter can claim.
 * @return The meter's own reason, or nothing when it is measuring.
 */
std::optional<EnergyAbsence> absenceOfAMeterNotMeasuring(EnergyAvailability availability) noexcept
{
    switch (availability)
    {
    case EnergyAvailability::Measured: return std::nullopt;
    case EnergyAvailability::Denied: return EnergyAbsence::Denied;
    case EnergyAvailability::Unreadable: return EnergyAbsence::Unreadable;
    case EnergyAvailability::Absent: return EnergyAbsence::Absent;
    }
    std::unreachable();
}

} // namespace

std::string_view energyAbsenceName(EnergyAbsence absence) noexcept
{
    switch (absence)
    {
    case EnergyAbsence::Denied: return "denied";
    case EnergyAbsence::Unreadable: return "unreadable";
    case EnergyAbsence::Absent: return "absent";
    case EnergyAbsence::NoRepetition: return "no-repetition";
    case EnergyAbsence::ShortWindow: return "short-window";
    case EnergyAbsence::ReadFailed: return "read-failed";
    case EnergyAbsence::AmbiguousWrap: return "ambiguous-wrap";
    }
    std::unreachable();
}

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

EnergyMeter::EnergyMeter(EnergyAvailability availability, std::string description, std::string counterPath,
                         core::u64 range)
    : _availability{availability}, _description{std::move(description)}, _counterPath{std::move(counterPath)},
      _range{range}
{
}

EnergyMeter EnergyMeter::unreadable(const std::string &path, std::string_view reason)
{
    return EnergyMeter{EnergyAvailability::Unreadable, "unreadable: " + path + ": " + std::string{reason}};
}

EnergyMeter EnergyMeter::probe(std::string_view powercapRoot)
{
    std::optional<EnergyMeter> firstDenied;
    std::optional<EnergyMeter> firstUnreadable;

    for (int zone = 0; zone < kMaximumZonesScanned; ++zone)
    {
        EnergyMeter candidate = probeZone(powercapRoot, zone);
        switch (candidate._availability)
        {
        case EnergyAvailability::Measured: return candidate;
        case EnergyAvailability::Denied:
            if (!firstDenied)
                firstDenied = std::move(candidate);
            break;
        case EnergyAvailability::Unreadable:
            if (!firstUnreadable)
                firstUnreadable = std::move(candidate);
            break;
        case EnergyAvailability::Absent: break;
        }
    }

    if (firstDenied)
        return *std::move(firstDenied);
    if (firstUnreadable)
        return *std::move(firstUnreadable);
    if (powercapRoot == kPowercapRoot && runningUnderWindowsSubsystem())
        return EnergyMeter{EnergyAvailability::Absent, "absent: WSL2 does not pass RAPL through to the Linux guest; "
                                                       "boot Linux natively or measure on the target hardware"};
    return EnergyMeter{EnergyAvailability::Absent,
                       "absent: no powercap package zone under " + std::string{powercapRoot}};
}

EnergyMeter EnergyMeter::probeZone(std::string_view powercapRoot, int zone)
{
    const std::string zoneId = "intel-rapl:" + std::to_string(zone);
    const std::string zoneDirectory = std::string{powercapRoot} + "/" + zoneId;
    const std::string zoneLabel = readSysfsLine(zoneDirectory + "/name");
    if (!zoneLabel.starts_with("package"))
        return EnergyMeter{EnergyAvailability::Absent, {}};

    const std::string counterPath = zoneDirectory + "/energy_uj";
    core::u64 counter = 0u;
    const int counterError = readSysfsValue(counterPath, counter);
    if (counterError == EACCES || counterError == EPERM)
        return EnergyMeter{EnergyAvailability::Denied,
                           "denied: " + counterPath +
                               " is readable by root only since the PLATYPUS side channel; run as root or grant "
                               "read access"};
    if (counterError != 0)
        return unreadable(counterPath, std::strerror(counterError));

    const std::string rangePath = zoneDirectory + "/max_energy_range_uj";
    core::u64 range = 0u;
    const int rangeError = readSysfsValue(rangePath, range);
    if (rangeError != 0)
        return unreadable(rangePath, std::strerror(rangeError));
    if (range == 0u)
        return unreadable(rangePath, "a range of zero cannot unwrap the counter");

    return EnergyMeter{EnergyAvailability::Measured, zoneLabel + " (" + zoneId + "), whole package", counterPath,
                       range};
}

std::optional<core::u64> EnergyMeter::read() const
{
    if (_availability != EnergyAvailability::Measured)
        return std::nullopt;

    core::u64 microjoules = 0u;
    if (readSysfsValue(_counterPath, microjoules) != 0)
        return std::nullopt;
    return microjoules;
}

EnergyBracket::EnergyBracket(const EnergyMeter &meter)
    : _meter{meter}, _openedAt{std::chrono::steady_clock::now()}, _counterAtOpening{meter.read()}
{
}

std::expected<core::f64, EnergyAbsence> EnergyBracket::microjoulesPerRepetition(core::usize repetitions) const
{
    if (const std::optional<EnergyAbsence> meterAbsence = absenceOfAMeterNotMeasuring(_meter.availability()))
        return std::unexpected{*meterAbsence};
    if (!_counterAtOpening)
        return std::unexpected{EnergyAbsence::ReadFailed};
    if (repetitions == 0u)
        return std::unexpected{EnergyAbsence::NoRepetition};
    if (std::chrono::steady_clock::now() - _openedAt < kMinimumWindow)
        return std::unexpected{EnergyAbsence::ShortWindow};

    const std::optional<core::u64> counterAtClosing = _meter.read();
    if (!counterAtClosing)
        return std::unexpected{EnergyAbsence::ReadFailed};

    const std::optional<core::u64> spent =
        energyDeltaMicrojoules(*_counterAtOpening, *counterAtClosing, _meter.rangeMicrojoules());
    if (!spent)
        return std::unexpected{EnergyAbsence::AmbiguousWrap};
    return static_cast<core::f64>(*spent) / static_cast<core::f64>(repetitions);
}

const EnergyMeter &energyMeter()
{
    static const EnergyMeter meter = EnergyMeter::probe();
    return meter;
}

} // namespace lpl::bench
