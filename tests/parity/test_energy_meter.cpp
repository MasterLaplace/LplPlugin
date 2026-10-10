/**
 * @file test_energy_meter.cpp
 * @brief The energy column of a benchmark: a counter that wraps, and a meter that must
 *        never report a number it did not read.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-09-24
 * @copyright MIT License
 */

#include <lpl/bench/EnergyMeter.hpp>
#include <lpl/core/NonCopyable.hpp>

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <expected>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <thread>

namespace {

using lpl::bench::EnergyAvailability;
using lpl::bench::EnergyBracket;
using lpl::bench::EnergyMeter;

int checks = 0;
int failures = 0;

void expect(bool condition, const char *what)
{
    ++checks;
    if (condition)
        return;
    ++failures;
    std::printf("  FAIL  %s\n", what);
}

bool reportsNoEnergyBecause(const std::expected<lpl::core::f64, lpl::bench::EnergyAbsence> &energy,
                            lpl::bench::EnergyAbsence reason)
{
    return !energy.has_value() && energy.error() == reason;
}

class FakePowercap final : public lpl::core::NonCopyable<FakePowercap> {
public:
    explicit FakePowercap(const std::string &scenario)
        : _root{std::filesystem::temp_directory_path() /
                ("lpl-fake-powercap-" + scenario + "-" +
                 std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))}
    {
    }

    ~FakePowercap()
    {
        std::error_code ignored;
        std::filesystem::remove_all(_root, ignored);
    }

    void write(const std::string &relativePath, const std::string &content) const
    {
        const std::filesystem::path path = _root / relativePath;
        std::filesystem::create_directories(path.parent_path());
        std::ofstream{path} << content;
    }

    void writePackageZone(const std::string &counter, const std::string &range) const
    {
        write("intel-rapl:0/name", "package-0\n");
        write("intel-rapl:0/energy_uj", counter);
        write("intel-rapl:0/max_energy_range_uj", range);
    }

    void remove(const std::string &relativePath) const { std::filesystem::remove(_root / relativePath); }

    [[nodiscard]] std::string root() const { return _root.string(); }

private:
    std::filesystem::path _root;
};

void expectAReadablePackageZoneToBeMeasured()
{
    const FakePowercap powercap{"measured"};
    powercap.writePackageZone("900\n", "1000\n");

    const EnergyMeter meter = EnergyMeter::probe(powercap.root());
    expect(meter.availability() == EnergyAvailability::Measured, "a readable package zone is measured");
    expect(meter.description() == "package-0 (intel-rapl:0), whole package",
           "a measured meter names the zone it reads");
    expect(meter.read() == 900u, "a measured meter reads the counter");
    expect(meter.rangeMicrojoules() == 1000u, "a measured meter knows where the counter wraps");
}

void expectAWindowAcrossOneWrapToShareTheUnwrappedEnergy()
{
    const FakePowercap powercap{"wrap"};
    powercap.writePackageZone("900\n", "1000\n");
    const EnergyMeter meter = EnergyMeter::probe(powercap.root());

    const EnergyBracket bracket{meter};
    powercap.write("intel-rapl:0/energy_uj", "50\n");
    std::this_thread::sleep_for(EnergyBracket::kMinimumWindow);

    expect(bracket.microjoulesPerRepetition(10u) == 15.0,
           "a window across one wrap shares the unwrapped energy among its repetitions");
    expect(reportsNoEnergyBecause(bracket.microjoulesPerRepetition(0u), lpl::bench::EnergyAbsence::NoRepetition),
           "a window with no repetition reports no energy, and says so");
}

void expectAShortWindowToReportNoEnergy()
{
    const FakePowercap powercap{"short"};
    powercap.writePackageZone("900\n", "1000\n");
    const EnergyMeter meter = EnergyMeter::probe(powercap.root());

    const EnergyBracket bracket{meter};
    powercap.write("intel-rapl:0/energy_uj", "950\n");

    expect(reportsNoEnergyBecause(bracket.microjoulesPerRepetition(10u), lpl::bench::EnergyAbsence::ShortWindow),
           "a window shorter than the minimum reports no energy, rather than one counter step, and says so");
}

void expectACounterThatVanishesToReportAFailedRead()
{
    const FakePowercap atOpening{"vanished-at-opening"};
    atOpening.writePackageZone("900\n", "1000\n");
    const EnergyMeter openingMeter = EnergyMeter::probe(atOpening.root());
    atOpening.remove("intel-rapl:0/energy_uj");
    const EnergyBracket openingBracket{openingMeter};
    std::this_thread::sleep_for(EnergyBracket::kMinimumWindow);
    expect(reportsNoEnergyBecause(openingBracket.microjoulesPerRepetition(10u), lpl::bench::EnergyAbsence::ReadFailed),
           "a counter that cannot be read when the window opens reports a failed read");

    const FakePowercap atClosing{"vanished-at-closing"};
    atClosing.writePackageZone("900\n", "1000\n");
    const EnergyMeter closingMeter = EnergyMeter::probe(atClosing.root());
    const EnergyBracket closingBracket{closingMeter};
    atClosing.remove("intel-rapl:0/energy_uj");
    std::this_thread::sleep_for(EnergyBracket::kMinimumWindow);
    expect(reportsNoEnergyBecause(closingBracket.microjoulesPerRepetition(10u), lpl::bench::EnergyAbsence::ReadFailed),
           "a counter that cannot be read when the window closes reports a failed read");
}

void expectReadingsBeyondTheRangeToReportAnAmbiguousWrap()
{
    const FakePowercap powercap{"beyond-range"};
    powercap.writePackageZone("1200\n", "1000\n");
    const EnergyMeter meter = EnergyMeter::probe(powercap.root());

    const EnergyBracket bracket{meter};
    powercap.write("intel-rapl:0/energy_uj", "50\n");
    std::this_thread::sleep_for(EnergyBracket::kMinimumWindow);

    expect(reportsNoEnergyBecause(bracket.microjoulesPerRepetition(10u), lpl::bench::EnergyAbsence::AmbiguousWrap),
           "readings that cannot describe one wrap report an ambiguous wrap, not a number");
}

void expectAMeterThatIsNotMeasuringToGiveItsOwnReason()
{
    const FakePowercap missingCounter{"bracket-unreadable"};
    missingCounter.write("intel-rapl:0/name", "package-0\n");
    const EnergyMeter unreadableMeter = EnergyMeter::probe(missingCounter.root());
    const EnergyBracket unreadableBracket{unreadableMeter};

    const FakePowercap noPackage{"bracket-absent"};
    noPackage.write("intel-rapl:0/name", "core\n");
    const EnergyMeter absentMeter = EnergyMeter::probe(noPackage.root());
    const EnergyBracket absentBracket{absentMeter};

    std::this_thread::sleep_for(EnergyBracket::kMinimumWindow);
    expect(
        reportsNoEnergyBecause(unreadableBracket.microjoulesPerRepetition(10u), lpl::bench::EnergyAbsence::Unreadable),
        "a bracket on an unreadable meter gives the meter's reason, not a failed read");
    expect(reportsNoEnergyBecause(absentBracket.microjoulesPerRepetition(10u), lpl::bench::EnergyAbsence::Absent),
           "a bracket on an absent meter gives the meter's reason, even over a long window");
    expect(reportsNoEnergyBecause(absentBracket.microjoulesPerRepetition(0u), lpl::bench::EnergyAbsence::Absent),
           "a meter that is not measuring is the first reason given, before the repetitions");
}

void expectEveryReasonToHaveItsOwnName()
{
    const std::string_view names[] = {
        lpl::bench::energyAbsenceName(lpl::bench::EnergyAbsence::Denied),
        lpl::bench::energyAbsenceName(lpl::bench::EnergyAbsence::Unreadable),
        lpl::bench::energyAbsenceName(lpl::bench::EnergyAbsence::Absent),
        lpl::bench::energyAbsenceName(lpl::bench::EnergyAbsence::NoRepetition),
        lpl::bench::energyAbsenceName(lpl::bench::EnergyAbsence::ShortWindow),
        lpl::bench::energyAbsenceName(lpl::bench::EnergyAbsence::ReadFailed),
        lpl::bench::energyAbsenceName(lpl::bench::EnergyAbsence::AmbiguousWrap),
    };
    const std::string_view expected[] = {"denied",       "unreadable",  "absent",        "no-repetition",
                                         "short-window", "read-failed", "ambiguous-wrap"};
    for (std::size_t i = 0; i < std::size(expected); ++i)
        expect(names[i] == expected[i], "each reason has the stable name a JSON row writes");
}

void expectAPackageZoneThatCannotBeReadToSayWhy()
{
    const FakePowercap missingCounter{"missing-counter"};
    missingCounter.write("intel-rapl:0/name", "package-0\n");
    const EnergyMeter withoutCounter = EnergyMeter::probe(missingCounter.root());
    expect(withoutCounter.availability() == EnergyAvailability::Unreadable,
           "a package zone without a readable counter is unreadable, not absent");
    expect(withoutCounter.description() ==
               "unreadable: " + missingCounter.root() + "/intel-rapl:0/energy_uj: " + std::strerror(ENOENT),
           "an unreadable meter names the file that failed and why");
    expect(!withoutCounter.read().has_value(), "an unreadable meter reads nothing");

    const FakePowercap zeroRange{"zero-range"};
    zeroRange.writePackageZone("5\n", "0\n");
    const EnergyMeter withZeroRange = EnergyMeter::probe(zeroRange.root());
    expect(withZeroRange.availability() == EnergyAvailability::Unreadable,
           "a package zone whose counter range is zero is unreadable, not absent");
    expect(withZeroRange.description().starts_with("unreadable: " + zeroRange.root() +
                                                   "/intel-rapl:0/max_energy_range_uj: "),
           "a zero range names the range file");
}

void expectATreeWithoutAPackageZoneToBeAbsent()
{
    const FakePowercap powercap{"no-package"};
    powercap.write("intel-rapl:0/name", "core\n");

    const EnergyMeter meter = EnergyMeter::probe(powercap.root());
    expect(meter.availability() == EnergyAvailability::Absent, "a tree without a package zone is absent");
    expect(meter.description() == "absent: no powercap package zone under " + powercap.root(),
           "an absent meter names the tree it looked in");
}

} // namespace

int main()
{
    using lpl::bench::energyDeltaMicrojoules;

    expect(energyDeltaMicrojoules(100u, 250u, 1000u) == 150u, "a counter that moved forward gives the difference");
    expect(energyDeltaMicrojoules(500u, 500u, 1000u) == 0u, "a counter that did not move spent nothing");

    // The case a plain subtraction gets wrong: on unsigned it wraps to an enormous number
    // that looks exactly like a measurement.
    expect(energyDeltaMicrojoules(900u, 50u, 1000u) == 150u, "one wrap is folded back into the range");

    expect(!energyDeltaMicrojoules(1200u, 50u, 1000u).has_value(),
           "a start reading above the range cannot describe one wrap, so no number");
    expect(!energyDeltaMicrojoules(900u, 50u, 0u).has_value(), "an unknown range cannot unwrap, so no number");

    expect(lpl::bench::formatEnergy(12.35) == "12.35 uJ", "microjoules stay microjoules");
    expect(lpl::bench::formatEnergy(4201.0) == "4.201 mJ", "thousands become millijoules");
    expect(lpl::bench::formatEnergy(2.5e6) == "2.5 J", "millions become joules");

    expectAReadablePackageZoneToBeMeasured();
    expectAWindowAcrossOneWrapToShareTheUnwrappedEnergy();
    expectAShortWindowToReportNoEnergy();
    expectACounterThatVanishesToReportAFailedRead();
    expectReadingsBeyondTheRangeToReportAnAmbiguousWrap();
    expectAMeterThatIsNotMeasuringToGiveItsOwnReason();
    expectEveryReasonToHaveItsOwnName();
    expectAPackageZoneThatCannotBeReadToSayWhy();
    expectATreeWithoutAPackageZoneToBeAbsent();

    // Whatever this machine offers, a meter that is not measuring must not produce a value.
    const EnergyMeter meter = EnergyMeter::probe();
    const bool measuring = meter.availability() == EnergyAvailability::Measured;
    expect(meter.read().has_value() == measuring, "a meter reads exactly when it says it is measuring");
    expect(!meter.description().empty(), "the meter always says where its joules come from, or why there are none");

    std::printf("  energy: %s\n", meter.description().c_str());
    if (failures == 0)
        std::printf("ALL PASS (0 failures, %d checks)\n", checks);
    else
        std::printf("FAILED (%d failures, %d checks)\n", failures, checks);
    return failures == 0 ? 0 : 1;
}
