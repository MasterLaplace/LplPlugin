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

#include <cstdio>
#include <string>

namespace {

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

    // Whatever this machine offers, a meter that is not measuring must not produce a value.
    const lpl::bench::EnergyMeter meter = lpl::bench::EnergyMeter::probe();
    lpl::core::u64 reading = 0xDEADBEEFu;
    const bool read = meter.read(reading);
    const bool measuring = meter.availability() == lpl::bench::EnergyAvailability::Measured;
    expect(read == measuring, "a meter reads exactly when it says it is measuring");
    expect(measuring || reading == 0xDEADBEEFu, "a meter that is not measuring leaves the output untouched");
    expect(!meter.description().empty(), "the meter always says where its joules come from, or why there are none");

    std::printf("  energy: %s\n", meter.description().c_str());
    if (failures == 0)
        std::printf("ALL PASS (0 failures, %d checks)\n", checks);
    else
        std::printf("FAILED (%d failures, %d checks)\n", failures, checks);
    return failures == 0 ? 0 : 1;
}
