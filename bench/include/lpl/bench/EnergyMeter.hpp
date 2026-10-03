/**
 * @file EnergyMeter.hpp
 * @brief Joules spent by a benchmark, read from the package energy counter, or an
 *        honest statement of why they cannot be.
 *
 * A benchmark that reports only time measures the one resource that is not the goal.
 * The package energy counter (RAPL, exposed by Linux through powercap) gives joules;
 * two readings around the timed loop and a subtraction give the energy of the run.
 *
 * Four outcomes, never a zero standing in for one of them. The counter may be readable;
 * it may exist and be denied to this process, since reading it unprivileged was
 * restricted after the PLATYPUS power side channel; it may exist and fail to read, or hold
 * no usable value, named with the file at fault; or there may be nothing to read, which
 * is the case under WSL2, where Hyper-V does not pass RAPL through to the Linux guest.
 * Each calls for a different remedy, so they are kept apart.
 *
 * What the number means, stated so it is not over-read: it is the energy of the WHOLE
 * package over the run, idle draw and every other process included, divided by the
 * repetitions. It ranks two implementations measured back to back on a quiet machine;
 * it is not the marginal cost of the work alone.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-09-24
 * @copyright MIT License
 */
#pragma once

#ifndef LPL_BENCH_ENERGYMETER_HPP
#    define LPL_BENCH_ENERGYMETER_HPP

#    include <lpl/core/Types.hpp>

#    include <chrono>
#    include <optional>
#    include <string>
#    include <string_view>

namespace lpl::bench {

/** What an energy reading can honestly claim. */
enum class EnergyAvailability : core::u8 {
    Measured,   ///< The package counter is readable.
    Denied,     ///< The counter exists and this process may not read it.
    Unreadable, ///< A package zone exists and one of its files could not be read or holds no usable value.
    Absent,     ///< There is no counter this build knows how to read.
};

/**
 * @brief Microjoules spent between two readings of a counter that wraps.
 *
 * @details The counter restarts from zero once it reaches @p rangeMicrojoules, so a
 *          reading lower than the previous one means exactly one wrap. More than one wrap
 *          between two readings cannot be seen from the values alone; at the power a
 *          package draws, the range lasts hours.
 *
 * @param before            Reading at the start of the window.
 * @param after             Reading at the end.
 * @param rangeMicrojoules  Value at which the counter wraps.
 * @return The energy spent, or nothing when the readings cannot describe a single wrap.
 */
[[nodiscard]] std::optional<core::u64> energyDeltaMicrojoules(core::u64 before, core::u64 after,
                                                              core::u64 rangeMicrojoules) noexcept;

/**
 * @brief Formats microjoules with an auto-scaled unit, e.g. "12.35 uJ", "4.201 mJ".
 * @param microjoules Energy.
 * @return Human-readable energy string.
 */
[[nodiscard]] std::string formatEnergy(core::f64 microjoules);

/** @brief One package energy counter, located once. */
class EnergyMeter final {
public:
    /** Where Linux exposes its powercap zones. */
    static constexpr std::string_view kPowercapRoot = "/sys/class/powercap";

    /**
     * @brief Looks for a readable package counter.
     * @param powercapRoot Directory holding the @c intel-rapl:N zones.
     * @return The first readable package zone; otherwise the first denied one, then the
     *         first unreadable one; otherwise a meter that says why there is none.
     */
    [[nodiscard]] static EnergyMeter probe(std::string_view powercapRoot = kPowercapRoot);

    /**
     * @brief What this meter can claim.
     * @return The availability.
     */
    [[nodiscard]] EnergyAvailability availability() const noexcept { return _availability; }

    /**
     * @brief Where the joules come from, or why there are none, in one line.
     * @return The description.
     */
    [[nodiscard]] const std::string &description() const noexcept { return _description; }

    /**
     * @brief The value at which the counter wraps.
     * @return The range in microjoules, zero when not measured.
     */
    [[nodiscard]] core::u64 rangeMicrojoules() const noexcept { return _range; }

    /**
     * @brief Reads the counter.
     * @return The cumulative energy in microjoules, or nothing when the meter is not
     *         measuring or the read failed.
     */
    [[nodiscard]] std::optional<core::u64> read() const;

private:
    explicit EnergyMeter(EnergyAvailability availability, std::string description, std::string counterPath = {},
                         core::u64 range = 0u);

    [[nodiscard]] static EnergyMeter probeZone(std::string_view powercapRoot, int zone);
    [[nodiscard]] static EnergyMeter unreadable(const std::string &path, std::string_view reason);

    EnergyAvailability _availability;
    std::string _description;
    std::string _counterPath;
    core::u64 _range;
};

/**
 * @brief The package energy spent across one window of repetitions, shared among them.
 *
 * @details The window brackets every repetition at once, because the counter advances
 *          about once a millisecond in steps of tens of microjoules, far coarser than one
 *          repetition. A window shorter than @ref kMinimumWindow reports nothing: one step
 *          of the counter would weigh more than a percent of it.
 */
class EnergyBracket final {
public:
    /** Shortest window whose energy is reported. */
    static constexpr std::chrono::milliseconds kMinimumWindow{100};

    /**
     * @brief Opens the window: reads the counter and starts the clock.
     * @param meter Counter to read; it must outlive the bracket.
     */
    explicit EnergyBracket(const EnergyMeter &meter);

    /**
     * @brief Shares the energy spent since the bracket opened among @p repetitions; each
     *        call reads the counter and the clock again.
     * @param repetitions How many repetitions ran since the bracket opened.
     * @return Microjoules per repetition, or nothing when the meter is not measuring, a read
     *         failed, the readings cannot describe a single wrap, no repetition ran, or less
     *         than @ref kMinimumWindow has passed since the bracket opened.
     */
    [[nodiscard]] std::optional<core::f64> microjoulesPerRepetition(core::usize repetitions) const;

private:
    const EnergyMeter &_meter;
    std::chrono::steady_clock::time_point _openedAt;
    std::optional<core::u64> _counterAtOpening;
};

/**
 * @brief The process-wide meter, probed on first use.
 * @return The meter.
 */
[[nodiscard]] const EnergyMeter &energyMeter();

} // namespace lpl::bench

#endif // LPL_BENCH_ENERGYMETER_HPP
