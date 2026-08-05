/**
 * @file Chronicle.hpp
 * @brief What the simulation actually did.
 *
 * The symmetric object of the timeline: emitted by the run, not fed into it. A
 * chronicle is the demon's own account of its world, and it is comparable to the
 * record because both are sextuplets.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_HISTORY_CHRONICLE_HPP
#    define LPL_LPL_HISTORY_CHRONICLE_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/history/Attestation.hpp>
#    include <lpl/history/Fact.hpp>
#    include <lpl/std/vector.hpp>

namespace lpl::history {

/**
 * @struct Event
 * @brief One thing the run says happened, and why.
 */
struct Event {
    Fact fact{};               ///< The same shape as a record's claim, deliberately.
    Attestation attestation{}; ///< Why it is here.
};

/**
 * @class Chronicle
 * @brief The run's own account of its world.
 */
class Chronicle {
public:
    Chronicle() = default;

    /**
     * @brief Records an event.
     * @param fact        What happened.
     * @param attestation Why.
     */
    void record(const Fact &fact, const Attestation &attestation);

    /**
     * @brief Events recorded.
     * @return The count.
     */
    [[nodiscard]] core::u32 size() const noexcept { return static_cast<core::u32>(_events.size()); }

    /**
     * @brief One event.
     * @param index Position, in emission order.
     * @return The event; a default one when out of range.
     */
    [[nodiscard]] const Event &at(core::u32 index) const noexcept;

    /**
     * @brief Events attributed to a given cause.
     * @param cause Which.
     * @return How many.
     */
    [[nodiscard]] core::u32 countByCause(Cause cause) const noexcept;

    /**
     * @brief FNV-1a over every event, in emission order.
     * @param seed Fold seed.
     * @return The signature.
     */
    [[nodiscard]] core::u32 fold(core::u32 seed) const noexcept;

private:
    lpl::pmr::vector<Event> _events{};
};

} // namespace lpl::history

#endif // LPL_LPL_HISTORY_CHRONICLE_HPP
