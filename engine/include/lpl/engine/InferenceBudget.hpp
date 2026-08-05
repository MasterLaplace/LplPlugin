/**
 * @file InferenceBudget.hpp
 * @brief Where the server profile's spare capacity goes.
 *
 * A headless server renders nothing, so the frame budget that a client spends on
 * rasterising is free. This declares it as an explicit, bounded resource for
 * inference — bounded because the tick contract still comes first: the demon may
 * think, but never at the price of a missed deadline.
 *
 * ── Counted in TURNS, not in milliseconds ────────────────────────────────────
 *
 * An earlier sketch spelled this `InferenceBudget::ofFrame(0.6f)` — sixty per cent
 * of the frame. That spelling is deliberately NOT kept, and the reason is the
 * determinism contract rather than taste: a budget read off a wall clock makes the
 * number of acts depend on how fast the machine was that day, so the same session
 * replays differently on a slower host and no test can pin it. Every budget in
 * this project that sits in a replayable path is a COUNT.
 *
 * What is lost is real and worth naming: a turn is not a fixed amount of work, so
 * a turn budget does not guarantee a deadline. A host that must not overrun should
 * spend its budget across frames — a few turns per tick — rather than ask this to
 * be a stopwatch.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_ENGINE_INFERENCEBUDGET_HPP
#    define LPL_LPL_ENGINE_INFERENCEBUDGET_HPP

#    include <lpl/core/Types.hpp>

namespace lpl::engine {

/**
 * @class InferenceBudget
 * @brief How much a demon may think before it must stop.
 */
class InferenceBudget {
public:
    /**
     * @brief A budget of @p turns reason-act-observe steps.
     * @param turns The number of turns to allocate.
     * @return An InferenceBudget instance with the specified turn count.
     */
    [[nodiscard]] static InferenceBudget ofTurns(core::u32 turns) noexcept;

    /**
     * @brief The default: enough to correct a handful of defects, never unbounded.
     * @return An InferenceBudget instance with the standard turn count.
     */
    [[nodiscard]] static InferenceBudget standard() noexcept { return ofTurns(8u); }

    /**
     * @brief A budget that permits nothing, for a host with no capacity to spare.
     * @return An InferenceBudget instance with zero turns.
     */
    [[nodiscard]] static InferenceBudget none() noexcept { return ofTurns(0u); }

    [[nodiscard]] core::u32 turns() const noexcept { return _turns; }
    [[nodiscard]] bool exhausted() const noexcept { return _turns == 0u; }

    /**
     * @brief The turn count at which a conclusion becomes mandatory.
     *
     * The last tenth of the budget, and never less than one turn: a session that
     * ran out mid-thought leaves a world half-corrected and says nothing about
     * why. Reserving the tail for a conclusion is what makes the outcome legible.
     */
    [[nodiscard]] core::u32 concludeAfter() const noexcept;

private:
    explicit InferenceBudget(core::u32 turns) noexcept : _turns(turns) {}

    core::u32 _turns{0u};
};

} // namespace lpl::engine

#endif // LPL_LPL_ENGINE_INFERENCEBUDGET_HPP
