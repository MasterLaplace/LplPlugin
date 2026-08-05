/**
 * @file Constraint.hpp
 * @brief How a fact binds the simulation.
 *
 * A fact says 'this settlement exists in 1204'. A constraint says what the
 * simulation must do about it: seed it, force it, or merely score it. Keeping the
 * three separate is what stops the historical record from silently becoming a
 * scripted animation.
 *
 * The distinction is the whole moral of the module. A record that is entirely FORCED
 * proves nothing — it is an animation of the record, and its divergence from the
 * record is zero by construction. A record entirely SCORED proves the most, and
 * usually reconstructs nothing. Seeding is the honest middle: put the initial
 * conditions in, let the simulation run, and measure.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_HISTORY_CONSTRAINT_HPP
#    define LPL_LPL_HISTORY_CONSTRAINT_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/history/Fact.hpp>

namespace lpl::history {

/**
 * @enum ConstraintKind
 * @brief What the simulation does about a fact.
 */
enum class ConstraintKind : core::u32 {
    /**
     * @brief Establish it once, then let the world take it from there.
     *
     * The initial condition. A city that existed in 1204 is placed in 1204 and is
     * afterwards as free to grow or be abandoned as any other.
     */
    Seed = 0u,

    /**
     * @brief Hold it true for its whole window, overriding the simulation.
     *
     * Spend these sparingly. Every forced fact is one the run did not have to explain,
     * and a timeline made of them measures nothing.
     */
    Force = 1u,

    /**
     * @brief Do nothing; compare against it at the end.
     *
     * The only kind that can falsify anything. A scored fact is a prediction the run
     * had to earn.
     */
    Score = 2u,

    Count = 3u
};

/**
 * @brief The word a document spells @p kind with.
 * @param kind The kind.
 * @return Its name.
 */
[[nodiscard]] constexpr const char *constraintKindName(ConstraintKind kind) noexcept
{
    switch (kind)
    {
    case ConstraintKind::Seed: return "seed";
    case ConstraintKind::Force: return "force";
    case ConstraintKind::Score: return "score";
    case ConstraintKind::Count: break;
    }
    return "score";
}

/**
 * @struct Constraint
 * @brief A fact, plus what the run must do about it.
 */
struct Constraint {
    Fact fact{};                                ///< What is claimed.
    ConstraintKind kind{ConstraintKind::Score}; ///< What to do about it.
    math::Fixed32 confidence{};                 ///< After fusion; may exceed the fact's own sigma.
};

} // namespace lpl::history

#endif // LPL_LPL_HISTORY_CONSTRAINT_HPP
