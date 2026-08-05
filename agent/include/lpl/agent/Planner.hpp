/**
 * @file Planner.hpp
 * @brief What decides the next act.
 *
 * The twelfth header of a module that was sketched with eleven, and it earns the
 * place: this is the seam LplAssistant plugs into. Everything else here describes
 * or validates or executes; this is the one thing that CHOOSES, and a module whose
 * choosing is buried inside its loop cannot have its choosing replaced.
 *
 * A planner sees the capabilities available now, the session so far and what the
 * critics found, and answers with one call — or concludes, meaning "there is nothing
 * left worth doing". It never touches the world: acting is @ref IWorldSurface's job,
 * and keeping the two apart is what lets a planner be swapped for a language model
 * without any of the safety moving.
 *
 * THE SEAM ITSELF IS NOT HERE. It is @ref IDecider in `agent/Decision.hpp`, shared
 * with the demon that runs in ring 0 — because "one is hosted, the other freestanding"
 * was never a reason for two of anything in this project, and for a while there were
 * two look-alike decision interfaces that nothing kept in step.
 *
 * ⚠ What stays here is what genuinely CANNOT cross: a critic's structured findings and
 * the last recipe as a JSON document this planner parses and overlays. Those are not
 * an accident of an API the way `sockaddr` was for `net::Endpoint` — they are unbounded
 * data — so they are held as state given by @ref CorrectionPlanner::observe rather than
 * squeezed into a context a kernel would have to be able to state.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_AGENT_PLANNER_HPP
#    define LPL_LPL_AGENT_PLANNER_HPP

#    include <lpl/agent/Decision.hpp>
#    include <lpl/agent/Observation.hpp>
#    include <lpl/agent/ToolRegistry.hpp>
#    include <lpl/agent/Transcript.hpp>
#    include <lpl/core/Types.hpp>

#    include <string>
#    include <string_view>

namespace lpl::agent {

/**
 * @class IHostedDecider
 * @brief A decider that also sees what only a host can state.
 *
 * An EXTENSION of the shared seam and not a second one, which is the distinction that
 * matters: everything about choosing an act is @ref IDecider, and this adds the two
 * inputs a demon in ring 0 could not be handed even in principle — a critic's
 * structured findings and a JSON recipe to overlay. The same shape as
 * `Config::Builder` gaining host-only knobs: one contract, one place that widens it.
 */
class IHostedDecider : public IDecider {
public:
    /**
     * @brief Hands the decider what @ref DecisionContext cannot carry.
     *
     * Called once per turn, before @ref IDecider::decide. Both arguments must outlive
     * that call.
     *
     * @param findings   What the critics found this turn.
     * @param lastRecipe The recipe last generated from, as JSON; may be empty.
     */
    virtual void observe(const Observations &findings, std::string_view lastRecipe) = 0;
};

/**
 * @class CorrectionPlanner
 * @brief Acts on the first defect a critic named, and concludes when none remain.
 *
 * Deterministic and model-free, which is what makes the correction loop TESTABLE:
 * `test-agent-loop` can assert that a defective world becomes a sound one without
 * any inference in the picture. It is also a real policy, not a mock — a critic
 * that names a defect and suggests a fix has already done the hard part, and
 * following that suggestion is exactly what a competent operator would do.
 *
 * It never reissues the call it just made: a suggestion that did not help is a
 * suggestion to skip, not one to repeat until the budget runs out.
 */
class CorrectionPlanner final : public IHostedDecider {
public:
    /**
     * @brief Hands the planner the two things a shared context cannot carry.
     *
     * Called once per turn, before @ref decide. Both arguments must outlive that
     * call. Passing them here rather than in @ref DecisionContext is the honest
     * placement: a JSON document and a vector of structured findings cannot be stated
     * by a demon in ring 0, and pretending otherwise would put a parser in a kernel.
     *
     * @param findings   What the critics found this turn.
     * @param lastRecipe The recipe last generated from, as JSON; may be empty.
     */
    void observe(const Observations &findings, std::string_view lastRecipe) override;

    /**
     * @brief Chooses the next act.
     *
     * @param context What the decision may depend on.
     * @return An @ref ActKind::Action carrying a tool-call JSON object, or an
     *         @ref ActKind::Answer meaning there is nothing left worth doing.
     */
    [[nodiscard]] Act decide(const DecisionContext &context) noexcept override;

    [[nodiscard]] const char *name() const noexcept override { return "correction"; }

private:
    const Observations *_findings{nullptr};
    std::string _lastRecipe;
};

} // namespace lpl::agent

#endif // LPL_LPL_AGENT_PLANNER_HPP
