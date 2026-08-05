/**
 * @file DemonHost.hpp
 * @brief Hosting an intelligence inside the engine loop.
 *
 * The counterpart of World: a World is what runs, a DemonHost is who watches and
 * acts on it. Present only where a model is available, absent everywhere else, so
 * the engine keeps booting on a machine that has no weights.
 *
 * This is where the reason-act-observe loop lives, and NOT in agent/Dialogue.hpp —
 * a dialogue is how the demon addresses the sovereign, which is a different thing
 * from how it works. The host owns the parts and turns the crank:
 *
 *   decider  chooses     (agent::IDecider — the seam a model plugs into, shared
 *                         with the demon in ring 0 rather than mirrored by it)
 *   dispatcher acts      (agent::Dispatcher — through the journal, so undoable)
 *   critics  observe     (agent::inspectWorld / reviewGeneration)
 *   transcript remembers (agent::Transcript)
 *
 * The dependency runs engine → agent, never the reverse: a tool surface over a
 * world has no business knowing about a game loop, and the cycle would be real.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_ENGINE_DEMONHOST_HPP
#    define LPL_LPL_ENGINE_DEMONHOST_HPP

#    include <lpl/agent/Dialogue.hpp>
#    include <lpl/agent/Dispatcher.hpp>
#    include <lpl/agent/Planner.hpp>
#    include <lpl/agent/Transcript.hpp>
#    include <lpl/core/Types.hpp>
#    include <lpl/engine/InferenceBudget.hpp>

#    include <string>

namespace lpl::ecs {
class Registry;
}

namespace lpl::editor {
class CommandJournal;
}

namespace lpl::engine {

/**
 * @enum ThinkOutcome
 * @brief Why a stretch of thinking ended.
 *
 * Four outcomes, not two, because "it stopped" is not a diagnosis. A caller has
 * to be able to tell a demon that finished from one that gave up.
 */
enum class ThinkOutcome : core::u8 {
    Concluded = 0,    ///< The planner had nothing left worth doing.
    BudgetExhausted,  ///< It ran out of turns with work outstanding.
    Stuck,            ///< It repeated itself and the world stopped answering differently.
    NoLegalMove       ///< Every fix the critics suggested is currently ungated.
};

/**
 * @struct ThinkResult
 * @brief What a stretch of thinking produced.
 */
struct ThinkResult {
    ThinkOutcome outcome{ThinkOutcome::Concluded};
    core::u32 turns{0u};             ///< Turns actually taken.
    core::u32 acts{0u};              ///< Calls that were accepted.
    core::u32 refusals{0u};          ///< Calls the validator or the world rejected.
    core::u32 defectsBefore{0u};     ///< Defects the critics saw on entry.
    core::u32 defectsAfter{0u};      ///< Defects still standing on exit.

    /**
     * @brief Whether the world ends up sound: no defect the critics can name.
     * @return True if the world is sound, false otherwise.
     */
    [[nodiscard]] bool sound() const noexcept { return defectsAfter == 0u; }
};

/**
 * @class DemonHost
 * @brief An intelligence attached to a world.
 *
 * Non-owning: the registry and its journal belong to whoever built them (an
 * Engine, a Server, a test, the `lpl-demon` entry point).
 */
class DemonHost {
public:
    /**
     * @param registry World to watch and act on; must outlive this.
     * @param journal  Its command journal, so every act is undoable and replayable.
     * @param decider  What chooses; must outlive this.
     */
    DemonHost(ecs::Registry &registry, editor::CommandJournal &journal, agent::IHostedDecider &decider);

    /**
     * @brief Runs the loop until it concludes or @p budget runs out.
     */
    [[nodiscard]] ThinkResult think(const InferenceBudget &budget);

    /**
     * @brief Takes what the sovereign asked for as the standing goal.
     *
     * The goal is passed to the planner and otherwise left alone: interpreting a
     * sentence is a model's job, and a host that tried to parse it would be
     * putting a second, worse language model in the way of the real one.
     */
    void consider(const agent::Intent &intent);

    /**
     * @brief The channel to the sovereign.
     * @return The dialogue the demon uses to receive intents and send replies.
     */
    [[nodiscard]] agent::Dialogue &openDialogue() noexcept { return _dialogue; }

    [[nodiscard]] const agent::Transcript &transcript() const noexcept { return _transcript; }
    [[nodiscard]] const std::string &goal() const noexcept { return _goal; }

    /**
     * @brief What the critics say about the world as it stands.
     * @return The observations made by the critics.
     */
    [[nodiscard]] agent::Observations observe() const;

private:
    ecs::Registry &_registry;
    editor::CommandJournal &_journal;
    agent::IHostedDecider &_decider;
    agent::Dispatcher _dispatcher;
    agent::Transcript _transcript;
    agent::Dialogue _dialogue;
    std::string _goal;

    /// The last `generate_world` call and its report, for the asked-for-vs-got critics.
    std::string _lastRecipe;
    std::string _lastGenerationReport;
};

} // namespace lpl::engine

#endif // LPL_LPL_ENGINE_DEMONHOST_HPP
