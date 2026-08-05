/**
 * @file Observation.hpp
 * @brief What the world answers.
 *
 * The half every design forgets. Stats, filtered queries, verdicts — always
 * bounded, always flagged when truncated, because an unbounded dump is precisely
 * what destroys a context window.
 *
 * ── Why these critics are deterministic, and not a vision model ───────────────
 *
 * The founding report proposes the loop the other way round: capture a frame,
 * hand it to a small vision-language model, and let it say "the bridge floats two
 * metres above the ground". That is the right design WHEN YOU DO NOT HAVE THE
 * SCENE. We have it. Positions, extents, counts and the generator's own report
 * are all in memory, so asking a model to GUESS what we can COMPUTE would be less
 * reliable and, worse, non-deterministic — a defect that appears on one run and
 * not the next teaches a correction loop nothing.
 *
 * So the defect channel is a set of critics, each seeded from a defect this
 * repository actually shipped: props detached from the ground they stand on, a
 * scatter that collapsed to a point because a raw Fixed32 constructor was read as
 * a value, extents that make an entity uncollidable, coordinates saturated past
 * what Q16.16 can hold, and a generator that was asked for rivers and produced
 * none.
 *
 * The screenshot stays (see Vision.hpp), for the two things a critic cannot do:
 * find the defect nobody has named yet, and let a human look.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_AGENT_OBSERVATION_HPP
#    define LPL_LPL_AGENT_OBSERVATION_HPP

#    include <lpl/core/Types.hpp>

#    include <string>
#    include <string_view>
#    include <vector>

namespace lpl::ecs {
class Registry;
}

namespace lpl::agent {

/**
 * @enum Severity
 * @brief How much a finding should change what happens next.
 */
enum class Severity : core::u8 {
    Info = 0, ///< Worth knowing, not worth acting on.
    Warning,  ///< Probably wrong; a caller may reasonably ignore it.
    Defect    ///< The world is not what was asked for. Fix or explain.
};

/**
 * @struct Finding
 * @brief One named defect, and what to do about it.
 *
 * @c suggestedTool and @c suggestedArgs are what make this a correction loop
 * rather than a complaint: a critic that says what is wrong without saying what
 * would fix it puts the whole burden of diagnosis back on the caller.
 */
struct Finding {
    Severity severity{Severity::Info};
    std::string code;    ///< Stable short identifier, e.g. "no-rivers".
    std::string message; ///< One sentence, actionable, in the caller's terms.
    std::string suggestedTool;
    std::string suggestedArgs; ///< JSON object to merge into the next call; may be empty.
};

/**
 * @struct Observations
 * @brief A bounded set of findings.
 */
struct Observations {
    std::vector<Finding> findings;
    core::u32 total{0u};   ///< How many were found, before the bound.
    bool truncated{false}; ///< Whether @c findings is shorter than @c total.

    /// How many findings are @c Severity::Defect.
    [[nodiscard]] core::u32 defects() const noexcept;

    /// The findings as a JSON report, in the shape the command surface uses.
    [[nodiscard]] std::string toJson() const;
};

/**
 * @brief Critics that read the world itself.
 *
 * Sees only what an @c ecs::Registry holds, which is why it cannot check
 * connectivity or biome coverage: those live on grids the generator keeps, not on
 * entities. @ref reviewGeneration covers that half by reading the generator's own
 * report instead of re-deriving it — a second derivation of a number procgen
 * already computed is exactly the duplication this project keeps paying for.
 *
 * @param registry World to inspect.
 * @param limit    How many findings to return; the rest are counted, not carried.
 */
[[nodiscard]] Observations inspectWorld(const ecs::Registry &registry, core::u32 limit = 8u);

/**
 * @brief Critics that compare what was asked for with what came back.
 *
 * @param recipeJson The `generate_world` command object that was issued.
 * @param reportJson The report it answered with.
 * @param limit      How many findings to return.
 */
[[nodiscard]] Observations reviewGeneration(std::string_view recipeJson, std::string_view reportJson,
                                            core::u32 limit = 8u);

} // namespace lpl::agent

#endif // LPL_LPL_AGENT_OBSERVATION_HPP
