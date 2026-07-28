/**
 * @file CommandProcessor.hpp
 * @brief JSON command interface driving an ECS world + procedural generators.
 *
 * The single entry point an editor UI — or the Caine AI bridge — uses to mutate
 * a scene. A command is a small JSON object @c {"cmd":"...", ...args}; a batch is
 * a JSON array of them. Every command is data-driven and exception-free, and the
 * authoritative results are Fixed32, so a recorded command stream replays to a
 * bit-identical world (the AI issues a few hundred tokens of intent; the
 * deterministic engine does the work). This is the seam the GBNF grammar will
 * constrain the model to.
 *
 * Supported commands:
 *  - @c generate_world  {seed,width,depth,cellSize,terrain{},erosion{},rivers{},climate{},
 *                        biomes{},caves{},settlement{},gate{},scatter[]}
 *                        — the command object IS a `.lplscene` "procedural" block,
 *                          read by the same parser, so an editor session and a
 *                          document cannot describe different worlds
 *  - @c load_scene      {scene:"<.lplscene text>"}
 *  - @c save_scene      {}                       → report carries the document
 *  - @c count           {}                       → report carries the entity count
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-16
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_EDITOR_COMMANDPROCESSOR_HPP
#    define LPL_EDITOR_COMMANDPROCESSOR_HPP

#    include <lpl/core/Expected.hpp>
#    include <lpl/core/Types.hpp>

#    include <string>
#    include <string_view>

namespace lpl::ecs {
class Registry;
}

namespace lpl::editor {

/**
 * @class CommandProcessor
 * @brief Executes JSON commands against a live @c ecs::Registry.
 */
class CommandProcessor {
public:
    explicit CommandProcessor(ecs::Registry &registry) : registry_(registry) {}

    /**
     * @brief Executes one command object or a batch array of them.
     * @param json A JSON object @c {"cmd":...} or an array of such objects.
     * @return A JSON report string (per-command results), or an error.
     */
    [[nodiscard]] core::Expected<std::string> execute(std::string_view json);

private:
    ecs::Registry &registry_;
};

/// Number of live entities across every partition of @p registry.
[[nodiscard]] core::u32 entityCount(const ecs::Registry &registry);

/**
 * @brief Destroys every live entity in @p registry.
 *
 * The primitive behind the `clear_world` command and behind journal replay,
 * which rebuilds a world from an empty one. Collects the ids before destroying:
 * destroying while iterating chunks mutates what is being iterated.
 *
 * @param registry World to empty.
 * @return Number of entities destroyed.
 */
core::u32 destroyAllEntities(ecs::Registry &registry);

} // namespace lpl::editor

#endif // LPL_EDITOR_COMMANDPROCESSOR_HPP
