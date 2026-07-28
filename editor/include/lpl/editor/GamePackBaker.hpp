/**
 * @file GamePackBaker.hpp
 * @brief Host-side oven: turns an authored `.lplscene` into a `.lplpak` image.
 *
 * This is the seam between the two encodings of a game. The JSON document is
 * what humans, the editor and the AI write, what git versions and what a diff
 * can be read on. The pack is what a target loads. Keeping the oven here — in
 * the host tooling module that already owns the JSON parser and std::string —
 * is what lets lpl::pack stay freestanding: nothing in the reader ever needs to
 * know that JSON exists.
 *
 * The `"procedural"` block of a `.lplscene` is the recipe: a seed plus the
 * passes that expand it. Baking copies it into the pack rather than expanding
 * it, so the client and the server each rebuild the same world from the same
 * few hundred bytes instead of shipping the entities themselves.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_EDITOR_GAMEPACKBAKER_HPP
#    define LPL_EDITOR_GAMEPACKBAKER_HPP

#    include <lpl/core/Expected.hpp>
#    include <lpl/core/Types.hpp>
#    include <lpl/procgen/WorldRecipe.hpp>

#    include <string>
#    include <string_view>
#    include <vector>

namespace lpl::editor {

/**
 * @brief Reads the `"procedural"` block of a `.lplscene` document.
 *
 * Absent fields keep the default from lpl::procgen::WorldRecipe, so a document
 * only has to state what it changes.
 *
 * @param document  The `.lplscene` text.
 * @param outRecipe Filled on success.
 * @return An error when the document is malformed or carries no
 *         `"procedural"` block.
 */
[[nodiscard]] core::ExpectedVoid parseSceneRecipe(std::string_view document, procgen::WorldRecipe &outRecipe);

/**
 * @brief Emits a `"procedural"` block for @p recipe, as it appears in a scene.
 * @param recipe The recipe to serialise.
 * @return The JSON object text, without a trailing comma or newline.
 */
[[nodiscard]] std::string emitSceneRecipe(const procgen::WorldRecipe &recipe);

/**
 * @brief Bakes a recipe into a complete `.lplpak` image.
 * @param recipe The recipe to write as the pack's WorldRecipe section.
 * @return The full byte image, ready to write to disk or hand to a target.
 */
[[nodiscard]] std::vector<core::u8> bakeGamePack(const procgen::WorldRecipe &recipe);

/**
 * @brief Convenience: parse a `.lplscene` document and bake it in one step.
 * @param document The `.lplscene` text.
 * @return The pack image, or the parse error.
 */
[[nodiscard]] core::Expected<std::vector<core::u8>> bakeSceneDocument(std::string_view document);

} // namespace lpl::editor

#endif // LPL_EDITOR_GAMEPACKBAKER_HPP
