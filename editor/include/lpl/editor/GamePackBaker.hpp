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
#    include <lpl/ecology/LivingRecipe.hpp>
#    include <lpl/editor/Json.hpp>
#    include <lpl/pack/GamePack.hpp>
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
 * @brief Reads a bare `procedural` block, without a document around it.
 *
 * A `generate_world` command object IS a procedural block — that is the whole
 * reason the command takes the shape it does, so that anything an editor can build
 * a document can carry. Three callers wanted a recipe out of one and each wrapped
 * it in the minimal document by hand, spelling the format string a third time; a
 * fourth was about to. The wrapping is here now, once.
 *
 * @param proceduralJson The block's JSON object text.
 * @param outRecipe      Filled on success; absent fields keep their defaults.
 * @return An error when the block is malformed.
 */
[[nodiscard]] core::ExpectedVoid parseProceduralBlock(std::string_view proceduralJson,
                                                      procgen::WorldRecipe &outRecipe);

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
 * @brief Bakes a world AND what lives on it.
 *
 * Two sections when @p living is given, one when it is null — and a one-section
 * image is byte-for-byte the one this function produced before living recipes
 * existed, which is what keeps every cartridge already baked valid.
 *
 * @param recipe The world.
 * @param living The ecosystem, or nullptr for a world with nothing declared on it.
 * @return The packed image.
 */
[[nodiscard]] std::vector<core::u8> bakeGamePack(const procgen::WorldRecipe &recipe,
                                                 const ecology::LivingRecipe *living);

/**
 * @brief Reads the optional "living" block of a scene object.
 * @param scene   Parsed scene (or document root, for the flat form).
 * @param outLiving Receives the recipe when the block is present.
 * @return true when the document declared one.
 */
[[nodiscard]] bool parseSceneLiving(const detail::JVal &scene, ecology::LivingRecipe &outLiving);

/**
 * @brief Emits a living recipe as the JSON a `.lplscene` carries.
 * @param living The recipe.
 * @return Its "living" object, every field written out.
 */
[[nodiscard]] std::string emitSceneLiving(const ecology::LivingRecipe &living);

/**
 * @brief Bakes a world, what lives on it, AND what it looks like.
 *
 * Three sections, two, or one — a null argument omits its section entirely, and a
 * one-section image is byte-for-byte what this produced before either extension
 * existed. That is the property that makes sections, rather than a grown recipe
 * struct, the way this format grows.
 *
 * @param recipe The world.
 * @param living The ecosystem, or nullptr.
 * @param view   The look, or nullptr for a world that keeps the host's.
 * @return The packed image.
 */
[[nodiscard]] std::vector<core::u8> bakeGamePack(const procgen::WorldRecipe &recipe,
                                                 const ecology::LivingRecipe *living, const pack::ViewV1 *view);

/**
 * @brief Reads the optional "view" block of a scene object.
 *
 * Wire form rather than engine form on purpose: translating a sky needs render
 * types, and this module bakes bytes. engine::toEngineView does that half.
 *
 * @param scene   Parsed scene (or document root, for the flat form).
 * @param outView Receives the profile when the block is present.
 * @return true when the document declared one.
 */
[[nodiscard]] bool parseSceneView(const detail::JVal &scene, pack::ViewV1 &outView);

/**
 * @brief Emits a view profile as the JSON a `.lplscene` carries.
 * @param view The wire profile.
 * @return Its "view" object, every field written out.
 */
[[nodiscard]] std::string emitSceneView(const pack::ViewV1 &view);

/**
 * @brief Convenience: parse a `.lplscene` document and bake it in one step.
 * @param document The `.lplscene` text.
 * @return The pack image, or the parse error.
 */
[[nodiscard]] core::Expected<std::vector<core::u8>> bakeSceneDocument(std::string_view document);

} // namespace lpl::editor

#endif // LPL_EDITOR_GAMEPACKBAKER_HPP
