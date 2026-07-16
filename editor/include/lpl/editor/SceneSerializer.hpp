/**
 * @file SceneSerializer.hpp
 * @brief Reflection-driven (de)serialization of an ECS world to `.lplscene`.
 *
 * A `.lplscene` document is the data-driven, human-editable representation of a
 * scene — the single source of truth an editor (or the AI bridge) manipulates,
 * with the live @c ecs::Registry as its view. Every component is (de)serialized
 * through the @c ecs::ComponentReflection registry, so there is no hand-written
 * per-component code here: the same declaration that drives the ECS layout also
 * drives the on-disk format. Authoritative Fixed32 fields are emitted as raw
 * integers (deterministic, exact round-trip).
 *
 * Format (v1). An optional @c "templates" object declares named prefabs; an
 * entity references one with @c "$use" and overrides fields on top of it (a
 * flattened prefab graph, the Flakkari pattern). @c toLplScene always emits
 * flattened, template-free entities.
 * @code
 * { "format": "lplscene/1",
 *   "templates": {
 *     "cube": { "AABB": {"halfExtents":{"x":26214,"y":26214,"z":26214}},
 *               "Mass": {"value":65536} } },
 *   "entities": [
 *     { "$use": "cube", "Position": {"value":{"x":98304,"y":0,"z":0}} },
 *     { "$use": "cube", "Position": {"value":{"x":0,"y":0,"z":0}},
 *       "Mass": {"value":131072} }
 *   ] }
 * @endcode
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-16
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_EDITOR_SCENESERIALIZER_HPP
#    define LPL_EDITOR_SCENESERIALIZER_HPP

#    include <lpl/core/Expected.hpp>
#    include <lpl/core/Types.hpp>

#    include <string>
#    include <string_view>

namespace lpl::ecs {
class Registry;
}

namespace lpl::editor {

/**
 * @brief Serializes every live entity of @p registry to a `.lplscene` document.
 * @param registry The world to serialize.
 * @return The `.lplscene` JSON text.
 */
[[nodiscard]] std::string toLplScene(const ecs::Registry &registry);

/**
 * @brief Parses a `.lplscene` document and instantiates its entities into
 *        @p registry (which should be empty for a clean load).
 * @param text     The `.lplscene` JSON text.
 * @param registry Destination world.
 * @return The number of entities created, or an error on malformed input.
 */
[[nodiscard]] core::Expected<core::u32> fromLplScene(std::string_view text, ecs::Registry &registry);

} // namespace lpl::editor

#endif // LPL_EDITOR_SCENESERIALIZER_HPP
