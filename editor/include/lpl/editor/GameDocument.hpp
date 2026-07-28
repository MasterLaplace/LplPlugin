/**
 * @file GameDocument.hpp
 * @brief A whole game as data — Flakkari's four-stage document, modernised.
 *
 * The research report (§5.2) asks for Flakkari's structure kept and its defects
 * fixed: metadata, named resources, an ORDERED system list, and templates plus
 * instances, all in one JSON document. Until now `.lplscene` carried only the
 * last stage (templates + entities) plus a procedural recipe. This is the rest.
 *
 * Two properties drive the design:
 *
 * **One file, two readers.** A server never loads pixels and a client never
 * needs the instance cap, but splitting them into two files means two things to
 * keep in sync and two ways to ship half a game. So every resource declares who
 * needs it (@ref ResourceScope) and each consumer asks for its own slice. It is
 * also what makes "owning the game" and "owning the server" the same thing.
 *
 * **A scene is a Registry.** Flakkari's `Game::loadScene` built a fresh registry
 * per scene, which made switching scenes a swap rather than a teardown. Same
 * here: @ref instantiateScene fills a registry the caller owns, so hosting N
 * scenes is having N registries and nothing else.
 *
 * Layering note: the document stores the game profile by NAME, as text, and
 * never includes `lpl/engine`. Resolving that name into a real preset is the
 * app's job (`engine::parseGameProfile`). An editor that depended on the engine
 * facade would invert the dependency graph — editor is tooling over ecs/procgen,
 * not a consumer of the top-level engine.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_EDITOR_GAMEDOCUMENT_HPP
#    define LPL_EDITOR_GAMEDOCUMENT_HPP

#    include <lpl/core/Expected.hpp>
#    include <lpl/core/Types.hpp>
#    include <lpl/procgen/WorldRecipe.hpp>

#    include <string>
#    include <string_view>
#    include <vector>

namespace lpl::ecs {
class Registry;
}

namespace lpl::editor {

/**
 * @enum ResourceScope
 * @brief Who actually needs a resource.
 *
 * A headless server can skip every texture without the document being wrong for
 * a client reading the same bytes.
 */
enum class ResourceScope : core::u8 {
    Shared = 0, ///< Both sides need it (a collision mesh, a table of stats).
    Client,     ///< Presentation only: textures, fonts, sounds, shaders.
    Server      ///< Authority only: tuning a client has no business reading.
};

/**
 * @enum ResourceKind
 * @brief What a resource is, for the loader that will eventually fetch it.
 */
enum class ResourceKind : core::u8 {
    Other = 0,
    Texture,
    Font,
    Sound,
    Music,
    Shader,
    Data
};

/**
 * @struct ResourceEntry
 * @brief A logical name bound to a path.
 *
 * Components reference the NAME, never the path — Flakkari's indirection, kept
 * because it is what lets a server ignore the asset entirely while still
 * understanding the component that mentions it.
 */
struct ResourceEntry {
    std::string name;                           ///< Logical name used by components.
    std::string path;                           ///< Where the bytes live.
    ResourceKind kind{ResourceKind::Other};     ///< Asset class.
    ResourceScope scope{ResourceScope::Shared}; ///< Who loads it.
};

/**
 * @struct SceneDescription
 * @brief One named scene: its systems, its resources, and how to fill it.
 */
struct SceneDescription {
    std::string name;                     ///< Scene identifier, referenced by startScene.
    std::vector<std::string> systems;     ///< ORDERED list of system names to enable.
    std::vector<ResourceEntry> resources; ///< Named assets this scene uses.

    bool hasRecipe{false};         ///< Does a procedural pass build this scene?
    procgen::WorldRecipe recipe{}; ///< The pass, when hasRecipe.

    std::string templatesJson; ///< Prefab table, as JSON text ("{}" when absent).
    std::string entitiesJson;  ///< Explicit instances, as JSON text ("[]" when absent).
};

/**
 * @struct GameMetadata
 * @brief The game-level stage: what Flakkari put at the root of config.cfg.
 */
struct GameMetadata {
    std::string title;          ///< Human-readable name.
    std::string version;        ///< Game version (distinct from the format version).
    std::string profile;        ///< Netcode preset NAME, e.g. "mmorpg" (see the layering note).
    std::string startScene;     ///< Scene to load first; empty means the first one.
    bool online{false};         ///< Does this game expect a server?
    core::u32 minPlayers{1u};   ///< Lower player bound.
    core::u32 maxPlayers{1u};   ///< Upper player bound.
    core::u32 maxInstances{1u}; ///< How many worlds a server may host of it.
};

/**
 * @struct GameDocument
 * @brief The parsed four-stage document.
 */
struct GameDocument {
    GameMetadata metadata{};              ///< Game-level stage.
    std::vector<SceneDescription> scenes; ///< Named scenes, in declaration order.

    /**
     * @brief Looks up a scene by name.
     * @param name Scene identifier.
     * @return The scene, or nullptr when absent.
     */
    [[nodiscard]] const SceneDescription *findScene(std::string_view name) const;

    /**
     * @brief The scene a fresh load should start from.
     * @return `metadata.startScene` when it resolves, else the first scene, else nullptr.
     */
    [[nodiscard]] const SceneDescription *startScene() const;
};

/**
 * @brief Parses a `.lplscene` document, in either shape.
 *
 * A document with a `"scenes"` array is the full four-stage form. One without is
 * treated as a single implicit scene built from the root's own
 * templates/entities/procedural blocks — which is exactly the shape every
 * existing document has, so they keep loading unchanged.
 *
 * @param text The document.
 * @return The parsed document, or a parse error.
 */
[[nodiscard]] core::Expected<GameDocument> parseGameDocument(std::string_view text);

/**
 * @brief Serialises a document back to the four-stage form.
 * @param document The document to write.
 * @return JSON text that @ref parseGameDocument reads back equivalently.
 */
[[nodiscard]] std::string emitGameDocument(const GameDocument &document);

/**
 * @brief Fills @p registry from @p scene.
 *
 * Order is part of the contract, because it fixes entity creation order and
 * therefore the world's fold: the procedural pass runs FIRST and lays down the
 * world, then the explicit entities are added on top of it. A scene that placed
 * its hand-authored entities before generating terrain would fold differently on
 * a reader that did it the other way round.
 *
 * @param scene    Scene to instantiate.
 * @param registry Destination world (should be empty on entry).
 * @return Total entities created, or an error.
 */
[[nodiscard]] core::Expected<core::u32> instantiateScene(const SceneDescription &scene, ecs::Registry &registry);

/**
 * @brief The resources a given consumer has to load.
 * @param scene    Scene whose resource table to filter.
 * @param consumer Who is asking (Client or Server; Shared always matches).
 * @return Pointers into @p scene's table, in declaration order.
 */
[[nodiscard]] std::vector<const ResourceEntry *> resourcesFor(const SceneDescription &scene, ResourceScope consumer);

/// @brief Text form of a resource kind, as it appears in the document.
[[nodiscard]] std::string_view resourceKindName(ResourceKind kind) noexcept;

/// @brief Text form of a resource scope, as it appears in the document.
[[nodiscard]] std::string_view resourceScopeName(ResourceScope scope) noexcept;

} // namespace lpl::editor

#endif // LPL_EDITOR_GAMEDOCUMENT_HPP
