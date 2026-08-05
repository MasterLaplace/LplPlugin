/**
 * @file EditorSession.hpp
 * @brief Headless editing model over an ECS world — the editor's data layer.
 *
 * The engine-side heart of the world editor, deliberately free of any UI: it
 * owns a live @c ecs::Registry and exposes reflection-driven operations an
 * inspector (imgui viewport, a CLI, or the AI bridge) drives — enumerate
 * entities, select one, read/write a component field lane in human units, run
 * procedural-generation commands, and save/load `.lplscene`. Because every edit
 * goes through the same component reflection registry that defines the ECS
 * layout and the on-disk format, the UI needs zero per-component code: it walks
 * @c ecs::schemaOf(id).fields and calls @c getField / @c setField per lane. This
 * is the "JSON is the source of truth, the registry is a view" seam, ready for
 * both hand-authoring and a live imgui editor.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-16
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_EDITOR_EDITORSESSION_HPP
#    define LPL_EDITOR_EDITORSESSION_HPP

#    include <lpl/core/Expected.hpp>
#    include <lpl/core/Types.hpp>
#    include <lpl/ecs/Component.hpp>
#    include <lpl/ecs/ComponentReflection.hpp>
#    include <lpl/ecs/Registry.hpp>
#    include <lpl/editor/CommandJournal.hpp>

#    include <string>
#    include <string_view>

namespace lpl::editor {

/// A resolved handle to one entity slot: its chunk plus its local index.
struct EntityLocation {
    ecs::Chunk *chunk{nullptr};
    core::u32 localIndex{0};
    [[nodiscard]] bool valid() const noexcept { return chunk != nullptr; }
};

/**
 * @class EditorSession
 * @brief Owns a world and exposes reflection-driven, UI-agnostic edits.
 */
class EditorSession {
public:
    EditorSession() = default;

    /// @return The live world (for rendering / stepping by a viewport).
    [[nodiscard]] ecs::Registry &registry() noexcept { return registry_; }
    [[nodiscard]] const ecs::Registry &registry() const noexcept { return registry_; }

    /// @return Total live entity count across every partition.
    [[nodiscard]] core::u32 entityCount() const;

    /// Resolves a flat entity index (iteration order) to its storage slot.
    [[nodiscard]] EntityLocation locate(core::u32 flatIndex) const;

    // ── Selection ────────────────────────────────────────────────────────────
    void select(core::u32 flatIndex) noexcept { selected_ = flatIndex; }
    void clearSelection() noexcept { selected_ = kNoSelection; }
    [[nodiscard]] bool hasSelection() const noexcept { return selected_ != kNoSelection && selected_ < entityCount(); }
    [[nodiscard]] core::u32 selection() const noexcept { return selected_; }

    // ── Reflection-driven field access (human units) ───────────────────────────
    /**
     * @brief Reads one lane of a component field on entity @p flatIndex.
     * @param flatIndex Entity slot (iteration order).
     * @param id        Component to read.
     * @param field     Field name (from the reflection schema).
     * @param lane      Lane index (0 for scalars; 0..2 Vec3; 0..3 Quat).
     * @param out       Receives the value in human units (Fixed32 → float value).
     * @return true if the entity has the component and the field/lane exist.
     */
    [[nodiscard]] bool getField(core::u32 flatIndex, ecs::ComponentId id, std::string_view field, core::u32 lane,
                                double &out) const;

    /// Writes one lane of a component field (Fixed32 fields are quantised).
    [[nodiscard]] bool setField(core::u32 flatIndex, ecs::ComponentId id, std::string_view field, core::u32 lane,
                                double value);

    // ── Scene / procgen passthrough ────────────────────────────────────────────
    /**
     * @brief Runs a JSON command (or batch), recording it in the journal.
     *
     * Every session edit goes through the journal rather than straight to the
     * processor, which is what makes @ref undo possible at all: no command knows
     * how to reverse itself, so undoing is replaying the journal without its last
     * entry. A session that executed and forgot could offer no undo without every
     * command growing an inverse — and the first inverse that is subtly wrong
     * corrupts a world with nothing to catch it.
     */
    [[nodiscard]] core::Expected<std::string> command(std::string_view json);

    /// Rewinds the last recorded command; false when there is nothing to undo.
    bool undo() { return journal_.undo(); }

    /// Re-applies the last undone command; false when there is nothing to redo.
    bool redo() { return journal_.redo(); }

    /// @return Commands currently applied to the world.
    [[nodiscard]] core::u32 historySize() const noexcept { return journal_.size(); }

    /// @return Commands available to @ref redo.
    [[nodiscard]] core::u32 redoSize() const noexcept { return journal_.redoSize(); }

    /**
     * @brief The session's history as a replayable document.
     *
     * The recipe of everything that was done, which replays to a bit-identical
     * world — the same parity slice a `.lplscene` gives, for authoring rather
     * than for the result.
     */
    [[nodiscard]] std::string history() const { return journal_.toJson(); }

    /**
     * @brief The journal itself, for something else that must edit undoably.
     *
     * Exposed for exactly one reason: engine::DemonHost takes a journal, and an
     * intelligence attached to an editor session has to record its acts in the SAME
     * journal the Undo button reads. Handing it a journal of its own would make the
     * demon's edits invisible to undo — which is the property that made routing
     * every act through a journal worth doing in the first place.
     *
     * @return The journal every edit of this session is recorded in.
     */
    [[nodiscard]] CommandJournal &journal() noexcept { return journal_; }

    /// Rebuilds the world from a serialised history, discarding the current one.
    [[nodiscard]] core::Expected<core::u32> replayHistory(std::string_view json) { return journal_.replay(json); }
    /// Serializes the world to a `.lplscene` document.
    [[nodiscard]] std::string save() const;
    /// Loads a `.lplscene` document into the world (appends entities).
    [[nodiscard]] core::Expected<core::u32> load(std::string_view text);

    /// Destroys every entity, emptying the world (keeps the same registry).
    void clear();

    /// Number of lanes a field of @p type exposes (1, 3, or 4).
    [[nodiscard]] static core::u32 laneCount(ecs::FieldType type) noexcept;

private:
    ecs::Registry registry_;
    CommandJournal journal_{registry_};
    static constexpr core::u32 kNoSelection = 0xFFFFFFFFu;
    core::u32 selected_{kNoSelection};
};

} // namespace lpl::editor

#endif // LPL_EDITOR_EDITORSESSION_HPP
