/**
 * @file Scene.hpp
 * @brief A 2D scene graph: node tree, world transforms, multi-select, undo/redo.
 *
 * Nodes form a parent-linked tree; world transforms compose Fixed32 affines from
 * the root, so the whole scene is deterministic (Fixed32/CORDIC authority). The
 * editing model is command-based: every local-transform edit pushes a reversible
 * record onto an undo stack, giving multi-step undo/redo.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-06-28
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_SCENE_SCENE_HPP
#    define LPL_SCENE_SCENE_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/scene/Transform2D.hpp>
#    include <lpl/std/vector.hpp>

namespace lpl::scene {

using NodeId = core::u32;
inline constexpr NodeId kInvalidNode = 0xFFFFFFFFu;

/** @brief One node: a local transform plus its parent link. */
struct Node {
    Transform2D local{};
    NodeId parent{kInvalidNode};
};

/**
 * @class Scene
 * @brief Parent-linked 2D scene graph with editing (selection + undo/redo).
 */
class Scene {
public:
    /** @brief Create a node under @p parent (kInvalidNode for a root). */
    [[nodiscard]] NodeId createNode(NodeId parent = kInvalidNode);

    [[nodiscard]] core::u32 nodeCount() const noexcept { return static_cast<core::u32>(_nodes.size()); }
    [[nodiscard]] bool isValid(NodeId id) const noexcept { return id < _nodes.size(); }
    [[nodiscard]] const Node &node(NodeId id) const noexcept { return _nodes[id]; }
    [[nodiscard]] NodeId parentOf(NodeId id) const noexcept { return _nodes[id].parent; }
    [[nodiscard]] const Transform2D &localTransform(NodeId id) const noexcept { return _nodes[id].local; }

    /** @brief Compose the world transform by walking from the root to @p id. */
    [[nodiscard]] Transform2D worldTransform(NodeId id) const;

    // --- Editing (undo/redo recorded) -------------------------------------

    /** @brief Set a node's local transform, recording it for undo. */
    void setLocalTransform(NodeId id, const Transform2D &transform);

    /** @brief Undo the last transform edit; false if nothing to undo. */
    bool undo();

    /** @brief Redo the last undone edit; false if nothing to redo. */
    bool redo();

    [[nodiscard]] core::u32 undoDepth() const noexcept { return static_cast<core::u32>(_undo.size()); }
    [[nodiscard]] core::u32 redoDepth() const noexcept { return static_cast<core::u32>(_redo.size()); }

    // --- Multi-selection --------------------------------------------------

    void select(NodeId id);
    void deselect(NodeId id);
    void clearSelection() noexcept { _selection.clear(); }
    [[nodiscard]] bool isSelected(NodeId id) const noexcept;
    [[nodiscard]] core::u32 selectionCount() const noexcept { return static_cast<core::u32>(_selection.size()); }

private:
    struct TransformEdit {
        NodeId node;
        Transform2D before;
        Transform2D after;
    };

    pmr::vector<Node> _nodes;
    pmr::vector<NodeId> _selection;
    pmr::vector<TransformEdit> _undo;
    pmr::vector<TransformEdit> _redo;
};

} // namespace lpl::scene

#endif // LPL_SCENE_SCENE_HPP
