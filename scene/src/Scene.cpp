/**
 * @file Scene.cpp
 * @brief Scene-graph world composition, undo/redo and selection.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-06-28
 * @copyright MIT License
 */
#include <lpl/scene/Scene.hpp>

namespace lpl::scene {

NodeId Scene::createNode(NodeId parent)
{
    const NodeId id = static_cast<NodeId>(_nodes.size());
    Node n;
    n.parent = parent;
    _nodes.push_back(n);
    return id;
}

Transform2D Scene::worldTransform(NodeId id) const
{
    if (!isValid(id))
        return Transform2D::identity();

    // Collect the chain root..id (parents first), then compose downward.
    pmr::vector<NodeId> chain;
    for (NodeId cur = id; cur != kInvalidNode; cur = _nodes[cur].parent)
        chain.push_back(cur);

    Transform2D world = Transform2D::identity();
    for (core::usize i = chain.size(); i-- > 0u;)
        world = world * _nodes[chain[i]].local;
    return world;
}

void Scene::setLocalTransform(NodeId id, const Transform2D &transform)
{
    if (!isValid(id))
        return;

    TransformEdit edit{id, _nodes[id].local, transform};
    _nodes[id].local = transform;
    _undo.push_back(edit);
    _redo.clear(); // a fresh edit invalidates the redo history
}

bool Scene::undo()
{
    if (_undo.empty())
        return false;
    const TransformEdit edit = _undo[_undo.size() - 1u];
    _undo.pop_back();
    _nodes[edit.node].local = edit.before;
    _redo.push_back(edit);
    return true;
}

bool Scene::redo()
{
    if (_redo.empty())
        return false;
    const TransformEdit edit = _redo[_redo.size() - 1u];
    _redo.pop_back();
    _nodes[edit.node].local = edit.after;
    _undo.push_back(edit);
    return true;
}

void Scene::select(NodeId id)
{
    if (!isValid(id) || isSelected(id))
        return;
    _selection.push_back(id);
}

void Scene::deselect(NodeId id)
{
    for (core::usize i = 0u; i < _selection.size(); ++i)
        if (_selection[i] == id)
        {
            _selection[i] = _selection[_selection.size() - 1u];
            _selection.pop_back();
            return;
        }
}

bool Scene::isSelected(NodeId id) const noexcept
{
    for (core::usize i = 0u; i < _selection.size(); ++i)
        if (_selection[i] == id)
            return true;
    return false;
}

} // namespace lpl::scene
