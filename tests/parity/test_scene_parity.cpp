/**
 * @file test_scene_parity.cpp
 * @brief Parity test: deterministic 2D scene graph (transforms + undo/redo).
 *
 * Verifies Fixed32 affine composition, parent/child world transforms, multi-
 * selection and the undo/redo command stack — all integer/Fixed32 so identical
 * across the Linux oracle and the freestanding kernel.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-06-28
 * @copyright MIT License
 */

#include <cstdio>
#include <lpl/scene/Scene.hpp>

using namespace lpl;
using scene::Fixed32;

static int failures = 0;

static void check(bool ok, const char *what)
{
    std::printf("  %s: %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok)
        ++failures;
}

static bool eqRaw(Fixed32 a, core::i32 raw) { return a.raw() == raw; }

int main()
{
    std::printf("== scene transforms ==\n");

    scene::Scene s;
    const scene::NodeId root = s.createNode();
    const scene::NodeId child = s.createNode(root);

    // Root translated by (10, 20); child translated by (5, 0) in root space.
    s.setLocalTransform(root, scene::Transform2D::translation(Fixed32::fromInt(10), Fixed32::fromInt(20)));
    s.setLocalTransform(child, scene::Transform2D::translation(Fixed32::fromInt(5), Fixed32::fromInt(0)));

    const scene::Transform2D world = s.worldTransform(child);
    check(eqRaw(world.tx, Fixed32::fromInt(15).raw()) && eqRaw(world.ty, Fixed32::fromInt(20).raw()),
          "child world translation = (15, 20)");

    // Apply the world transform to the origin.
    Fixed32 px{Fixed32::fromInt(0)};
    Fixed32 py{Fixed32::fromInt(0)};
    world.apply(Fixed32::fromInt(0), Fixed32::fromInt(0), px, py);
    check(eqRaw(px, Fixed32::fromInt(15).raw()) && eqRaw(py, Fixed32::fromInt(20).raw()),
          "world.apply(origin) = (15, 20)");

    std::printf("== undo / redo ==\n");

    // Two more edits to the child, then walk the stack.
    s.setLocalTransform(child, scene::Transform2D::translation(Fixed32::fromInt(7), Fixed32::fromInt(7)));
    check(eqRaw(s.localTransform(child).tx, Fixed32::fromInt(7).raw()), "edit set child tx = 7");
    check(s.undoDepth() == 3u && s.redoDepth() == 0u, "undo depth 3 after 3 edits");

    check(s.undo() && eqRaw(s.localTransform(child).tx, Fixed32::fromInt(5).raw()), "undo restores child tx = 5");
    check(s.redo() && eqRaw(s.localTransform(child).tx, Fixed32::fromInt(7).raw()), "redo reapplies child tx = 7");

    // A fresh edit clears the redo history.
    s.undo();
    s.setLocalTransform(child, scene::Transform2D::translation(Fixed32::fromInt(9), Fixed32::fromInt(9)));
    check(s.redoDepth() == 0u, "fresh edit clears redo history");

    std::printf("== multi-selection ==\n");
    s.select(root);
    s.select(child);
    s.select(root); // duplicate ignored
    check(s.selectionCount() == 2u && s.isSelected(root) && s.isSelected(child), "select root+child (no dup)");
    s.deselect(root);
    check(s.selectionCount() == 1u && !s.isSelected(root) && s.isSelected(child), "deselect root");
    s.clearSelection();
    check(s.selectionCount() == 0u, "clearSelection empties");

    std::printf("== rotation determinism ==\n");
    // 90-degree rotation (pi/2) of point (1,0) -> ~ (0,1); check tx/ty exact raw.
    const Fixed32 halfPi = Fixed32::fromFloat(1.57079632679f);
    const scene::Transform2D rot = scene::Transform2D::fromTRS(Fixed32::fromInt(0), Fixed32::fromInt(0), halfPi,
                                                               Fixed32::fromInt(1), Fixed32::fromInt(1));
    Fixed32 rx{Fixed32::fromInt(0)};
    Fixed32 ry{Fixed32::fromInt(0)};
    rot.apply(Fixed32::fromInt(1), Fixed32::fromInt(0), rx, ry);
    std::printf("  rot(1,0) = raw(%d, %d)\n", rx.raw(), ry.raw());
    // x' ~ 0, y' ~ 1 (Q16.16: 65536). Allow CORDIC quantisation slack.
    check(rx.raw() > -512 && rx.raw() < 512 && ry.raw() > 65024 && ry.raw() < 66048,
          "fromTRS 90deg maps (1,0) -> ~(0,1)");

    std::printf("%s (%d failure%s)\n", failures == 0 ? "ALL PASS" : "FAILURES", failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
