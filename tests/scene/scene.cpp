#include <lpl/scene/Scene.hpp>
#include <lpl/testing/Test.hpp>

LPL_TEST_SUITE(scene);

namespace {

[[nodiscard]] lpl::core::i32 raw(lpl::core::i32 units) { return lpl::scene::Fixed32::fromInt(units).raw(); }

[[nodiscard]] lpl::scene::Transform2D translation(lpl::core::i32 x, lpl::core::i32 y)
{
    return lpl::scene::Transform2D::translation(lpl::scene::Fixed32::fromInt(x), lpl::scene::Fixed32::fromInt(y));
}

} // namespace

/**
 * @brief Gate P4 scene, composition: a child translated by (5, 0) under a parent at (10, 20) is at (15, 20).
 */
LPL_TEST(children_compose_their_parents_transform)
{
    lpl::scene::Scene scene;
    const lpl::scene::NodeId root = scene.createNode();
    const lpl::scene::NodeId child = scene.createNode(root);

    scene.setLocalTransform(root, translation(10, 20));
    scene.setLocalTransform(child, translation(5, 0));

    const lpl::scene::Transform2D world = scene.worldTransform(child);
    lpl::scene::Fixed32 x{lpl::scene::Fixed32::fromInt(0)};
    lpl::scene::Fixed32 y{lpl::scene::Fixed32::fromInt(0)};

    world.apply(lpl::scene::Fixed32::fromInt(0), lpl::scene::Fixed32::fromInt(0), x, y);
    test.check(world.tx.raw() == raw(15) && world.ty.raw() == raw(20), "the child's world translation is (15, 20)");
    test.check(x.raw() == raw(15) && y.raw() == raw(20), "and it carries the origin there");
}

LPL_TEST(undo_and_redo_walk_the_edits)
{
    lpl::scene::Scene scene;
    const lpl::scene::NodeId root = scene.createNode();
    const lpl::scene::NodeId child = scene.createNode(root);

    scene.setLocalTransform(root, translation(10, 20));
    scene.setLocalTransform(child, translation(5, 0));
    scene.setLocalTransform(child, translation(7, 7));

    test.check(scene.localTransform(child).tx.raw() == raw(7), "the last edit sets the child's x to 7");
    test.check(scene.undoDepth() == 3u && scene.redoDepth() == 0u, "three edits can be undone");
    test.check(scene.undo() && scene.localTransform(child).tx.raw() == raw(5), "undo brings x back to 5");
    test.check(scene.redo() && scene.localTransform(child).tx.raw() == raw(7), "redo sets it to 7 again");

    scene.undo();
    scene.setLocalTransform(child, translation(9, 9));
    test.check(scene.redoDepth() == 0u, "an edit after an undo drops what could be redone");
}

LPL_TEST(selection_ignores_a_duplicate)
{
    lpl::scene::Scene scene;
    const lpl::scene::NodeId root = scene.createNode();
    const lpl::scene::NodeId child = scene.createNode(root);

    scene.select(root);
    scene.select(child);
    scene.select(root);
    test.check(scene.selectionCount() == 2u && scene.isSelected(root) && scene.isSelected(child),
               "selecting the root twice selects it once");

    scene.deselect(root);
    test.check(scene.selectionCount() == 1u && !scene.isSelected(root) && scene.isSelected(child),
               "deselecting it leaves the child");

    scene.clearSelection();
    test.check(scene.selectionCount() == 0u, "clearing empties the selection");
}

/**
 * @brief Gate P4 scene, rotation: a quarter turn built through CORDIC carries (1, 0) to about
 *        (0, 1), with the same words on both targets.
 */
LPL_TEST(quarter_turn_carries_x_onto_y)
{
    const lpl::scene::Transform2D turn =
        lpl::scene::Transform2D::fromTRS(lpl::scene::Fixed32::fromInt(0), lpl::scene::Fixed32::fromInt(0),
                                         lpl::scene::Fixed32::fromFloat(1.57079632679f),
                                         lpl::scene::Fixed32::fromInt(1), lpl::scene::Fixed32::fromInt(1));
    lpl::scene::Fixed32 x{lpl::scene::Fixed32::fromInt(0)};
    lpl::scene::Fixed32 y{lpl::scene::Fixed32::fromInt(0)};

    turn.apply(lpl::scene::Fixed32::fromInt(1), lpl::scene::Fixed32::fromInt(0), x, y);
    test.check(x.raw() > -512 && x.raw() < 512, "x lands within 1/128 of 0");
    test.check(y.raw() > 65024 && y.raw() < 66048, "and y within 1/128 of 1");

    test.measure("turned_x", x.raw());
    test.measure("turned_y", y.raw());
}
