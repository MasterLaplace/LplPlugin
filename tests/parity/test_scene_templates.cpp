#include <cstdio>

#include <lpl/ecs/Component.hpp>
#include <lpl/ecs/Partition.hpp>
#include <lpl/ecs/Registry.hpp>
#include <lpl/editor/SceneSerializer.hpp>
#include <lpl/math/FixedPoint.hpp>
#include <lpl/math/Vec3.hpp>

using FVec3 = lpl::math::Vec3<lpl::math::Fixed32>;

static int failures = 0;
static void check(bool ok, const char *what)
{
    std::printf("  %s: %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok)
        ++failures;
}

/**
 * @brief FNV-1a fold of Position + AABB + Mass (all raw Fixed32) over a registry.
 */
static lpl::core::u32 foldState(const lpl::ecs::Registry &registry)
{
    lpl::core::u32 h = 0x811C9DC5u;
    auto step = [&](lpl::core::u32 v) { h = (h ^ v) * 0x01000193u; };
    for (const auto &part : registry.partitions())
    {
        if (!part)
            continue;
        for (const auto &chunk : part->chunks())
        {
            const lpl::core::u32 n = chunk->count();
            const auto *pos = static_cast<const FVec3 *>(chunk->readComponent(lpl::ecs::ComponentId::Position));
            const auto *aabb = static_cast<const FVec3 *>(chunk->readComponent(lpl::ecs::ComponentId::AABB));
            const auto *mass =
                static_cast<const lpl::math::Fixed32 *>(chunk->readComponent(lpl::ecs::ComponentId::Mass));
            for (lpl::core::u32 i = 0; i < n; ++i)
            {
                if (pos)
                {
                    step(static_cast<lpl::core::u32>(pos[i].x.raw()));
                    step(static_cast<lpl::core::u32>(pos[i].y.raw()));
                    step(static_cast<lpl::core::u32>(pos[i].z.raw()));
                }
                if (aabb)
                {
                    step(static_cast<lpl::core::u32>(aabb[i].x.raw()));
                    step(static_cast<lpl::core::u32>(aabb[i].y.raw()));
                    step(static_cast<lpl::core::u32>(aabb[i].z.raw()));
                }
                if (mass)
                    step(static_cast<lpl::core::u32>(mass[i].raw()));
            }
        }
    }
    return h;
}

/**
 * @brief A template reference plus a field override expands to the hand-flattened scene.
 *
 * @details A "cube" prefab: shared AABB half-extents (0.4 = raw 26214) + Mass 1kg (raw 65536). Entity 0 uses
 *          it verbatim; entity 1 overrides Mass to 2kg. The second scene is the hand-flattened equivalent — no
 *          template, everything written out. toLplScene then emits flattened, template-free entities that
 *          round-trip.
 */
static void checkTemplateExpandsToTheHandFlattenedScene()
{
    const char *withTemplate = R"({"format":"lplscene/1",
      "templates":{
        "cube":{"AABB":{"halfExtents":{"x":26214,"y":26214,"z":26214}},
                "Mass":{"kilograms":65536}}},
      "entities":[
        {"$use":"cube","Position":{"value":{"x":98304,"y":0,"z":0}}},
        {"$use":"cube","Position":{"value":{"x":0,"y":0,"z":0}},"Mass":{"kilograms":131072}}
      ]})";

    const char *inlined = R"({"format":"lplscene/1",
      "entities":[
        {"AABB":{"halfExtents":{"x":26214,"y":26214,"z":26214}},"Mass":{"kilograms":65536},
         "Position":{"value":{"x":98304,"y":0,"z":0}}},
        {"AABB":{"halfExtents":{"x":26214,"y":26214,"z":26214}},"Mass":{"kilograms":131072},
         "Position":{"value":{"x":0,"y":0,"z":0}}}
      ]})";

    lpl::ecs::Registry tpl, flat;
    const auto lt = lpl::editor::fromLplScene(withTemplate, tpl);
    const auto lf = lpl::editor::fromLplScene(inlined, flat);

    check(lt.has_value() && lt.value() == 2u, "template scene loads 2 entities");
    check(lf.has_value() && lf.value() == 2u, "inline scene loads 2 entities");
    check(foldState(tpl) == foldState(flat), "template instantiation == hand-flattened scene");

    std::printf("\n  template fold = 0x%08X   inline fold = 0x%08X\n", foldState(tpl), foldState(flat));

    const std::string flatDoc = lpl::editor::toLplScene(tpl);
    lpl::ecs::Registry reload;
    const auto lr = lpl::editor::fromLplScene(flatDoc, reload);
    check(flatDoc.find("$use") == std::string::npos, "serialized document is flattened (no $use)");
    check(lr.has_value() && foldState(reload) == foldState(tpl), "flattened document round-trips");
}

/**
 * @brief Template chains: a template may extend another (§5.2 prefab graph).
 *
 * @details boss inherits Position from itself, Health from elite (200, not base's 50).
 */
static void checkTemplateMayExtendAnother()
{
    const char *chained = R"({"format":"lplscene/1",
          "templates":{
            "base":   {"Health":{"points":50}},
            "elite":  {"$use":"base","Health":{"points":200}},
            "boss":   {"$use":"elite","Position":{"value":{"x":65536,"y":0,"z":0}}}
          },
          "entities":[{"$use":"boss"}]})";
    lpl::ecs::Registry chainWorld;
    const auto loaded = lpl::editor::fromLplScene(chained, chainWorld);
    check(loaded.has_value() && loaded.value() == 1u, "a 3-deep template chain instantiates");

    const std::string flat = lpl::editor::toLplScene(chainWorld);
    check(flat.find("\"points\":200") != std::string::npos, "the most-derived template wins the field");
    check(flat.find("\"points\":50") == std::string::npos, "the base value is overridden, not kept");
    check(flat.find("65536") != std::string::npos, "the leaf template's own component survives");
}

/**
 * @brief A cycle must not hang or crash.
 */
static void checkCycleDoesNotHangOrCrash()
{
    const char *cyclic = R"({"format":"lplscene/1",
          "templates":{
            "a":{"$use":"b","Health":{"points":10}},
            "b":{"$use":"a","Mass":{"kilograms":65536}}
          },
          "entities":[{"$use":"a"}]})";
    lpl::ecs::Registry cycleWorld;
    const auto loaded = lpl::editor::fromLplScene(cyclic, cycleWorld);
    check(loaded.has_value(), "a cyclic template chain is survived, not hung on");
}

/**
 * @brief Prefab-graph proof for the `.lplscene` template system.
 *
 * @details Loads a scene that declares a named template and instantiates it twice via "$use" — once verbatim,
 *          once with a field override — then loads a second, fully-inlined scene meant to be the flattened
 *          equivalent. The authoritative state folds must match, proving that a template reference plus
 *          field-level overrides expand to exactly the same entities as writing them out by hand (the Flakkari
 *          prefab pattern). Also checks that @c toLplScene re-emits a flattened, template-free document that
 *          round-trips.
 *
 *          Host-only. Build via xmake: `xmake run test-scene-templates`.
 */
int main()
{
    std::printf("== .lplscene template (prefab) parity ==\n\n");

    checkTemplateExpandsToTheHandFlattenedScene();
    checkTemplateMayExtendAnother();
    checkCycleDoesNotHangOrCrash();

    std::printf("\n%s (%d failures)\n", failures == 0 ? "ALL PASS" : "FAILURES", failures);
    return failures == 0 ? 0 : 1;
}
