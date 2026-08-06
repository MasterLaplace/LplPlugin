/**
 * @file test_agent_tools.cpp
 * @brief The tool surface, its schema and its grammar agree.
 *
 * Three artefacts derived from one registry must stay consistent; a call the
 * schema accepts must be a call the grammar can emit and the dispatcher can run.
 *
 * The assertions that matter here are the ones that can fail for a real reason:
 *  1. every declared tool is a command the engine actually accepts — the table
 *     cannot advertise a capability that does not exist;
 *  2. the offered set NARROWS on an empty world, and the grammar narrows with it,
 *     which is the whole claim behind regenerating it every step;
 *  3. a call outside the declaration is refused BEFORE it touches the world, with
 *     a reason, for each of the six ways a call can be wrong;
 *  4. an act of the agent is undoable, because it went through the journal;
 *  5. the surface signature is stable, and moves when the surface moves.
 *
 * @author MasterLaplace
 * @copyright MIT License
 */

#include <lpl/agent/Dispatcher.hpp>
#include <lpl/agent/Grammar.hpp>
#include <lpl/agent/Observation.hpp>
#include <lpl/agent/Parity.hpp>
#include <lpl/agent/Schema.hpp>
#include <lpl/agent/Tool.hpp>
#include <lpl/agent/ToolCall.hpp>
#include <lpl/agent/ToolRegistry.hpp>
#include <lpl/agent/Vision.hpp>
#include <lpl/ecs/ComponentReflection.hpp>
#include <lpl/ecs/Registry.hpp>
#include <lpl/editor/CommandJournal.hpp>
#include <lpl/editor/CommandProcessor.hpp>
#include <lpl/editor/Json.hpp>
#include <lpl/image/Image.hpp>
#include <lpl/procgen/WorldRecipe.hpp>

#include <cstdio>
#include <string>

using namespace lpl;

static int failures = 0;

static void check(bool ok, const std::string &what)
{
    std::printf("  %s: %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok)
        ++failures;
}

/// A world with entities in it, so the RequiresWorld gate is satisfied.
static void seedWorld(ecs::Registry &registry)
{
    procgen::WorldRecipe recipe = procgen::parityWorldRecipe();
    (void) procgen::bakeWorld(registry, recipe);
}

int main()
{
    std::printf("== agent tool surface ==\n\n");

    // ── 1. Every declared tool is a command the engine accepts ────────────────
    // CommandProcessor's dispatch lives in an anonymous namespace, so this
    // agreement cannot be checked at compile time. It is checked here instead,
    // by asking the processor and refusing an "unknown command" reply. A tool
    // named in the table but absent from the processor fails right here.
    std::printf("-- every tool is a real command --\n");
    {
        ecs::Registry probe;
        editor::CommandProcessor processor{probe};
        for (const agent::ToolDesc &tool : agent::kTools)
        {
            if (tool.host != agent::ToolHost::Journal)
                continue; // Agent-hosted; checked below, against its own handler.
            const std::string command = "{\"cmd\":\"" + std::string{tool.name} + "\"}";
            const auto report = processor.execute(command);
            const bool known = report.has_value() && report.value().find("unknown command") == std::string::npos;
            check(known, std::string{tool.name} + " is dispatchable");
        }
    }

    // ── 2. The offered set is state-dependent, and so is the grammar ──────────
    std::printf("\n-- gating narrows the surface --\n");
    ecs::Registry empty;
    ecs::Registry full;
    seedWorld(full);

    const agent::ToolRegistry onEmpty = agent::ToolRegistry::forWorld(empty);
    const agent::ToolRegistry onFull = agent::ToolRegistry::forWorld(full);

    check(onEmpty.state().entityCount == 0u, "an empty world reports no entities");
    check(onFull.state().entityCount != 0u,
          "a generated world reports " + std::to_string(onFull.state().entityCount) + " entities");
    check(onEmpty.size() < onFull.size(), "fewer tools on an empty world (" + std::to_string(onEmpty.size()) + " < " +
                                              std::to_string(onFull.size()) + ")");
    check(!onEmpty.offers("spawn_from_template"), "spawn_from_template is NOT offered before a world exists");
    check(onFull.offers("spawn_from_template"), "spawn_from_template IS offered once one does");
    check(onEmpty.offers("generate_world"), "generate_world is always offered");

    const std::string gEmpty = agent::emitGbnf(onEmpty);
    const std::string gFull = agent::emitGbnf(onFull);
    // The claim of DWG-010: what the model cannot see, it cannot call. Not
    // "is refused afterwards" — cannot be produced by the sampler at all.
    check(gEmpty.find("spawn_from_template") == std::string::npos,
          "the empty-world grammar cannot even spell spawn_from_template");
    check(gFull.find("spawn_from_template") != std::string::npos, "the populated grammar can");
    check(gEmpty != gFull, "the grammar is regenerated, not cached");

    // The closed set of component names comes from the reflection registry, so a
    // new component widens the grammar with no edit to the tool table.
    for (const ecs::ComponentSchema &schema : ecs::allSchemas())
        if (schema.name == "Position" || schema.name == "Health")
            check(gFull.find(std::string{"\\\""} + std::string{schema.name} + "\\\"") != std::string::npos,
                  std::string{"the grammar carries the component name "} + std::string{schema.name});

    // The second closed set, and the reason DWG-010 asks for sets in the grammar at
    // all: the underground generators are WORDS, so a director cannot even generate a
    // kind that does not exist. Before the recipe could name them, three of the four
    // were reachable only by hand-written builder calls — a world nothing could save,
    // bake or replay. "auto" is in the set too, and has to be: it is the only way a
    // director can decline to choose, and a set that omits it forces a choice on every
    // world whether or not the director has an opinion.
    for (core::u32 i = 0u; i <= static_cast<core::u32>(procgen::CaveKind::Auto); ++i)
    {
        const char *word = procgen::caveKindName(static_cast<procgen::CaveKind>(i));
        check(gFull.find(std::string{"\\\""} + std::string{word} + std::string{"\\\""}) != std::string::npos,
              std::string{"the grammar carries the cave kind "} + std::string{word});
    }
    check(gFull.find("spelunk") == std::string::npos, "and cannot spell a kind that does not exist");

    // ── 3. The schema is well-formed JSON, and honours DESIGN §3 ──────────────
    std::printf("\n-- JSON-Schema from the same declaration --\n");
    {
        bool ok = false;
        const std::string doc = agent::emitJsonSchema(onFull);
        (void) editor::detail::parse(doc, &ok);
        check(ok, "the emitted tool schema re-parses as JSON");

        for (const ecs::ComponentSchema &schema : ecs::allSchemas())
        {
            bool componentOk = false;
            const std::string emitted = agent::emitJsonSchema(schema);
            (void) editor::detail::parse(emitted, &componentOk);
            check(componentOk && !emitted.empty(),
                  std::string{"component schema re-parses: "} + std::string{schema.name});
        }

        // The bounds FieldDesc has carried unused since 2026-07-16 now reach the
        // model. If this disappears, DESIGN §3's fourth consumer is gone again.
        const std::string health = agent::emitJsonSchema(ecs::schemaOf(ecs::ComponentId::Health));
        check(health.find("\"minimum\"") != std::string::npos && health.find("\"maximum\"") != std::string::npos,
              "declared bounds reach the schema");

        const std::string query = agent::emitJsonSchema(*onFull.find("query_entities"));
        check(query.find("\"enum\"") != std::string::npos && query.find("\"Position\"") != std::string::npos,
              "a dynamic enum expands from allSchemas()");
        check(query.find("\"additionalProperties\":false") != std::string::npos,
              "the argument object is closed to undeclared parameters");
    }

    // ── 4. Six ways a call can be wrong, each refused with a reason ───────────
    std::printf("\n-- a bad call never reaches the world --\n");
    {
        struct Case {
            const char *what;
            const char *json;
        };
        const Case cases[] = {
            {"unknown tool",                   R"({"tool":"summon_dragon","args":{}})"                 },
            {"tool not offered in this state", R"({"tool":"clear_world","args":{}})"                   },
            {"missing required parameter",     R"({"tool":"load_scene","args":{}})"                    },
            {"value out of declared bounds",   R"({"tool":"query_entities","args":{"limit":999999}})"  },
            {"wrong JSON type",                R"({"tool":"load_scene","args":{"scene":42}})"          },
            {"undeclared parameter",           R"({"tool":"count","args":{"colour":"blue"}})"          },
            {"value outside a closed set",     R"({"tool":"query_entities","args":{"with":"Sparkle"}})"},
            {"malformed JSON",                 R"({"tool":)"                                           },
        };
        // Gated on the EMPTY registry so that "clear_world" is genuinely not
        // offered — the case only means something if the state really forbids it.
        for (const Case &c : cases)
        {
            const auto refused = agent::parseToolCall(c.json, onEmpty);
            check(!refused.has_value(), std::string{"refused: "} + c.what);
        }

        // And a well-formed call is accepted, or the eight above would prove
        // nothing but that the validator says no to everything.
        const auto accepted = agent::parseToolCall(R"({"thought":"look first","tool":"count","args":{}})", onEmpty);
        check(accepted.has_value(), "a well-formed call is accepted");
        if (accepted.has_value())
            check(accepted.value().thought == "look first", "the thought travels with the act");
    }

    // ── 5. An act of the agent is undoable, because it went through the journal ─
    std::printf("\n-- acting through the journal --\n");
    {
        ecs::Registry world;
        editor::CommandJournal journal{world};
        agent::Dispatcher dispatcher{journal, world};

        const agent::ToolRegistry gate = agent::ToolRegistry::forWorld(world);
        const auto report =
            dispatcher.dispatchJson(R"({"tool":"generate_world","args":{"seed":42,"width":32,"depth":32}})", gate);
        check(report.has_value(), "the agent generated a world");
        const core::u32 afterGenerate = editor::entityCount(world);
        check(afterGenerate != 0u, "the world has " + std::to_string(afterGenerate) + " entities");
        check(dispatcher.mutations() == 1u, "one mutating call recorded");

        check(dispatcher.undo(), "undo rewinds it");
        check(editor::entityCount(world) == 0u, "the world is empty again");
        check(dispatcher.mutations() == 0u, "the mutation count follows");

        // An inspection must NOT lengthen the history, or undo would depend on
        // how often somebody looked at the world.
        const agent::ToolRegistry after = agent::ToolRegistry::forWorld(world);
        const auto looked = dispatcher.dispatchJson(R"({"tool":"count","args":{}})", after);
        check(looked.has_value(), "counting works on an empty world");
        check(dispatcher.mutations() == 0u, "looking is not an act");
    }

    // ── 6. The surface signature is stable, and moves when the surface moves ──
    std::printf("\n-- the surface signature --\n");
    {
        const core::u32 a = agent::foldToolSurface(onFull);
        const core::u32 b = agent::foldToolSurface(onFull);
        char buf[80];
        std::snprintf(buf, sizeof(buf), "tool surface fold 0x%08X is deterministic", a);
        check(a == b, buf);
        check(agent::foldToolSurface(onEmpty) != a, "a narrower surface signs differently");

        // The ungated view offers every declared capability, so it signs like a
        // populated world — and that equality is the assertion, not an accident:
        // it holds exactly as long as no tool is gated on the world being EMPTY.
        // The day one is, this line fails and asks to be re-read.
        const agent::ToolRegistry every = agent::ToolRegistry::ungated();
        check(every.size() == agent::kToolCount,
              "the ungated surface offers all " + std::to_string(agent::kToolCount) + " capabilities");
        check(agent::foldToolSurface(every) == a, "no capability is gated on an empty world today");
        check(agent::foldToolSurface(every) != agent::foldToolSurface(onEmpty), "and the empty world sees fewer");
    }

    // ── 7. Looking: the capture is repeatable, and it is agent-hosted ─────────
    std::printf("\n-- looking at the world --\n");
    {
        editor::CommandJournal journal{full};
        agent::Dispatcher dispatcher{journal, full};

        const char *first = "/tmp/lpl-agent-shot-a.ppm";
        const char *second = "/tmp/lpl-agent-shot-b.ppm";
        const std::string poseA =
            std::string{R"({"tool":"take_screenshot","args":{"path":")"} + first + R"(","width":160,"height":100}})";
        const std::string poseB =
            std::string{R"({"tool":"take_screenshot","args":{"path":")"} + second + R"(","width":160,"height":100}})";

        const auto shotA = dispatcher.dispatchJson(poseA, onFull);
        const auto shotB = dispatcher.dispatchJson(poseB, onFull);
        check(shotA.has_value() && shotB.has_value(), "an agent-hosted tool has a handler");
        check(dispatcher.mutations() == 0u, "looking is not a mutation");

        if (shotA.has_value() && shotB.has_value())
        {
            // Byte-for-byte, because a signature that wobbled would make every
            // "the picture changed" observation meaningless.
            const auto readAll = [](const char *path) {
                std::string bytes;
                if (std::FILE *handle = std::fopen(path, "rb"); handle != nullptr)
                {
                    char buffer[4096];
                    std::size_t n = 0u;
                    while ((n = std::fread(buffer, 1u, sizeof(buffer), handle)) != 0u)
                        bytes.append(buffer, n);
                    std::fclose(handle);
                }
                return bytes;
            };
            const std::string bytesA = readAll(first);
            const std::string bytesB = readAll(second);
            check(!bytesA.empty(), "the capture wrote " + std::to_string(bytesA.size()) + " bytes");
            check(bytesA == bytesB, "two captures of the same pose are byte-identical");
            check(bytesA.rfind("P6", 0) == 0, "and they are binary PPM");
        }

        // A different pose must produce a different picture, or the pose is being
        // ignored and the previous check would pass for the wrong reason.
        image::Image straight;
        image::Image tilted;
        const agent::Screenshot a = agent::renderWorld(full, 96, 64, agent::CameraPose{0.0f, 20.0f, 0.0f}, straight);
        const agent::Screenshot b = agent::renderWorld(full, 96, 64, agent::CameraPose{90.0f, 20.0f, 0.0f}, tilted);
        check(a.fold != b.fold, "a different pose gives a different frame signature");
        check(a.entitiesDrawn == onFull.state().entityCount, "every entity was drawn");
        check(a.triangles != 0u, std::to_string(a.triangles) + " triangles rasterised");
    }

    // ── 8. The critics find injected defects and stay quiet on a good world ───
    std::printf("\n-- deterministic critics --\n");
    {
        const agent::Observations clean = agent::inspectWorld(full);
        check(clean.defects() == 0u, "a generated world raises no defect");

        ecs::Registry nothing;
        const agent::Observations empty = agent::inspectWorld(nothing);
        check(empty.defects() == 1u, "an empty world raises exactly one defect");
        check(!empty.findings.empty() && empty.findings[0].code == "empty-world", "and names it: empty-world");
        check(!empty.findings.empty() && empty.findings[0].suggestedTool == "generate_world",
              "and says what would fix it");

        // The asked-for-versus-got half, on reports the generator really emits.
        const agent::Observations dry = agent::reviewGeneration(
            R"({"cmd":"generate_world","rivers":{"enabled":true}})",
            R"({"cmd":"generate_world","ok":true,"created":128,"riverCells":0,"caveFloor":40,"plots":2,"reachable":true,"passed":true})");
        check(dry.defects() == 1u, "rivers enabled and none carved is a defect");
        check(!dry.findings.empty() && dry.findings[0].code == "no-rivers", "and it is named no-rivers");

        const agent::Observations blocked = agent::reviewGeneration(
            R"({"cmd":"generate_world"})",
            R"({"cmd":"generate_world","ok":true,"created":128,"riverCells":3,"caveFloor":40,"plots":2,"reachable":false,"passed":false})");
        check(blocked.defects() == 1u, "an unreachable goal is a defect");
        check(!blocked.findings.empty() && blocked.findings[0].code == "goal-unreachable", "and it is named");

        const agent::Observations fine = agent::reviewGeneration(
            R"({"cmd":"generate_world"})",
            R"({"cmd":"generate_world","ok":true,"created":128,"riverCells":3,"caveFloor":40,"plots":2,"reachable":true,"passed":true})");
        check(fine.defects() == 0u, "a world that got what it asked for raises nothing");

        // A pass the recipe switched OFF must not be reported as missing, or the
        // loop would chase a defect the caller asked for.
        const agent::Observations disabled = agent::reviewGeneration(
            R"({"cmd":"generate_world","rivers":{"enabled":false}})",
            R"({"cmd":"generate_world","ok":true,"created":128,"riverCells":0,"caveFloor":40,"plots":2,"reachable":true,"passed":true})");
        check(disabled.defects() == 0u, "a pass switched off is not a missing pass");

        bool findingsOk = false;
        (void) editor::detail::parse(dry.toJson(), &findingsOk);
        check(findingsOk, "findings serialise as valid JSON");
    }

    // ── 9. diff_scenes names what changed, and nothing when nothing did ───────
    std::printf("\n-- diffing two documents --\n");
    {
        ecs::Registry scratch;
        editor::CommandProcessor processor{scratch};
        const char *docA = R"({\"format\":\"lplscene/1\",\"procedural\":{\"seed\":1,\"terrain\":{\"octaves\":4}}})";
        const char *docB = R"({\"format\":\"lplscene/1\",\"procedural\":{\"seed\":2,\"terrain\":{\"octaves\":4}}})";

        const std::string same = std::string{R"({"cmd":"diff_scenes","a":")"} + docA + R"(","b":")" + docA + R"("})";
        const auto identical = processor.execute(same);
        check(identical.has_value() && identical.value().find("\"identical\":true") != std::string::npos,
              "two identical documents differ in nothing");

        const std::string changed = std::string{R"({"cmd":"diff_scenes","a":")"} + docA + R"(","b":")" + docB + R"("})";
        const auto diff = processor.execute(changed);
        check(diff.has_value() && diff.value().find("\"differences\":1") != std::string::npos,
              "one changed field is exactly one difference");
        check(diff.has_value() && diff.value().find("procedural.seed") != std::string::npos,
              "and it is named by its path");
        bool diffOk = false;
        if (diff.has_value())
            (void) editor::detail::parse(diff.value(), &diffOk);
        check(diffOk, "the diff report is valid JSON");
    }

    std::printf("\n%s (%d failures)\n", failures == 0 ? "ALL PASS" : "FAILURES", failures);
    return failures == 0 ? 0 : 1;
}
