#include <lpl/agent/Dispatcher.hpp>
#include <lpl/agent/Grammar.hpp>
#include <lpl/agent/Observation.hpp>
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
#include <string_view>

static int failures = 0;

static void check(bool ok, const std::string &what)
{
    std::printf("  %s: %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok)
        ++failures;
}

/**
 * @brief A world with entities in it, so the RequiresWorld gate is satisfied.
 */
static void seedWorld(lpl::ecs::Registry &registry)
{
    lpl::procgen::WorldRecipe recipe = lpl::procgen::parityWorldRecipe();
    (void) lpl::procgen::bakeWorld(registry, recipe);
}

static constexpr lpl::core::u32 kFnv1aOffsetBasis = 0x811C9DC5u;
static constexpr lpl::core::u32 kFnv1aPrime = 0x01000193u;

static lpl::core::u32 foldBytes(lpl::core::u32 seed, std::string_view bytes)
{
    lpl::core::u32 hash = seed;

    for (const char byte : bytes)
    {
        hash ^= static_cast<lpl::core::u32>(static_cast<unsigned char>(byte));
        hash *= kFnv1aPrime;
    }
    return hash;
}

/**
 * @brief FNV-1a over the JSON Schema and the GBNF grammar of @p registry.
 *
 * @details A tripwire on the surface an intelligence is offered, not a parity gate between
 *          targets: nothing of agent/ is linked into the kernel. Rename a tool, move a bound, add an
 *          enum value or reorder the table, and the signature moves. Both artefacts, not one: two
 *          emitters derive them from the same table, and a change that moved only one of them is
 *          the silent divergence this catches.
 */
static lpl::core::u32 foldToolSurface(const lpl::agent::ToolRegistry &registry)
{
    return foldBytes(foldBytes(kFnv1aOffsetBasis, lpl::agent::emitJsonSchema(registry)),
                     lpl::agent::emitGbnf(registry));
}

/**
 * @brief Every declared tool is a command the engine accepts.
 *
 * @details CommandProcessor's dispatch lives in an anonymous namespace, so this agreement cannot be checked at
 *          compile time. It is checked here instead, by asking the processor and refusing an "unknown command"
 *          reply. A tool named in the table but absent from the processor fails right here.
 *
 * An agent-hosted tool is skipped: checkLookingIsRepeatableAndAgentHosted checks it against its own handler.
 */
static void checkEveryDeclaredToolIsACommandTheEngineAccepts()
{
    std::printf("-- every tool is a real command --\n");
    lpl::ecs::Registry probe;
    lpl::editor::CommandProcessor processor{probe};
    for (const lpl::agent::ToolDesc &tool : lpl::agent::kTools)
    {
        if (tool.host != lpl::agent::ToolHost::Journal)
            continue;
        const std::string command = "{\"cmd\":\"" + std::string{tool.name} + "\"}";
        const auto report = processor.execute(command);
        const bool known = report.has_value() && report.value().find("unknown command") == std::string::npos;
        check(known, std::string{tool.name} + " is dispatchable");
    }
}

/**
 * @brief The offered set is state-dependent, and so is the grammar: gating narrows the surface.
 *
 * @details The claim of DWG-010: what the model cannot see, it cannot call. Not "is refused afterwards" — cannot be
 *          produced by the sampler at all.
 *
 * The closed set of component names comes from the reflection registry, so a new component widens the grammar with
 * no edit to the tool table.
 *
 * The second closed set, and the reason DWG-010 asks for sets in the grammar at all: the underground generators are
 * WORDS, so a director cannot even generate a kind that does not exist. Before the recipe could name them, three of
 * the four were reachable only by hand-written builder calls — a world nothing could save, bake or replay. "auto"
 * is in the set too, and has to be: it is the only way a director can decline to choose, and a set that omits it
 * forces a choice on every world whether or not the director has an opinion.
 */
static void checkTheOfferedSetAndTheGrammarAreStateDependent(const lpl::agent::ToolRegistry &onEmpty,
                                                             const lpl::agent::ToolRegistry &onFull)
{
    check(onEmpty.state().entityCount == 0u, "an empty world reports no entities");
    check(onFull.state().entityCount != 0u,
          "a generated world reports " + std::to_string(onFull.state().entityCount) + " entities");
    check(onEmpty.size() < onFull.size(), "fewer tools on an empty world (" + std::to_string(onEmpty.size()) + " < " +
                                              std::to_string(onFull.size()) + ")");
    check(!onEmpty.offers("spawn_from_template"), "spawn_from_template is NOT offered before a world exists");
    check(onFull.offers("spawn_from_template"), "spawn_from_template IS offered once one does");
    check(onEmpty.offers("generate_world"), "generate_world is always offered");

    const std::string gEmpty = lpl::agent::emitGbnf(onEmpty);
    const std::string gFull = lpl::agent::emitGbnf(onFull);
    check(gEmpty.find("spawn_from_template") == std::string::npos,
          "the empty-world grammar cannot even spell spawn_from_template");
    check(gFull.find("spawn_from_template") != std::string::npos, "the populated grammar can");
    check(gEmpty != gFull, "the grammar is regenerated, not cached");

    for (const lpl::ecs::ComponentSchema &schema : lpl::ecs::allSchemas())
        if (schema.name == "Position" || schema.name == "Health")
            check(gFull.find(std::string{"\\\""} + std::string{schema.name} + "\\\"") != std::string::npos,
                  std::string{"the grammar carries the component name "} + std::string{schema.name});

    for (lpl::core::u32 i = 0u; i <= static_cast<lpl::core::u32>(lpl::procgen::CaveKind::Auto); ++i)
    {
        const char *word = lpl::procgen::caveKindName(static_cast<lpl::procgen::CaveKind>(i));
        check(gFull.find(std::string{"\\\""} + std::string{word} + std::string{"\\\""}) != std::string::npos,
              std::string{"the grammar carries the cave kind "} + std::string{word});
    }
    check(gFull.find("spelunk") == std::string::npos, "and cannot spell a kind that does not exist");
}

/**
 * @brief The schema is well-formed JSON, and honours DESIGN §3.
 *
 * @details The bounds FieldDesc has carried unused since 2026-07-16 now reach the model. If this disappears, DESIGN
 *          §3's fourth consumer is gone again.
 */
static void checkTheSchemaIsWellFormedAndHonoursTheDesign(const lpl::agent::ToolRegistry &onFull)
{
    std::printf("\n-- JSON-Schema from the same declaration --\n");
    bool ok = false;
    const std::string doc = lpl::agent::emitJsonSchema(onFull);
    (void) lpl::editor::detail::parse(doc, &ok);
    check(ok, "the emitted tool schema re-parses as JSON");

    for (const lpl::ecs::ComponentSchema &schema : lpl::ecs::allSchemas())
    {
        bool componentOk = false;
        const std::string emitted = lpl::agent::emitJsonSchema(schema);
        (void) lpl::editor::detail::parse(emitted, &componentOk);
        check(componentOk && !emitted.empty(), std::string{"component schema re-parses: "} + std::string{schema.name});
    }

    const std::string health = lpl::agent::emitJsonSchema(lpl::ecs::schemaOf(lpl::ecs::ComponentId::Health));
    check(health.find("\"minimum\"") != std::string::npos && health.find("\"maximum\"") != std::string::npos,
          "declared bounds reach the schema");

    const std::string query = lpl::agent::emitJsonSchema(*onFull.find("query_entities"));
    check(query.find("\"enum\"") != std::string::npos && query.find("\"Position\"") != std::string::npos,
          "a dynamic enum expands from allSchemas()");
    check(query.find("\"additionalProperties\":false") != std::string::npos,
          "the argument object is closed to undeclared parameters");
}

/**
 * @brief Eight ways a call can be wrong, each refused with a reason: a bad call never reaches the world.
 *
 * @details The calls are gated on the EMPTY registry so that "clear_world" is genuinely not offered — the case only
 *          means something if the state really forbids it. And a well-formed call is accepted, or the eight would
 *          prove nothing but that the validator says no to everything.
 */
static void checkEveryWrongCallIsRefusedWithAReason(const lpl::agent::ToolRegistry &onEmpty)
{
    std::printf("\n-- a bad call never reaches the world --\n");
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
    for (const Case &c : cases)
    {
        const auto refused = lpl::agent::parseToolCall(c.json, onEmpty);
        check(!refused.has_value(), std::string{"refused: "} + c.what);
    }

    const auto accepted = lpl::agent::parseToolCall(R"({"thought":"look first","tool":"count","args":{}})", onEmpty);
    check(accepted.has_value(), "a well-formed call is accepted");
    if (accepted.has_value())
        check(accepted.value().thought == "look first", "the thought travels with the act");
}

/**
 * @brief An act of the agent is undoable, because it went through the journal.
 *
 * @details An inspection must NOT lengthen the history, or undo would depend on how often somebody looked at the
 *          world.
 */
static void checkAnActOfTheAgentIsUndoable()
{
    std::printf("\n-- acting through the journal --\n");
    lpl::ecs::Registry world;
    lpl::editor::CommandJournal journal{world};
    lpl::agent::Dispatcher dispatcher{journal, world};

    const lpl::agent::ToolRegistry gate = lpl::agent::ToolRegistry::forWorld(world);
    const auto report =
        dispatcher.dispatchJson(R"({"tool":"generate_world","args":{"seed":42,"width":32,"depth":32}})", gate);
    check(report.has_value(), "the agent generated a world");
    const lpl::core::u32 afterGenerate = lpl::editor::entityCount(world);
    check(afterGenerate != 0u, "the world has " + std::to_string(afterGenerate) + " entities");
    check(dispatcher.mutations() == 1u, "one mutating call recorded");

    check(dispatcher.undo(), "undo rewinds it");
    check(lpl::editor::entityCount(world) == 0u, "the world is empty again");
    check(dispatcher.mutations() == 0u, "the mutation count follows");

    const lpl::agent::ToolRegistry after = lpl::agent::ToolRegistry::forWorld(world);
    const auto looked = dispatcher.dispatchJson(R"({"tool":"count","args":{}})", after);
    check(looked.has_value(), "counting works on an empty world");
    check(dispatcher.mutations() == 0u, "looking is not an act");
}

/**
 * @brief The surface signature is stable, and moves when the surface moves.
 *
 * @details The ungated view offers every declared capability, so it signs like a populated world — and that
 *          equality is the assertion, not an accident: it holds exactly as long as no tool is gated on the world
 *          being EMPTY. The day one is, this check fails and asks to be re-read.
 */
static void checkTheSurfaceSignatureIsStableAndMovesWithTheSurface(const lpl::agent::ToolRegistry &onEmpty,
                                                                   const lpl::agent::ToolRegistry &onFull)
{
    std::printf("\n-- the surface signature --\n");
    const lpl::core::u32 a = foldToolSurface(onFull);
    const lpl::core::u32 b = foldToolSurface(onFull);
    char buf[80];
    std::snprintf(buf, sizeof(buf), "tool surface fold 0x%08X is deterministic", a);
    check(a == b, buf);
    check(foldToolSurface(onEmpty) != a, "a narrower surface signs differently");

    const lpl::agent::ToolRegistry every = lpl::agent::ToolRegistry::ungated();
    check(every.size() == lpl::agent::kToolCount,
          "the ungated surface offers all " + std::to_string(lpl::agent::kToolCount) + " capabilities");
    check(foldToolSurface(every) == a, "no capability is gated on an empty world today");
    check(foldToolSurface(every) != foldToolSurface(onEmpty), "and the empty world sees fewer");
}

/**
 * @brief Looking: the capture is repeatable, and it is agent-hosted.
 *
 * @details Two captures are compared byte-for-byte, because a signature that wobbled would make every "the picture
 *          changed" observation meaningless. A different pose must produce a different picture, or the pose is
 *          being ignored and the byte-for-byte check would pass for the wrong reason.
 */
static void checkLookingIsRepeatableAndAgentHosted(lpl::ecs::Registry &full, const lpl::agent::ToolRegistry &onFull)
{
    std::printf("\n-- looking at the world --\n");
    lpl::editor::CommandJournal journal{full};
    lpl::agent::Dispatcher dispatcher{journal, full};

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

    lpl::image::Image straight;
    lpl::image::Image tilted;
    const lpl::agent::Screenshot a =
        lpl::agent::renderWorld(full, 96, 64, lpl::agent::CameraPose{0.0f, 20.0f, 0.0f}, straight);
    const lpl::agent::Screenshot b =
        lpl::agent::renderWorld(full, 96, 64, lpl::agent::CameraPose{90.0f, 20.0f, 0.0f}, tilted);
    check(a.fold != b.fold, "a different pose gives a different frame signature");
    check(a.entitiesDrawn == onFull.state().entityCount, "every entity was drawn");
    check(a.triangles != 0u, std::to_string(a.triangles) + " triangles rasterised");
}

/**
 * @brief The critics find injected defects and stay quiet on a good world.
 *
 * @details The asked-for-versus-got half is checked on reports the generator really emits. A pass the recipe
 *          switched OFF must not be reported as missing, or the loop would chase a defect the caller asked for.
 */
static void checkTheCriticsFindInjectedDefectsOnly(const lpl::ecs::Registry &full)
{
    std::printf("\n-- deterministic critics --\n");
    const lpl::agent::Observations clean = lpl::agent::inspectWorld(full);
    check(clean.defects() == 0u, "a generated world raises no defect");

    lpl::ecs::Registry nothing;
    const lpl::agent::Observations empty = lpl::agent::inspectWorld(nothing);
    check(empty.defects() == 1u, "an empty world raises exactly one defect");
    check(!empty.findings.empty() && empty.findings[0].code == "empty-world", "and names it: empty-world");
    check(!empty.findings.empty() && empty.findings[0].suggestedTool == "generate_world", "and says what would fix it");

    const lpl::agent::Observations dry = lpl::agent::reviewGeneration(
        R"({"cmd":"generate_world","rivers":{"enabled":true}})",
        R"({"cmd":"generate_world","ok":true,"created":128,"riverCells":0,"caveFloor":40,"plots":2,"reachable":true,"passed":true})");
    check(dry.defects() == 1u, "rivers enabled and none carved is a defect");
    check(!dry.findings.empty() && dry.findings[0].code == "no-rivers", "and it is named no-rivers");

    const lpl::agent::Observations blocked = lpl::agent::reviewGeneration(
        R"({"cmd":"generate_world"})",
        R"({"cmd":"generate_world","ok":true,"created":128,"riverCells":3,"caveFloor":40,"plots":2,"reachable":false,"passed":false})");
    check(blocked.defects() == 1u, "an unreachable goal is a defect");
    check(!blocked.findings.empty() && blocked.findings[0].code == "goal-unreachable", "and it is named");

    const lpl::agent::Observations fine = lpl::agent::reviewGeneration(
        R"({"cmd":"generate_world"})",
        R"({"cmd":"generate_world","ok":true,"created":128,"riverCells":3,"caveFloor":40,"plots":2,"reachable":true,"passed":true})");
    check(fine.defects() == 0u, "a world that got what it asked for raises nothing");

    const lpl::agent::Observations disabled = lpl::agent::reviewGeneration(
        R"({"cmd":"generate_world","rivers":{"enabled":false}})",
        R"({"cmd":"generate_world","ok":true,"created":128,"riverCells":0,"caveFloor":40,"plots":2,"reachable":true,"passed":true})");
    check(disabled.defects() == 0u, "a pass switched off is not a missing pass");

    bool findingsOk = false;
    (void) lpl::editor::detail::parse(dry.toJson(), &findingsOk);
    check(findingsOk, "findings serialise as valid JSON");
}

/**
 * @brief diff_scenes names what changed, and nothing when nothing did.
 */
static void checkDiffScenesNamesWhatChanged()
{
    std::printf("\n-- diffing two documents --\n");
    lpl::ecs::Registry scratch;
    lpl::editor::CommandProcessor processor{scratch};
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
    check(diff.has_value() && diff.value().find("procedural.seed") != std::string::npos, "and it is named by its path");
    bool diffOk = false;
    if (diff.has_value())
        (void) lpl::editor::detail::parse(diff.value(), &diffOk);
    check(diffOk, "the diff report is valid JSON");
}

/**
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
 *     a reason, for each of the eight ways a call can be wrong;
 *  4. an act of the agent is undoable, because it went through the journal;
 *  5. the surface signature is stable, and moves when the surface moves.
 */
int main()
{
    std::printf("== agent tool surface ==\n\n");

    checkEveryDeclaredToolIsACommandTheEngineAccepts();

    std::printf("\n-- gating narrows the surface --\n");
    lpl::ecs::Registry empty;
    lpl::ecs::Registry full;
    seedWorld(full);

    const lpl::agent::ToolRegistry onEmpty = lpl::agent::ToolRegistry::forWorld(empty);
    const lpl::agent::ToolRegistry onFull = lpl::agent::ToolRegistry::forWorld(full);

    checkTheOfferedSetAndTheGrammarAreStateDependent(onEmpty, onFull);

    checkTheSchemaIsWellFormedAndHonoursTheDesign(onFull);

    checkEveryWrongCallIsRefusedWithAReason(onEmpty);

    checkAnActOfTheAgentIsUndoable();

    checkTheSurfaceSignatureIsStableAndMovesWithTheSurface(onEmpty, onFull);

    checkLookingIsRepeatableAndAgentHosted(full, onFull);

    checkTheCriticsFindInjectedDefectsOnly(full);

    checkDiffScenesNamesWhatChanged();

    std::printf("\n%s (%d failures)\n", failures == 0 ? "ALL PASS" : "FAILURES", failures);
    return failures == 0 ? 0 : 1;
}
