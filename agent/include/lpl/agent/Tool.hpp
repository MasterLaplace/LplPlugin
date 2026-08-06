/**
 * @file Tool.hpp
 * @brief One callable capability, its parameters and their bounds.
 *
 * Derived from the reflection registry rather than written by hand — the lesson
 * of the R-Type editor, whose hard-coded getDefaultComponentValue silently broke
 * every time the engine moved.
 *
 * The table below is `constexpr` for the same reason `procgen::parityWorldRecipe`
 * is: the schema emitter, the grammar emitter, the call validator and the parity
 * fold must all read the SAME declaration. Four consumers of one table cannot
 * disagree; four hand-written lists always end up disagreeing.
 *
 * Every @ref ToolDesc::name here must be a command @c editor::CommandProcessor
 * accepts. That processor's dispatch lives in an anonymous namespace, so the
 * agreement cannot be checked at compile time — it is asserted at run time by
 * `test-agent-tools`, which dispatches every declared tool and refuses an
 * "unknown command" reply.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_AGENT_TOOL_HPP
#    define LPL_LPL_AGENT_TOOL_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/ecs/ComponentReflection.hpp>
#    include <lpl/procgen/WorldRecipe.hpp>

#    include <span>
#    include <string_view>

namespace lpl::agent {

/**
 * @enum ParamKind
 * @brief The JSON type of a tool argument.
 *
 * Deliberately NOT @c ecs::FieldType. That enum describes memory — @c Vec3Fixed
 * is three raw Q16.16 lanes at known offsets — whereas this describes what a
 * caller may write in a JSON object. The two vocabularies answer different
 * questions, and the bridge between them is @ref paramKindOf, which is the one
 * mapping both emitters go through.
 */
enum class ParamKind : core::u8 {
    Number = 0, ///< JSON number, human units (metres, seconds, 0..1 ratios).
    Integer,    ///< JSON number constrained to integers (counts, seeds, raw Q16.16).
    String,     ///< JSON string.
    Bool,       ///< JSON true/false.
    Object,     ///< Nested JSON object, validated by whoever consumes it.
    Array       ///< JSON array.
};

/**
 * @brief The JSON type that represents a component field.
 *
 * Extracted from @c tests/parity/test_reflection.cpp, whose own header said the
 * emitters "will move into the editor module". They moved here instead, because
 * here is where the second consumer appeared: a JSON-Schema is only ever wanted
 * in order to constrain a model. The test now calls this one, so the emitter it
 * pins and the emitter the grammar uses are the same function.
 */
[[nodiscard]] constexpr ParamKind paramKindOf(ecs::FieldType t) noexcept
{
    switch (t)
    {
    case ecs::FieldType::F32: return ParamKind::Number;
    // Authoritative fields cross the wire as their RAW Q16.16 integer, never as
    // a decimal: a model that wrote 1.5 and a kernel that read 98304 would agree
    // in prose and diverge in arithmetic.
    case ecs::FieldType::Fixed32:
    case ecs::FieldType::I32:
    case ecs::FieldType::U32:
    case ecs::FieldType::U16:
    case ecs::FieldType::U8: return ParamKind::Integer;
    default: return ParamKind::Object;
    }
}

/** @brief The JSON-Schema spelling of a @ref ParamKind. */
[[nodiscard]] constexpr std::string_view jsonTypeName(ParamKind k) noexcept
{
    switch (k)
    {
    case ParamKind::Number: return "number";
    case ParamKind::Integer: return "integer";
    case ParamKind::String: return "string";
    case ParamKind::Bool: return "boolean";
    case ParamKind::Array: return "array";
    case ParamKind::Object: return "object";
    }
    return "object";
}

/**
 * @enum DynamicEnum
 * @brief A closed set of accepted strings that is not known at declaration time.
 *
 * The component names a query may filter on are the twelve entries of
 * @c ecs::allSchemas(). Writing them out here would be a thirteenth list to keep
 * in step, so the parameter names its SOURCE and the emitters expand it. Adding a
 * component therefore widens the grammar with no edit to this file.
 */
enum class DynamicEnum : core::u8 {
    None = 0,      ///< Any string of the declared kind is accepted.
    ComponentName, ///< One of @c ecs::allSchemas() names.
    CaveKind       ///< One of @c procgen::caveKindName's words, "auto" included.
};

/**
 * @brief Visits every word the closed set @p choices stands for.
 *
 * ONE definition of "what set is this", because three consumers ask: the grammar puts
 * it in the sampler, the schema publishes it, and the validator refuses anything
 * outside it. Each of those used to test @c choices itself, which is three places to
 * remember when a fourth set is added — and a set the grammar knows but the validator
 * does not is a rejection with no explanation.
 *
 * @param choices The set.
 * @param sink    Called with each @c std::string_view in the set; nothing for None.
 */
template <typename Sink> void forEachChoice(DynamicEnum choices, Sink &&sink)
{
    switch (choices)
    {
    case DynamicEnum::ComponentName:
        for (const ecs::ComponentSchema &schema : ecs::allSchemas())
            sink(schema.name);
        return;
    case DynamicEnum::CaveKind:
        for (core::u32 i = 0u; i <= static_cast<core::u32>(procgen::CaveKind::Auto); ++i)
            sink(std::string_view{procgen::caveKindName(static_cast<procgen::CaveKind>(i))});
        return;
    case DynamicEnum::None: return;
    }
}

/// @return Whether @p text belongs to @p choices. None accepts anything.
[[nodiscard]] inline bool inChoices(DynamicEnum choices, std::string_view text)
{
    if (choices == DynamicEnum::None)
        return true;
    bool found = false;
    forEachChoice(choices, [&](std::string_view word) { found = found || word == text; });
    return found;
}

/**
 * @struct ToolParam
 * @brief One named argument of a tool.
 *
 * Bounds are in the argument's own units and are inclusive, mirroring
 * @c ecs::FieldDesc::hasBounds/minRaw/maxRaw — which DESIGN §3 added for exactly
 * this purpose and which nothing had consumed until now. They are @c double
 * rather than @c i64 because this is a JSON surface and JSON numbers are doubles;
 * every bound declared below is exactly representable.
 */
struct ToolParam {
    std::string_view name;
    ParamKind kind{ParamKind::Number};
    bool required{false};
    bool hasBounds{false};
    double minValue{0.0};
    double maxValue{0.0};
    DynamicEnum choices{DynamicEnum::None};
    std::string_view brief{};
};

/**
 * @enum ToolGate
 * @brief The world-state precondition under which a tool is offered at all.
 *
 * This is the whole point of regenerating the grammar every step: a tool that
 * cannot work right now is not described to the model, so it cannot be called
 * and then refused. Refusals cost a turn; absence costs nothing.
 */
enum class ToolGate : core::u8 {
    Always = 0,        ///< Offered unconditionally.
    RequiresWorld,     ///< Offered once the world holds at least one entity.
    RequiresEmptyWorld ///< Offered only while nothing has been generated yet.
};

/**
 * @enum ToolHost
 * @brief Who actually executes a capability.
 *
 * There is ONE surface that changes a world — @c editor::CommandProcessor, driven
 * through @c editor::CommandJournal — and that is not negotiable: a mutation that
 * bypassed the journal would be a mutation no replay could reproduce.
 *
 * Observation is different. Taking a picture needs @c render and @c image, which
 * @c editor deliberately does not link (it is the module that reads and writes
 * documents, not the one that draws). Rather than drag a rasteriser into the
 * editor, or invert the dependency, a purely observing capability may be served
 * here. The rule that keeps this from becoming two command surfaces is asserted
 * at compile time in Tool.cpp: **anything that mutates is hosted by the journal**.
 */
enum class ToolHost : core::u8 {
    Journal = 0, ///< editor::CommandProcessor, through the journal: undoable, replayable.
    Agent        ///< Served by agent/, because it needs what editor does not link.
};

/**
 * @struct ToolDesc
 * @brief One callable capability.
 */
struct ToolDesc {
    std::string_view name; ///< For a Journal-hosted tool, a command CommandProcessor accepts.
    std::string_view brief;
    std::span<const ToolParam> params;
    ToolGate gate{ToolGate::Always};
    bool mutates{false}; ///< Recorded in the CommandJournal, hence undoable.
    ToolHost host{ToolHost::Journal};
};

namespace detail {

// ── generate_world ─────────────────────────────────────────────────────────────
// The nested blocks are declared as opaque objects ON PURPOSE. Their ~90 field
// names live in editor/src/GamePackBaker.cpp's parseSceneRecipe, which is the one
// authority on what a recipe may say; restating them here would be the second
// recipe parser §18.2 avoided, and it would drift on the first field added.
// parseSceneRecipe already rejects a malformed block with a reason, so a wrong
// nested field is caught — one turn later, by an error rather than by a grammar.
// Deriving these from a reflection table over WorldRecipe (the way components are
// derived) is a real improvement and a chantier of its own; no such table exists.
inline constexpr ToolParam kGenerateWorldParams[] = {
    {"seed",              ParamKind::Integer, false, true,  0.0,  4294967295.0, DynamicEnum::None,
     "Master seed; same seed, same world."                                                                                                     },
    {"width",             ParamKind::Integer, false, true,  4.0,  1024.0,       DynamicEnum::None,     "Heightfield columns."                  },
    {"depth",             ParamKind::Integer, false, true,  4.0,  1024.0,       DynamicEnum::None,     "Heightfield rows."                     },
    {"cellSize",          ParamKind::Number,  false, true,  0.25, 256.0,        DynamicEnum::None,     "World units per cell."                 },
    {"materializeGround", ParamKind::Bool,    false, false, 0.0,  0.0,          DynamicEnum::None,
     "Spawn ground entities, not just the field."                                                                                              },
    {"terrain",           ParamKind::Object,  false, false, 0.0,  0.0,          DynamicEnum::None,     "Noise: frequency, octaves, warp, kind."},
    {"erosion",           ParamKind::Object,  false, false, 0.0,  0.0,          DynamicEnum::None,     "Thermal and hydraulic relaxation."     },
    {"rivers",            ParamKind::Object,  false, false, 0.0,  0.0,          DynamicEnum::None,     "Drainage: density, carve depth."       },
    {"climate",           ParamKind::Object,  false, false, 0.0,  0.0,          DynamicEnum::None,
     "Rainfall, wind, sea level, rain shadow."                                                                                                 },
    {"biomes",            ParamKind::Object,  false, false, 0.0,  0.0,          DynamicEnum::None,     "Elevation bands and snowline."         },
    {"climateAxes",       ParamKind::Object,  false, false, 0.0,  0.0,          DynamicEnum::None,     "The six-axis climate hypercube."       },
    {"terraceSteps",      ParamKind::Integer, false, true,  0.0,  64.0,         DynamicEnum::None,
     "Terrace the field into this many steps; 0 leaves it smooth."                                                                             },
    {"provinces",         ParamKind::Object,  false, false, 0.0,  0.0,          DynamicEnum::None,
     "Voronoi districting of the surface: cellSize, jitter, metric."                                                                           },
    // The ONE parameter whose legal values are a closed set of words, so they belong
    // in the sampler and not merely in a rejection: DWG-010 to the letter.
    {"caveKind",          ParamKind::String,  false, false, 0.0,  0.0,          DynamicEnum::CaveKind,
     "Which underground generator runs. `layered` cannot be judged by the gate."                                                               },
    {"caves",             ParamKind::Object,  false, false, 0.0,  0.0,          DynamicEnum::None,     "Cellular-automaton cave carving."      },
    {"rooms",             ParamKind::Object,  false, false, 0.0,  0.0,          DynamicEnum::None,
     "Underground as a recursive room partition, when caveKind is bsp."                                                                        },
    {"aggregation",       ParamKind::Object,  false, false, 0.0,  0.0,          DynamicEnum::None,
     "Underground as diffusion-limited aggregation, when caveKind is dla."                                                                     },
    {"caveSystem",        ParamKind::Object,  false, false, 0.0,  0.0,          DynamicEnum::None,
     "Underground as a stack of plans joined by shafts, when caveKind is layered."                                                             },
    {"buildings",         ParamKind::Object,  false, false, 0.0,  0.0,          DynamicEnum::None,
     "Raise the plots with the shape grammar: storeys, roof, materials."                                                                       },
    {"roadside",          ParamKind::Object,  false, false, 0.0,  0.0,          DynamicEnum::None,
     "Decorate the verges from an L-system pattern."                                                                                           },
    {"settlement",        ParamKind::Object,  false, false, 0.0,  0.0,          DynamicEnum::None,     "Roads, plots, shape grammar."          },
    {"gate",              ParamKind::Object,  false, false, 0.0,  0.0,          DynamicEnum::None,
     "Playability gate: reachability, path length."                                                                                            },
    {"scatter",           ParamKind::Array,   false, false, 0.0,  0.0,          DynamicEnum::None,     "Per-biome prop and vegetation rules."  },
};

inline constexpr ToolParam kLoadSceneParams[] = {
    {"scene", ParamKind::String, true, false, 0.0, 0.0, DynamicEnum::None, "A whole .lplscene document."},
};

inline constexpr ToolParam kSpawnFromTemplateParams[] = {
    {"templates", ParamKind::Object,  true,  false, 0.0, 0.0,    DynamicEnum::None,
     "Named prefabs; $use chains are resolved."                                                                     },
    {"name",      ParamKind::String,  true,  false, 0.0, 0.0,    DynamicEnum::None, "Which template to instantiate."},
    {"count",     ParamKind::Integer, false, true,  1.0, 4096.0, DynamicEnum::None, "How many instances."           },
    {"overrides", ParamKind::Object,  false, false, 0.0, 0.0,    DynamicEnum::None, "Per-instance field overrides." },
};

inline constexpr ToolParam kQueryEntitiesParams[] = {
    {"with",  ParamKind::String,  false, false, 0.0,      0.0,     DynamicEnum::ComponentName,
     "Only entities carrying this component."                                                                                      },
    // Bounds in HUMAN units, converted with Fixed32::fromFloat by the command.
    // Fixed32 saturates near +-32767, so a range wider than that is not "very
    // large", it is undefined — the same trap as the raw-vs-value constructor.
    {"minX",  ParamKind::Number,  false, true,  -32767.0, 32767.0, DynamicEnum::None,          "Box lower bound on X."             },
    {"maxX",  ParamKind::Number,  false, true,  -32767.0, 32767.0, DynamicEnum::None,          "Box upper bound on X."             },
    {"minY",  ParamKind::Number,  false, true,  -32767.0, 32767.0, DynamicEnum::None,          "Box lower bound on Y."             },
    {"maxY",  ParamKind::Number,  false, true,  -32767.0, 32767.0, DynamicEnum::None,          "Box upper bound on Y."             },
    {"minZ",  ParamKind::Number,  false, true,  -32767.0, 32767.0, DynamicEnum::None,          "Box lower bound on Z."             },
    {"maxZ",  ParamKind::Number,  false, true,  -32767.0, 32767.0, DynamicEnum::None,          "Box upper bound on Z."             },
    {"limit", ParamKind::Integer, false, true,  1.0,      4096.0,  DynamicEnum::None,          "How many sample indices to return."},
};

inline constexpr ToolParam kTakeScreenshotParams[] = {
    {"path",     ParamKind::String,  true,  false, 0.0,    0.0,    DynamicEnum::None, "Where to write the binary PPM."},
    {"width",    ParamKind::Integer, false, true,  16.0,   4096.0, DynamicEnum::None, "Image width in pixels."        },
    {"height",   ParamKind::Integer, false, true,  16.0,   4096.0, DynamicEnum::None, "Image height in pixels."       },
    {"yawDeg",   ParamKind::Number,  false, true,  -360.0, 360.0,  DynamicEnum::None, "Orbit camera yaw, degrees."    },
    {"pitchDeg", ParamKind::Number,  false, true,  -89.0,  89.0,   DynamicEnum::None, "Orbit camera pitch, degrees."  },
    {"distance", ParamKind::Number,  false, true,  0.0,    4096.0, DynamicEnum::None,
     "Orbit radius; 0 frames the whole world."                                                                        },
};

inline constexpr ToolParam kDiffScenesParams[] = {
    {"a",     ParamKind::String,  true,  false, 0.0, 0.0,    DynamicEnum::None, "First .lplscene document."    },
    {"b",     ParamKind::String,  true,  false, 0.0, 0.0,    DynamicEnum::None, "Second .lplscene document."   },
    {"limit", ParamKind::Integer, false, true,  1.0, 4096.0, DynamicEnum::None, "How many differences to name."},
};

} // namespace detail

/**
 * @brief Every capability the engine can offer, before gating.
 *
 * Order is stable and load-bearing: @ref foldToolSurface signs this table, so a
 * reordering is a deliberate change that a test reports rather than a silent one.
 */
inline constexpr ToolDesc kTools[] = {
    {"generate_world",
     "Generate a whole world from one recipe: terrain, erosion, rivers, climate, biomes, caves, "
     "settlement, props. One call, because a world is one pipeline.", detail::kGenerateWorldParams, ToolGate::Always, true},
    {"load_scene", "Replace the world with a .lplscene document.", detail::kLoadSceneParams, ToolGate::Always, true},
    {"save_scene", "Return the current world as a .lplscene document.", std::span<const ToolParam>{},
     ToolGate::RequiresWorld, false},
    {"count", "Number of live entities.", std::span<const ToolParam>{}, ToolGate::Always, false},
    {"spawn_from_template", "Place instances of a named prefab.", detail::kSpawnFromTemplateParams,
     ToolGate::RequiresWorld, true},
    {"clear_world", "Destroy every entity.", std::span<const ToolParam>{}, ToolGate::RequiresWorld, true},
    {"get_world_stats", "Entity, archetype and chunk counts, bounds, per-component tallies, state signature.",
     std::span<const ToolParam>{}, ToolGate::RequiresWorld, false},
    {"query_entities", "Count entities matching a component and/or a box; returns a bounded sample.",
     detail::kQueryEntitiesParams, ToolGate::RequiresWorld, false},
    {"diff_scenes", "Structured differences between two .lplscene documents.", detail::kDiffScenesParams,
     ToolGate::Always, false},
    // Agent-hosted: it needs render/ and image/, which editor/ does not link.
    // Purely observing, so the single-mutation-surface rule is untouched.
    {"take_screenshot", "Render the world off-screen from an orbit pose and write a PPM. Look at what you built.",
     detail::kTakeScreenshotParams, ToolGate::RequiresWorld, false, ToolHost::Agent},
};

/** @brief How many capabilities exist in total, gating aside. */
inline constexpr core::u32 kToolCount = static_cast<core::u32>(std::size(kTools));

// Every Journal-hosted entry above must be a command editor::CommandProcessor
// already accepts — a table that advertised a capability the engine does not have
// would be a lie the model pays for, one wasted turn at a time. A capability is
// declared here on the same change that implements it, never before, and
// `test-agent-tools` asserts the agreement so the rule has a consequence.
static_assert(kToolCount == 10u, "kTools changed size — the tool-surface signature moves with it");

/** @brief The capability named @p name, or nullptr. */
[[nodiscard]] constexpr const ToolDesc *findTool(std::string_view name) noexcept
{
    for (const ToolDesc &tool : kTools)
        if (tool.name == name)
            return &tool;
    return nullptr;
}

} // namespace lpl::agent

#endif // LPL_LPL_AGENT_TOOL_HPP
