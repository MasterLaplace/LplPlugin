-- /////////////////////////////////////////////////////////////////////////////
-- @file xmake.lua
-- @brief Build configuration for the lpl::agent module.
-- agent/ build configuration — the surface through which an intelligence acts on
-- the world. The seam between LplAssistant and the engine. It holds no model and
-- no inference: it describes the callable surface, validates calls against it,
-- and reports observations back. The tool descriptions are derived from the
-- component reflection registry, so engine, format, editor and the model's
-- grammar all come from one declaration rather than four that drift.
--
-- Depends on editor/, NOT on engine/. The arrow points that way because the
-- engine HOSTS a demon (engine::DemonHost) — the reverse dependency would be a
-- cycle, and it would also be backwards: a tool surface over a world does not
-- need a game loop.
--
-- HOSTED, like editor/. An earlier draft of this file called the module
-- freestanding "because the demon runs in ring 0 on the server profile"; that is
-- not what was decided. Inference is ring 3 and stays there, and dispatch goes
-- through editor::CommandProcessor — the WRITER side of the reader/writer line
-- that keeps ring 0 free of tooling. Nothing here is listed in
-- libengine/arch/i386/make.config or in the kernel's xmake source list, and
-- agent/Parity.hpp is a drift tripwire on the tool surface, not a cross-ring
-- parity gate.
-- /////////////////////////////////////////////////////////////////////////////

target("lpl-agent")
    set_kind("static")
    set_group("modules")
    -- lpl-render and lpl-image are DIRECT dependencies (Vision.cpp), listed even
    -- though lpl-engine would drag them in: a transitive dependency is one
    -- refactor away from disappearing. lpl-render is safe unconditionally — the
    -- module declares a headeronly stub when --renderer is off, and the software
    -- rasteriser this uses lives entirely in headers.
    add_deps("lpl-core", "lpl-math", "lpl-ecs", "lpl-procgen", "lpl-editor", "lpl-image", "lpl-render")
    add_includedirs("include", { public = true })
    add_files("src/**.cpp")
    add_headerfiles("include/(lpl/agent/**.hpp)")
target_end()
