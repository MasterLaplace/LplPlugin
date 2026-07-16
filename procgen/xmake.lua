-- /////////////////////////////////////////////////////////////////////////////
-- @file xmake.lua
-- @brief Build configuration for the lpl::procgen module.
-- procgen/ build configuration — deterministic procedural world generation
-- (Fixed32 value noise, heightfield/scatter generators) that materialise into
-- the ECS. The AI/editor picks seed + high-level params; the engine builds it.
-- /////////////////////////////////////////////////////////////////////////////

target("lpl-procgen")
    set_kind("static")
    set_group("modules")
    add_deps("lpl-core", "lpl-math", "lpl-ecs")
    add_includedirs("include", { public = true })
    add_files("src/**.cpp")
    add_headerfiles("include/(lpl/procgen/**.hpp)")
target_end()
