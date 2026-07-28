-- /////////////////////////////////////////////////////////////////////////////
-- @file xmake.lua
-- @brief Build configuration for the lpl::ai module.
-- ai/ build configuration — authoritative agent behaviour: stigmergy fields,
-- directional pathfinding, abstract/realised creatures, swarms and spring
-- bodies. Ring-0 safe: Fixed32, lpl::pmr, no libm, no __int128.
-- /////////////////////////////////////////////////////////////////////////////

target("lpl-ai")
    set_kind("static")
    set_group("modules")
    add_deps("lpl-core", "lpl-math", "lpl-ecs", "lpl-procgen")
    add_includedirs("include", { public = true })
    add_files("src/**.cpp")
    add_headerfiles("include/(lpl/ai/**.hpp)")
target_end()
