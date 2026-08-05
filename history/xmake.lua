-- /////////////////////////////////////////////////////////////////////////////
-- @file xmake.lua
-- @brief Build configuration for the lpl::history module.
-- history/ build configuration — the demon's knowledge, applied to a running world.
-- procgen/ builds a plausible world; history/ constrains a world to be OUR world.
-- It adds no simulation of its own: it binds dated, sourced, weighted facts to the
-- systems that already run (ecology populations, settlements, agents), and it
-- measures the gap between what the simulation did and what the record says. That
-- gap is the demon's score. Authoritative, therefore Fixed32 and ring 0.
-- /////////////////////////////////////////////////////////////////////////////

target("lpl-history")
    set_kind("static")
    set_group("modules")
    add_deps("lpl-core", "lpl-math", "lpl-ecs", "lpl-ecology", "lpl-procgen")
    add_includedirs("include", { public = true })
    add_files("src/**.cpp")
    add_headerfiles("include/(lpl/history/**.hpp)")
target_end()
