-- /////////////////////////////////////////////////////////////////////////////
-- @file xmake.lua
-- @brief Build configuration for the lpl::ecology module.
-- ecology/ build configuration — populations over time: bounded Lotka-Volterra,
-- trophic webs, heritable genomes, drift and the island rule. Ticks in seconds,
-- not frames, and reads procgen's maps to know where isolation is.
-- /////////////////////////////////////////////////////////////////////////////

target("lpl-ecology")
    set_kind("static")
    set_group("modules")
    add_deps("lpl-core", "lpl-math", "lpl-ecs", "lpl-procgen", "lpl-ai")
    add_includedirs("include", { public = true })
    add_files("src/**.cpp")
    add_headerfiles("include/(lpl/ecology/**.hpp)")
target_end()
