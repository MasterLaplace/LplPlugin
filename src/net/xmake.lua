-- /////////////////////////////////////////////////////////////////////////////
-- /// @file xmake.lua
-- /// @brief Build configuration for the lpl::net module.
-- /////////////////////////////////////////////////////////////////////////////

target("lpl-net")
    set_kind("static")
    set_group("modules")
    add_deps("lpl-core", "lpl-math", "lpl-memory", "lpl-container", "lpl-concurrency", "lpl-ecs")
    add_headerfiles("include/(lpl/net/**.hpp)")
    add_includedirs("include", {public = true})
    add_files("src/**.cpp")
