-- /////////////////////////////////////////////////////////////////////////////
-- /// @file xmake.lua
-- /// @brief Build configuration for the lpl::ecs module.
-- /////////////////////////////////////////////////////////////////////////////

target("lpl-ecs")
    set_kind("static")
    set_group("modules")
    add_deps("lpl-core", "lpl-math", "lpl-memory", "lpl-container", "lpl-concurrency")
    add_headerfiles("include/(lpl/ecs/*.hpp)")
    add_includedirs("include", {public = true})
    add_files("src/*.cpp")
