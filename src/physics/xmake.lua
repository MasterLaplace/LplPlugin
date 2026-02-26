-- /////////////////////////////////////////////////////////////////////////////
-- /// @file xmake.lua
-- /// @brief Build configuration for the lpl::physics module.
-- /////////////////////////////////////////////////////////////////////////////

target("lpl-physics")
    set_kind("static")
    set_group("modules")
    add_deps("lpl-core", "lpl-math", "lpl-memory", "lpl-container", "lpl-ecs")
    add_headerfiles("include/(lpl/physics/*.hpp)")
    add_includedirs("include", {public = true})
    add_files("src/*.cpp")
