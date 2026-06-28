-- /////////////////////////////////////////////////////////////////////////////
-- @file xmake.lua
-- @brief Build configuration for the lpl::scene module (2D scene graph).
-- /////////////////////////////////////////////////////////////////////////////

target("lpl-scene")
    set_kind("static")
    set_group("modules")
    add_deps("lpl-core", "lpl-math")
    add_headerfiles("include/(lpl/scene/*.hpp)")
    add_includedirs("include", {public = true})
    add_files("src/*.cpp")
