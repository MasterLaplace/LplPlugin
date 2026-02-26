-- /////////////////////////////////////////////////////////////////////////////
-- /// @file xmake.lua
-- /// @brief Build configuration for the lpl::input module.
-- /////////////////////////////////////////////////////////////////////////////

target("lpl-input")
    set_kind("static")
    set_group("modules")
    add_deps("lpl-core", "lpl-math")
    add_headerfiles("include/(lpl/input/*.hpp)")
    add_includedirs("include", {public = true})
    add_files("src/*.cpp")
