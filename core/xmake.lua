-- /////////////////////////////////////////////////////////////////////////////
-- /// @file xmake.lua
-- /// @brief Build configuration for the lpl::core module.
-- /////////////////////////////////////////////////////////////////////////////

target("lpl-core")
    set_kind("static")
    set_group("modules")
    add_headerfiles("include/(lpl/core/*.hpp)")
    add_includedirs("include", {public = true})
    add_files("src/*.cpp")
