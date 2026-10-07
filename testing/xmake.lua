-- /////////////////////////////////////////////////////////////////////////////
-- /// @file xmake.lua
-- /// @brief Build configuration for the lpl::testing module.
-- /////////////////////////////////////////////////////////////////////////////

target("lpl-testing")
    set_kind("static")
    set_group("modules")
    add_deps("lpl-core")
    add_headerfiles("include/(lpl/testing/*.hpp)")
    add_includedirs("include", {public = true})
    add_files("src/*.cpp")
target_end()
