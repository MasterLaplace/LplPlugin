-- /////////////////////////////////////////////////////////////////////////////
-- /// @file xmake.lua
-- /// @brief Build configuration for the lpl::audio module.
-- /////////////////////////////////////////////////////////////////////////////

target("lpl-audio")
    set_kind("static")
    set_group("modules")
    add_deps("lpl-core", "lpl-math")
    add_headerfiles("include/(lpl/audio/*.hpp)")
    add_includedirs("include", {public = true})
    add_files("src/*.cpp")
