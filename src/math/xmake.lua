-- /////////////////////////////////////////////////////////////////////////////
-- /// @file xmake.lua
-- /// @brief Build configuration for the lpl::math module.
-- /////////////////////////////////////////////////////////////////////////////

target("lpl-math")
    set_kind("static")
    set_group("modules")
    add_deps("lpl-core")
    add_headerfiles("include/(lpl/math/*.hpp)", "include/(lpl/math/*.inl)")
    add_includedirs("include", {public = true})
    add_files("src/*.cpp")
