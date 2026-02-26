-- /////////////////////////////////////////////////////////////////////////////
-- /// @file xmake.lua
-- /// @brief Build configuration for the lpl::container module.
-- /////////////////////////////////////////////////////////////////////////////

target("lpl-container")
    set_kind("static")
    set_group("modules")
    add_deps("lpl-core", "lpl-memory")
    add_headerfiles("include/(lpl/container/*.hpp)", "include/(lpl/container/*.inl)")
    add_includedirs("include", {public = true})
    add_files("src/*.cpp")
