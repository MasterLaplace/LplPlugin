-- /////////////////////////////////////////////////////////////////////////////
-- /// @file xmake.lua
-- /// @brief Build configuration for the lpl::memory module.
-- /////////////////////////////////////////////////////////////////////////////

target("lpl-memory")
    set_kind("static")
    set_group("modules")
    add_deps("lpl-core")
    add_headerfiles("include/(lpl/memory/*.hpp)", "include/(lpl/memory/*.inl)")
    add_includedirs("include", {public = true})
    add_files("src/*.cpp")
