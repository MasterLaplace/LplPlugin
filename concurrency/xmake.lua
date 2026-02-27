-- /////////////////////////////////////////////////////////////////////////////
-- /// @file xmake.lua
-- /// @brief Build configuration for the lpl::concurrency module.
-- /////////////////////////////////////////////////////////////////////////////

target("lpl-concurrency")
    set_kind("static")
    set_group("modules")
    add_deps("lpl-core")
    add_headerfiles("include/(lpl/concurrency/*.hpp)")
    add_includedirs("include", {public = true})
    add_files("src/*.cpp")
