-- /////////////////////////////////////////////////////////////////////////////
-- @file xmake.lua
-- @brief Build configuration for the lpl::bench module (micro-benchmark harness
--        and host introspection).
-- /////////////////////////////////////////////////////////////////////////////

target("lpl-bench")
    set_kind("static")
    set_group("modules")
    add_deps("lpl-core")
    add_headerfiles("include/(lpl/bench/*.hpp)")
    add_includedirs("include", {public = true})
    add_files("src/*.cpp")
