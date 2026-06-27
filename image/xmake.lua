-- /////////////////////////////////////////////////////////////////////////////
-- @file xmake.lua
-- @brief Build configuration for the lpl::image module (portable 2D imaging).
-- /////////////////////////////////////////////////////////////////////////////

target("lpl-image")
    set_kind("static")
    set_group("modules")
    add_deps("lpl-core")
    add_headerfiles("include/(lpl/image/*.hpp)")
    add_includedirs("include", {public = true})
    add_files("src/*.cpp")
