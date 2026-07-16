-- /////////////////////////////////////////////////////////////////////////////
-- @file xmake.lua
-- @brief Build configuration for the lpl::editor module.
-- editor/ build configuration — data-driven scene document (.lplscene) I/O,
-- driven by the ECS component reflection registry. The JSON emit/parse is
-- hand-rolled (no external dependency, exception-free).
-- /////////////////////////////////////////////////////////////////////////////

target("lpl-editor")
    set_kind("static")
    set_group("modules")
    add_deps("lpl-core", "lpl-math", "lpl-ecs")
    add_includedirs("include", { public = true })
    add_files("src/**.cpp")
    add_headerfiles("include/(lpl/editor/**.hpp)")
target_end()
