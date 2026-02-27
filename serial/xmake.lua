-- /////////////////////////////////////////////////////////////////////////////
-- serial/ build configuration — Serialization, snapshots, replay
-- /////////////////////////////////////////////////////////////////////////////
target("lpl-serial")
    set_kind("static")
    set_group("modules")
    add_deps("lpl-core", "lpl-math", "lpl-ecs", "lpl-net", "lpl-input")
    add_includedirs("include", { public = true })
    add_files("src/**.cpp")
    add_headerfiles("include/(lpl/serial/**.hpp)")
target_end()
