-- /////////////////////////////////////////////////////////////////////////////
-- engine/ build configuration — Top-level engine facade & game loop
-- /////////////////////////////////////////////////////////////////////////////
target("lpl-engine")
    set_kind("static")
    set_group("modules")
    add_deps(
        "lpl-core",
        "lpl-math",
        "lpl-memory",
        "lpl-container",
        "lpl-concurrency",
        "lpl-ecs",
        "lpl-physics",
        "lpl-net",
        "lpl-gpu",
        "lpl-input",
        "lpl-audio",
        "lpl-haptic",
        "lpl-bci",
        "lpl-serial"
    )

    if has_config("renderer") then
        add_deps("lpl-render")
    end

    add_includedirs("include", { public = true })
    add_files("src/**.cpp")
    add_headerfiles("include/(lpl/engine/**.hpp)")
target_end()
