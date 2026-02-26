-- /////////////////////////////////////////////////////////////////////////////
-- bci/ build configuration — Brain-Computer Interface bridge
-- /////////////////////////////////////////////////////////////////////////////
target("lpl-bci")
    set_kind("static")
    set_group("modules")
    add_deps("lpl-core", "lpl-math", "lpl-input")
    add_includedirs("include", { public = true })
    add_files("src/**.cpp")
    add_headerfiles("include/(lpl/bci/**.hpp)")
target_end()
