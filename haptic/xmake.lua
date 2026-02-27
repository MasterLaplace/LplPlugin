-- /////////////////////////////////////////////////////////////////////////////
-- haptic/ build configuration
-- /////////////////////////////////////////////////////////////////////////////
target("lpl-haptic")
    set_kind("static")
    set_group("modules")
    add_deps("lpl-core", "lpl-math")
    add_includedirs("include", { public = true })
    add_files("src/**.cpp")
    add_headerfiles("include/(lpl/haptic/**.hpp)")
target_end()
