-- /////////////////////////////////////////////////////////////////////////////
-- @file xmake.lua
-- @brief Build configuration for the lpl::pack module.
-- pack/ build configuration — the baked game package (.lplpak): a flat,
-- little-endian POD image a constrained target loads without a JSON parser.
-- Freestanding by construction (no allocation, no libc strings, no exceptions)
-- because the kernel is one of its readers. The host-side baker that produces
-- these images lives in editor/, which is where the JSON belongs.
-- /////////////////////////////////////////////////////////////////////////////

target("lpl-pack")
    set_kind("static")
    set_group("modules")
    add_deps("lpl-core", "lpl-math", "lpl-procgen")
    add_includedirs("include", { public = true })
    add_files("src/**.cpp")
    add_headerfiles("include/(lpl/pack/**.hpp)")
target_end()
