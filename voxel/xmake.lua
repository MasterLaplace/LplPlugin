-- /////////////////////////////////////////////////////////////////////////////
-- @file xmake.lua
-- @brief Build configuration for the lpl::voxel module.
--
-- voxel/ is direct volume rendering over a bounded resident set of bricks: no
-- mesh, no threshold, no ownership and no I/O. It knows nothing about where the
-- samples came from, which is what lets the same marcher serve a tomographic
-- scan, a microscopy stack or a simulation grid.
-- /////////////////////////////////////////////////////////////////////////////

target("lpl-voxel")
    set_kind("static")
    set_group("modules")
    add_deps("lpl-core", "lpl-math")
    add_includedirs("include", { public = true })
    add_files("src/**.cpp")
    add_headerfiles("include/(lpl/voxel/**.hpp)")
target_end()
