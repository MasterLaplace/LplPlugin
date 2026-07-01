-- /////////////////////////////////////////////////////////////////////////////
-- @file xmake.lua
-- @brief Build configuration for the lpl::samples module — swappable, kernel-
--        independent simulations driven by the generic application runtime.
-- /////////////////////////////////////////////////////////////////////////////

target("lpl-samples")
    set_kind("headeronly")
    set_group("modules")
    -- Header-only sims; they use the software rasterizer + fixed-point math.
    add_deps("lpl-core", "lpl-math", "lpl-render")
    add_headerfiles("include/(lpl/samples/*.hpp)")
    add_includedirs("include", {public = true})
target_end()
