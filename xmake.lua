-- /////////////////////////////////////////////////////////////////////////////
--  @file xmake.lua
--  @brief Root build configuration for the LplPlugin FullDive Engine.
--
--  Orchestrates all sub-modules as static libraries and links them into
--  three application targets: lpl-server, lpl-client, lpl-benchmark.
--
--  @author MasterLaplace
--  @version 0.2.0
--  @date 2026-02-26
-- /////////////////////////////////////////////////////////////////////////////

set_project("LplPlugin")
set_version("0.2.0")
set_xmakever("2.9.0")

set_languages("c++23", "c17")
set_warnings("allextra", "error")

add_rules("mode.debug", "mode.release", "mode.profile")

if is_mode("debug") then
    set_symbols("debug")
    set_optimize("none")
    add_defines("LPL_DEBUG")
elseif is_mode("release") then
    set_symbols("hidden")
    set_optimize("fastest")
    set_strip("all")
    add_defines("LPL_RELEASE", "NDEBUG")
elseif is_mode("profile") then
    set_symbols("debug")
    set_optimize("fastest")
    add_defines("LPL_PROFILE")
end

add_cxxflags("-fno-rtti", {force = true})
add_cxxflags("-fno-exceptions", {force = true})

-- /////////////////////////////////////////////////////////////////////////////
-- Build options
-- /////////////////////////////////////////////////////////////////////////////

option("renderer")
    set_default(true)
    set_showmenu(true)
    set_description("Enable Vulkan renderer (disable for headless server)")
option_end()

option("cuda")
    set_default(false)
    set_showmenu(true)
    set_description("Enable CUDA GPU physics kernels")
option_end()

-- /////////////////////////////////////////////////////////////////////////////
-- Conditional packages
-- /////////////////////////////////////////////////////////////////////////////

if has_config("renderer") then
    add_requires("vulkan-headers", "vulkan-loader", "vulkan-hpp")
    add_requires("imgui", {system = false, configs = {glfw = true, vulkan = true}})
    add_defines("LPL_HAS_RENDERER")
    add_defines("VULKAN_HPP_NO_EXCEPTIONS")
end

-- /////////////////////////////////////////////////////////////////////////////
-- Sub-modules (ordered by dependency depth, leaves first)
-- /////////////////////////////////////////////////////////////////////////////

includes(
    "core",
    "math",
    "memory",
    "container",
    "concurrency",
    "ecs",
    "physics",
    "net",
    "gpu",
    "input",
    "render",
    "audio",
    "haptic",
    "bci",
    "serial",
    "kernel",
    "engine",
    "apps"
)
