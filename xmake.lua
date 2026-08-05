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

option("mapview")
    set_default(false)
    set_showmenu(true)
    set_description("Build lpl-mapview, the standalone X11/GLX viewer for generated worlds")
option_end()

option("worldforge")
    set_default(false)
    set_showmenu(true)
    set_description("Build the standalone lpl-worldforge OpenGL world editor (GLFW+imgui, no engine renderer)")
option_end()

-- /////////////////////////////////////////////////////////////////////////////
-- Conditional packages
-- /////////////////////////////////////////////////////////////////////////////

if has_config("renderer") then
    add_requires("vulkan-headers", "vulkan-loader", "vulkan-hpp")
    add_requires("glm")
    add_defines("LPL_HAS_RENDERER")
    add_defines("VULKAN_HPP_NO_EXCEPTIONS")
end

-- imgui and glfw are declared ONCE, here, with the union of the backends the
-- enabled options need. Declaring the same package twice with different configs
-- (render wanting vulkan, worldforge wanting opengl2) does not give two
-- packages: xmake resolves the name to a single instance, and the loser's
-- backend headers simply are not there — which is exactly how worldforge ended
-- up unable to find imgui.h while imgui was "installed".
if has_config("renderer") or has_config("worldforge") then
    add_requires("glfw 3.4", {system = false})
    add_requires("imgui", {
        system = false,
        configs = {
            glfw = true,
            vulkan = has_config("renderer") or false,
            opengl2 = has_config("worldforge") or false,
        }
    })
end

-- The hosted build always has a sockets stack and the BCI dependencies, so the
-- networked session and the BCI adapter are always compiled in. The freestanding
-- kernel build (libengine/Makefile) defines neither and drops both.
add_defines("LPL_HAS_NET", "LPL_HAS_BCI")

-- /////////////////////////////////////////////////////////////////////////////
-- Sub-modules (ordered by dependency depth, leaves first)
-- /////////////////////////////////////////////////////////////////////////////

includes(
    "core",
    "bench",
    "math",
    "memory",
    "container",
    "concurrency",
    "ecs",
    "physics",
    "net",
    "gpu",
    "platform",
    "input",
    "image",
    "scene",
    "render",
    "audio",
    "haptic",
    "bci",
    "serial",
    "editor",
    "procgen",
    "ai",
    "ecology",
    "pack",
    "codec",
    "history",
    "agent",
    "rosetta",
    "kernel",
    "engine",
    "samples",
    "apps",
    "tests"
)
