-- plugins/bci/xmake.lua
-- Build configuration for the LplPlugin BCI module (V2)
-- Requirements: C++23, Eigen3, Boost, liblsl, BrainFlow, Catch2

set_xmakever("2.8.0")

-- ─── Dependencies ────────────────────────────────────────────────────────────

add_requires("eigen")
add_requires("boost")
add_requires("catch2", {optional = true})

-- liblsl / brainflow: prefer system packages, fall back to xmake repo
package("liblsl")
    set_homepage("https://github.com/sccn/liblsl")
    set_description("Lab Streaming Layer library")
    on_fetch(function (package, opt)
        if opt.system then
            return package:find_package("system::lsl", {includes = "lsl_cpp.h", links = "lsl"})
        end
    end)
package_end()
add_requires("liblsl", {system = true, optional = true})

-- ─── Options ─────────────────────────────────────────────────────────────────

option("with_brainflow")
    set_default(true)
    set_showmenu(true)
    set_description("Enable BrainFlow acquisition backend")
option_end()

-- ─── BCI Static Library ──────────────────────────────────────────────────────

target("lpl-bci")
    set_kind("static")
    set_languages("c++23")
    set_warnings("all", "error")

    add_includedirs("include", {public = true})

    add_files("src/lpl/bci/core/*.cpp")
    add_files("src/lpl/bci/dsp/*.cpp")
    add_files("src/lpl/bci/math/*.cpp")
    add_files("src/lpl/bci/metric/*.cpp")
    add_files("src/lpl/bci/calibration/*.cpp")
    add_files("src/lpl/bci/stream/*.cpp")
    add_files("src/lpl/bci/openvibe/*.cpp")
    add_files("src/lpl/bci/source/*.cpp")
    add_files("src/lpl/bci/source/sim/*.cpp")

    if is_plat("linux", "macosx") then
        add_files("src/lpl/bci/source/serial/SerialPortPosix.cpp")
    elseif is_plat("windows") then
        add_files("src/lpl/bci/source/serial/SerialPortWin32.cpp")
    end

    add_packages("eigen", "boost", "liblsl")

    if has_config("with_brainflow") then
        add_packages("brainflow")
        add_defines("LPL_HAS_BRAINFLOW")
    end

    add_cxxflags("-fno-rtti", {tools = {"gcc", "clang"}})
target_end()

-- ─── Unit Tests ──────────────────────────────────────────────────────────────

target("lpl-bci-tests")
    set_kind("binary")
    set_languages("c++23")
    set_warnings("all", "error")
    set_default(false)

    add_deps("lpl-bci")
    add_packages("catch2")

    add_files("tests/*.cpp")

    add_defines("CATCH_CONFIG_FAST_COMPILE")
target_end()
