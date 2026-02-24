-- plugins/bci/xmake.lua
-- Build configuration for the LplPlugin BCI module (V2)
-- Requirements: C++23, Eigen3, Boost, liblsl, BrainFlow, Catch2

set_xmakever("2.8.0")

add_repositories("package_repo https://github.com/MasterLaplace/xmake-repo.git")

-- ─── Dependencies ────────────────────────────────────────────────────────────

add_requires("eigen")
add_requires("boost")
add_requires("catch2", {optional = true})
add_requires("liblsl")
add_requires("brainflow", {optional = true})

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

    add_files("src/core/*.cpp")
    add_files("src/dsp/*.cpp")
    add_files("src/math/*.cpp")
    add_files("src/metric/*.cpp")
    add_files("src/calibration/*.cpp")
    add_files("src/stream/*.cpp")
    add_files("src/openvibe/*.cpp")
    add_files("src/source/*.cpp")
    add_files("src/source/sim/*.cpp")

    if is_plat("linux", "macosx") then
        add_files("src/source/serial/SerialPortPosix.cpp")
    elseif is_plat("windows") then
        add_files("src/source/serial/SerialPortWin32.cpp")
    end

    add_packages("eigen", "boost", "liblsl", {public = true})

    if has_config("with_brainflow") then
        add_packages("brainflow", {public = true})
        add_defines("LPL_HAS_BRAINFLOW", {public = true})
    end

    add_cxxflags("-fno-rtti", {tools = {"gcc", "clang"}})
target_end()

-- ─── Unit Tests ──────────────────────────────────────────────────────────────

target("lpl-bci-tests")
    set_kind("binary")
    set_languages("c++23")
    set_warnings("all", "error")
    set_default(false)

    add_deps("lpl-bci", {public = true})
    add_packages("catch2", "eigen", "boost", "liblsl")

    if has_config("with_brainflow") then
        add_packages("brainflow")
    end

    add_files("tests/*.cpp")

    add_defines("CATCH_CONFIG_FAST_COMPILE")
target_end()
