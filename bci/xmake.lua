-- /////////////////////////////////////////////////////////////////////////////
-- bci/ build configuration — Brain-Computer Interface bridge (hosted-only)
--
-- The acquisition and analysis backends pull in host-only third-party
-- libraries:
--   * Eigen              — covariance / Riemannian geometry (math, metrics)
--   * liblsl             — Lab Streaming Layer source and outlet
--   * BrainFlow (opt.)   — OpenBCI / BrainFlow acquisition backend
--
-- The DSP acquisition ring buffer uses the engine's own SPSC lpl-container
-- RingBuffer (no Boost dependency — see container/RingBuffer.hpp).
--
-- This module is hosted-only: the freestanding kernel build does not compile
-- it, so these dependencies never reach the kernel target.
-- /////////////////////////////////////////////////////////////////////////////

add_repositories("laplace-xmake-repo https://github.com/MasterLaplace/xmake-repo.git feat/brainflow-liblsl")

add_requires("eigen")
add_requires("liblsl")
add_requires("brainflow", { optional = true })

option("with_brainflow")
    set_default(true)
    set_showmenu(true)
    set_description("Enable the BrainFlow acquisition backend in lpl-bci")
option_end()

target("lpl-bci")
    set_kind("static")
    set_group("modules")
    set_warnings("all", "error")
    add_deps("lpl-core", "lpl-math", "lpl-input", "lpl-container")
    add_includedirs("include", { public = true })
    add_files("src/**.cpp")
    add_headerfiles("include/(lpl/bci/**.hpp)")

    if is_plat("windows") then
        remove_files("src/source/serial/SerialPortPosix.cpp")
    else
        remove_files("src/source/serial/SerialPortWin32.cpp")
    end

    add_cxxflags("-fexceptions", { force = true })

    add_packages("eigen", "liblsl", { public = true })

    if has_config("with_brainflow") then
        add_packages("brainflow", { public = true })
    end

    on_config(function (target)
        if target:pkg("brainflow") then
            target:add("defines", "LPL_HAS_BRAINFLOW", { public = true })
        end
    end)
target_end()
