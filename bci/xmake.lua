-- /////////////////////////////////////////////////////////////////////////////
-- bci/ build configuration — Brain-Computer Interface bridge
-- Merged: src/bci adapter + plugins/bci pipeline (DSP, metrics, sources)
-- /////////////////////////////////////////////////////////////////////////////
target("lpl-bci")
    set_kind("static")
    set_group("modules")
    add_deps("lpl-core", "lpl-math", "lpl-input")
    add_includedirs("include", { public = true })
    add_files("src/**.cpp")
    add_headerfiles("include/(lpl/bci/**.hpp)")
    add_headerfiles("include/(lpl/bci/**.inl)")

    -- Exclude files requiring external dependencies not yet installed
    -- These will be re-enabled when Eigen, Boost, liblsl, BrainFlow are added
    remove_files("src/math/Covariance.cpp")
    remove_files("src/math/Riemannian.cpp")
    remove_files("src/metric/StabilityMetric.cpp")
    remove_files("src/openvibe/StabilityMonitorBox.cpp")
    remove_files("src/source/BrainFlowSource.cpp")
    remove_files("src/source/OpenBciSource.cpp")
    remove_files("src/source/LslSource.cpp")
    remove_files("src/source/SourceFactory.cpp")
    remove_files("src/stream/LslOutlet.cpp")
    remove_files("src/source/serial/**.cpp")
target_end()
