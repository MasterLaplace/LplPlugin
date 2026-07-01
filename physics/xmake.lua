-- /////////////////////////////////////////////////////////////////////////////
-- @file xmake.lua
-- @brief Build configuration for the lpl::physics module.
-- /////////////////////////////////////////////////////////////////////////////

target("lpl-physics")
    set_kind("static")
    set_group("modules")
    add_deps("lpl-core", "lpl-math", "lpl-memory", "lpl-container", "lpl-ecs")
    add_headerfiles("include/(lpl/physics/*.hpp)")
    add_includedirs("include", {public = true})
    -- Default (freestanding-safe) sources only. GpuPhysicsBackend is excluded
    -- here so the CPU-only / kernel build never pulls the CUDA-host gpu module.
    add_files("src/*.cpp")
    remove_files("src/GpuPhysicsBackend.cpp")

    -- GPU bridge: compiled only with the CUDA toolchain. It drives lpl::gpu's
    -- IComputeBackend; gpu depends on neither physics nor ecs, so no cycle.
    if has_config("cuda") then
        add_deps("lpl-gpu")
        add_files("src/GpuPhysicsBackend.cpp")
    end
target_end()
