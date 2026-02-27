-- /////////////////////////////////////////////////////////////////////////////
-- /// @file xmake.lua
-- /// @brief Build configuration for the lpl::gpu module.
-- /////////////////////////////////////////////////////////////////////////////

target("lpl-gpu")
    set_kind("static")
    set_group("modules")
    add_deps("lpl-core", "lpl-math", "lpl-memory")
    add_headerfiles("include/(lpl/gpu/*.hpp)")
    add_headerfiles("include/(lpl/gpu/*.cuh)")
    add_includedirs("include", {public = true})
    add_files("src/*.cpp")

    -- CUDA sources are only compiled when the CUDA toolchain is available.
    if has_config("cuda") then
        add_files("src/*.cu")
    end
