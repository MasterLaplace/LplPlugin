-- /////////////////////////////////////////////////////////////////////////////
-- /// @file xmake.lua
-- /// @brief Build configuration for the lpl::render module.
-- /////////////////////////////////////////////////////////////////////////////

if has_config("renderer") then
    target("lpl-render")
        set_kind("static")
        set_group("modules")
        add_includedirs("include", {public = true})

        -- Expose the internal headers of the Vulkan implementation specifically to the render module itself
        add_includedirs("src/vk", {public = false})

        add_files("src/*.cpp")
        add_files("src/vk/**/*.cpp")

        add_packages("vulkan-hpp", "stb", "tinyobjloader", "imgui")

        add_deps("lpl-core", "lpl-math", "lpl-memory")

        -- Runtime asset paths: absolute at configure time so the app works
        -- regardless of the working directory when launched via `xmake run`.
        add_defines('LPL_SHADER_DIR="' .. os.projectdir() .. '/shaders/"')
        add_defines('LPL_ASSET_DIR="'  .. os.projectdir() .. '/assets/"')
    target_end()
else
    -- Provide a headeronly stub so that other modules can still compile
    target("lpl-render")
        set_kind("headeronly")
        set_group("modules")
        add_includedirs("include", {public = true})
        add_deps("lpl-core")
    target_end()
end
