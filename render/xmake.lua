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
