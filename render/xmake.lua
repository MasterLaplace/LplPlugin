-- /////////////////////////////////////////////////////////////////////////////
-- @file xmake.lua
-- @brief Build configuration for the lpl::render module.
-- /////////////////////////////////////////////////////////////////////////////

if has_config("renderer") then
    -- ─────────────────────────────────────────────────────────────────────────
    -- Shader compilation (phony target — GLSL → SPIR-V)
    -- ─────────────────────────────────────────────────────────────────────────
    target("lpl-shaders")
        set_kind("phony")
        set_group("shaders")
        on_build(function (target)
            import("core.project.config")

            local shaderSrcDir = path.join(os.scriptdir(), "src", "shaders")
            local shaderOutDir = path.join(os.projectdir(), "shaders")
            os.mkdir(shaderOutDir)

            local shaderFiles = {}
            for _, pat in ipairs({"*.vert", "*.frag"}) do
                table.join2(shaderFiles, os.files(path.join(shaderSrcDir, pat)))
            end

            for _, shaderFile in ipairs(shaderFiles) do
                local ext = path.extension(shaderFile):sub(2)  -- "vert" | "frag"
                local outputFile = path.join(shaderOutDir, ext .. ".spv")
                os.execv("glslangValidator", {"-V", shaderFile, "-o", outputFile})
            end
        end)
    target_end()

    -- ─────────────────────────────────────────────────────────────────────────
    -- Render module
    -- ─────────────────────────────────────────────────────────────────────────
    target("lpl-render")
        set_kind("static")
        set_group("modules")
        add_deps("lpl-shaders")  -- compile shaders before linking any consumer
        add_includedirs("include", {public = true})

        -- Expose the internal headers of the Vulkan implementation specifically to the render module itself
        add_includedirs("src/vk", {public = false})

        add_files("src/*.cpp")
        add_files("src/vk/**/*.cpp")

        add_packages("imgui", "vulkan-hpp", "vulkan-loader", "stb", "tinyobjloader", "glm")

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
