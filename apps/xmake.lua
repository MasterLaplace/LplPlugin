-- /////////////////////////////////////////////////////////////////////////////
-- apps/ build configuration — Server, Client, Benchmark executables
-- /////////////////////////////////////////////////////////////////////////////

-- ─────────────────────────────────────────────────────────────────────────────
-- Server (headless)
-- ─────────────────────────────────────────────────────────────────────────────
target("lpl-server")
    set_kind("binary")
    set_group("apps")
    add_deps("lpl-engine", "lpl-samples")
    add_files("server/main.cpp")
target_end()

-- ─────────────────────────────────────────────────────────────────────────────
-- Client (desktop / VR)
-- ─────────────────────────────────────────────────────────────────────────────
target("lpl-client")
    set_kind("binary")
    set_group("apps")
    add_deps("lpl-engine")
    add_files("client/main.cpp")
target_end()

-- ─────────────────────────────────────────────────────────────────────────────
-- Editor (deterministic scene command REPL — human twin of the Caine AI bridge)
-- ─────────────────────────────────────────────────────────────────────────────
target("lpl-editor-cli")
    set_kind("binary")
    set_group("apps")
    add_deps("lpl-core", "lpl-math", "lpl-ecs", "lpl-editor", "lpl-procgen")
    add_files("editor/main.cpp")
target_end()

-- ─────────────────────────────────────────────────────────────────────────────
-- Worldforge (standalone OpenGL world editor — GLFW + legacy GL + imgui, all in
-- worldforge/main.cpp). Deliberately NOT wired to the engine's Vulkan renderer:
-- a throwaway immediate-mode viewport so world editing works today, reusing only
-- the renderer-agnostic logic (EditorSession / procgen / ecs / physics). Built
-- only when `--worldforge` is on so it never burdens the headless/kernel builds.
-- ─────────────────────────────────────────────────────────────────────────────
if has_config("worldforge") then
    add_requires("glfw 3.4", {system = false})
    add_requires("imgui", {alias = "imgui-gl", configs = {glfw = true, opengl2 = true}})

target("lpl-worldforge")
    set_kind("binary")
    set_group("apps")
    add_deps("lpl-core", "lpl-math", "lpl-ecs", "lpl-editor", "lpl-procgen", "lpl-physics")
    add_packages("glfw", "imgui-gl")
    if is_plat("linux") then
        add_syslinks("GL")
    elseif is_plat("windows") then
        add_syslinks("opengl32")
    elseif is_plat("macosx") then
        add_frameworks("OpenGL")
    end
    add_files("worldforge/main.cpp")
target_end()
end

-- ─────────────────────────────────────────────────────────────────────────────
-- Benchmark
-- ─────────────────────────────────────────────────────────────────────────────
target("lpl-benchmark")
    set_kind("binary")
    set_group("apps")
    add_deps(
        "lpl-bench",
        "lpl-core",
        "lpl-math",
        "lpl-memory",
        "lpl-container",
        "lpl-concurrency",
        "lpl-ecs",
        "lpl-physics",
        "lpl-net",
        "lpl-input",
        "lpl-engine"
    )
    add_files("benchmark/main.cpp")
target_end()
