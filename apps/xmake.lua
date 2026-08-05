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
target("lpl-bake")
    set_kind("binary")
    set_group("apps")
    add_deps("lpl-core", "lpl-math", "lpl-ecs", "lpl-procgen", "lpl-pack", "lpl-editor")
    add_files("bake/main.cpp")
target_end()

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
    -- glfw/imgui are required once at the root, with the backend union.
target("lpl-worldforge")
    set_kind("binary")
    set_group("apps")
    -- lpl-agent + lpl-engine are here for the Caine panel: engine::DemonHost owns
    -- the generate/look/correct loop, and a second copy inside a panel would be the
    -- duplication this repository keeps paying for.
    add_deps("lpl-core", "lpl-math", "lpl-ecs", "lpl-editor", "lpl-procgen", "lpl-physics", "lpl-agent", "lpl-engine")
    add_packages("glfw", "imgui")
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

-- ─────────────────────────────────────────────────────────────────────────────
-- Map viewer (standalone X11 + GLX, no package dependencies)
--
-- Behind an option so the headless and kernel builds never see it: it links
-- libGL and libX11, which exist on a desktop and nowhere else.
-- ─────────────────────────────────────────────────────────────────────────────
if has_config("mapview") then
    target("lpl-mapview")
        set_kind("binary")
        set_group("apps")
        add_deps("lpl-engine", "lpl-procgen", "lpl-image", "lpl-ai", "lpl-ecology")
        add_files("mapview/main.cpp")
        add_syslinks("GL", "X11", "m")
    target_end()
end

-- ─────────────────────────────────────────────────────────────────────────────
-- The demon's tooling. Each of these exists because a claim in the plan needs an
-- instrument: a coding scheme judged without a wet lab, an artifact that proves
-- it carries its own reader, a world run across centuries, and the server profile
-- with a mind attached.
-- ─────────────────────────────────────────────────────────────────────────────
target("lpl-dna-lab")
    set_kind("binary")
    set_group("apps")
    add_deps("lpl-core", "lpl-math", "lpl-codec")
    add_files("dna-lab/main.cpp")
target_end()

target("lpl-rosetta-forge")
    set_kind("binary")
    set_group("apps")
    add_deps("lpl-core", "lpl-codec", "lpl-rosetta", "lpl-pack")
    add_files("rosetta-forge/main.cpp")
target_end()

target("lpl-chronicle")
    set_kind("binary")
    set_group("apps")
    add_deps("lpl-engine", "lpl-history", "lpl-ecology", "lpl-procgen")
    add_files("chronicle/main.cpp")
target_end()

target("lpl-demon")
    set_kind("binary")
    set_group("apps")
    -- lpl-editor for the scene serializer it writes with; lpl-history stays for
    -- the chronicle the demon will keep once that chantier lands.
    add_deps("lpl-engine", "lpl-agent", "lpl-editor", "lpl-history")
    add_files("demon/main.cpp")
target_end()
