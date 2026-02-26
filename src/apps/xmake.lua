-- /////////////////////////////////////////////////////////////////////////////
-- apps/ build configuration — Server, Client, Benchmark executables
-- /////////////////////////////////////////////////////////////////////////////

-- ─────────────────────────────────────────────────────────────────────────────
-- Server (headless)
-- ─────────────────────────────────────────────────────────────────────────────
target("lpl-server")
    set_kind("binary")
    set_group("apps")
    add_deps("lpl-engine")
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
-- Benchmark
-- ─────────────────────────────────────────────────────────────────────────────
target("lpl-benchmark")
    set_kind("binary")
    set_group("apps")
    add_deps(
        "lpl-core",
        "lpl-math",
        "lpl-memory",
        "lpl-container",
        "lpl-concurrency",
        "lpl-ecs",
        "lpl-physics"
    )
    add_files("benchmark/main.cpp")
target_end()
