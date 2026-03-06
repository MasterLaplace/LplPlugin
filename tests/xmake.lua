-- /////////////////////////////////////////////////////////////////////////////
-- tests/ build configuration — Parity (determinism / regression) tests
-- /////////////////////////////////////////////////////////////////////////////

-- ─────────────────────────────────────────────────────────────────────────────
-- Fixed32 arithmetic parity
-- ─────────────────────────────────────────────────────────────────────────────
target("test-fixed32-parity")
    set_kind("binary")
    set_group("tests")
    add_deps("lpl-core", "lpl-math")
    add_files("parity/test_fixed32_parity.cpp")
target_end()

-- ─────────────────────────────────────────────────────────────────────────────
-- Morton encoding/decoding roundtrip
-- ─────────────────────────────────────────────────────────────────────────────
target("test-morton-parity")
    set_kind("binary")
    set_group("tests")
    add_deps("lpl-core", "lpl-math")
    add_files("parity/test_morton_parity.cpp")
target_end()

-- ─────────────────────────────────────────────────────────────────────────────
-- Physics integration determinism
-- ─────────────────────────────────────────────────────────────────────────────
target("test-physics-parity")
    set_kind("binary")
    set_group("tests")
    add_deps("lpl-core", "lpl-math", "lpl-ecs")
    add_files("parity/test_physics_parity.cpp")
target_end()
