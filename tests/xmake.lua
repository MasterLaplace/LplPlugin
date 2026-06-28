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

-- ─────────────────────────────────────────────────────────────────────────────
-- Image color/HSB/histogram/sampling determinism
-- ─────────────────────────────────────────────────────────────────────────────
target("test-image-parity")
    set_kind("binary")
    set_group("tests")
    add_deps("lpl-core", "lpl-image")
    add_files("parity/test_image_parity.cpp")
target_end()

-- ─────────────────────────────────────────────────────────────────────────────
-- Scene graph: transforms / world composition / undo-redo / selection
-- ─────────────────────────────────────────────────────────────────────────────
target("test-scene-parity")
    set_kind("binary")
    set_group("tests")
    add_deps("lpl-core", "lpl-math", "lpl-scene")
    add_files("parity/test_scene_parity.cpp")
target_end()

-- ─────────────────────────────────────────────────────────────────────────────
-- 3D camera/projection determinism (Fixed32 geometry → float projection)
-- ─────────────────────────────────────────────────────────────────────────────
target("test-render-parity")
    set_kind("binary")
    set_group("tests")
    add_deps("lpl-core", "lpl-math", "lpl-render")
    add_files("parity/test_render_parity.cpp")
target_end()
