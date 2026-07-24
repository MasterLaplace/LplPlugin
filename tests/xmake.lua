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
-- SPSC RingBuffer semantics (FIFO, boundaries, wraparound, move path)
-- ─────────────────────────────────────────────────────────────────────────────
target("test-ringbuffer-parity")
    set_kind("binary")
    set_group("tests")
    add_deps("lpl-core", "lpl-container")
    add_files("parity/test_ringbuffer_parity.cpp")
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

target("test-p6-parity")
    set_kind("binary")
    set_group("tests")
    add_deps("lpl-core", "lpl-math", "lpl-render")
    add_files("parity/test_p6_parity.cpp")
target_end()

target("test-cubepile-parity")
    set_kind("binary")
    set_group("tests")
    add_deps("lpl-core", "lpl-math", "lpl-render", "lpl-samples")
    add_files("parity/test_cubepile_parity.cpp")
target_end()

target("test-scene-document")
    set_kind("binary")
    set_group("tests")
    add_deps("lpl-core", "lpl-math", "lpl-ecs", "lpl-editor")
    add_files("parity/test_scene_document.cpp")
target_end()

target("test-procgen")
    set_kind("binary")
    set_group("tests")
    add_deps("lpl-core", "lpl-math", "lpl-ecs", "lpl-editor", "lpl-procgen")
    add_files("parity/test_procgen.cpp")
target_end()

target("test-scene-templates")
    set_kind("binary")
    set_group("tests")
    add_deps("lpl-core", "lpl-math", "lpl-ecs", "lpl-editor")
    add_files("parity/test_scene_templates.cpp")
target_end()

target("test-editor-commands")
    set_kind("binary")
    set_group("tests")
    add_deps("lpl-core", "lpl-math", "lpl-ecs", "lpl-editor", "lpl-procgen")
    add_files("parity/test_editor_commands.cpp")
target_end()

target("test-editor-session")
    set_kind("binary")
    set_group("tests")
    add_deps("lpl-core", "lpl-math", "lpl-ecs", "lpl-editor", "lpl-procgen")
    add_files("parity/test_editor_session.cpp")
target_end()

target("test-reflection")
    set_kind("binary")
    set_group("tests")
    add_deps("lpl-core", "lpl-math", "lpl-ecs")
    add_files("parity/test_reflection.cpp")
target_end()

target("test-simd-fixed-parity")
    set_kind("binary")
    set_group("tests")
    add_deps("lpl-core", "lpl-math")
    add_files("parity/test_simd_fixed_parity.cpp")
target_end()

-- ─────────────────────────────────────────────────────────────────────────────
-- Authoritative tick allocation audit (diagnostic, NOT a parity gate)
-- ─────────────────────────────────────────────────────────────────────────────
target("test-tick-allocations")
    set_kind("binary")
    set_group("tests")
    set_symbols("debug")
    add_deps("lpl-core", "lpl-math", "lpl-ecs", "lpl-physics", "lpl-engine", "lpl-samples", "lpl-image", "lpl-render")
    add_ldflags("-rdynamic", {force = true})
    add_files("parity/test_tick_allocations.cpp")
target_end()

target("test-server-routing")
    set_kind("binary")
    set_group("tests")
    add_deps("lpl-core", "lpl-math", "lpl-engine", "lpl-net")
    add_files("parity/test_server_routing.cpp")
target_end()

target("test-transport-batching")
    set_kind("binary")
    set_group("tests")
    add_deps("lpl-core", "lpl-net")
    add_files("parity/test_transport_batching.cpp")
target_end()

target("test-bitstream-quant")
    set_kind("binary")
    set_group("tests")
    add_deps("lpl-core", "lpl-net")
    add_files("parity/test_bitstream_quant.cpp")
target_end()

target("test-entity-delta")
    set_kind("binary")
    set_group("tests")
    add_deps("lpl-core", "lpl-net")
    add_files("parity/test_entity_delta.cpp")
target_end()

target("test-relevancy")
    set_kind("binary")
    set_group("tests")
    add_deps("lpl-core", "lpl-net")
    add_files("parity/test_relevancy.cpp")
target_end()

target("test-interpolation")
    set_kind("binary")
    set_group("tests")
    add_deps("lpl-core", "lpl-math", "lpl-net")
    add_files("parity/test_interpolation.cpp")
target_end()

target("test-lag-compensation")
    set_kind("binary")
    set_group("tests")
    add_deps("lpl-core", "lpl-math", "lpl-net")
    add_files("parity/test_lag_compensation.cpp")
target_end()

target("test-desync")
    set_kind("binary")
    set_group("tests")
    add_deps("lpl-core", "lpl-math", "lpl-engine", "lpl-net")
    add_files("parity/test_desync.cpp")
target_end()

target("test-server-mesh")
    set_kind("binary")
    set_group("tests")
    add_deps("lpl-core", "lpl-net")
    add_files("parity/test_server_mesh.cpp")
target_end()

target("test-config-profiles")
    set_kind("binary")
    set_group("tests")
    add_deps("lpl-core", "lpl-math", "lpl-engine", "lpl-net")
    add_files("parity/test_config_profiles.cpp")
target_end()

target("test-aoi")
    set_kind("binary")
    set_group("tests")
    add_deps("lpl-core", "lpl-math", "lpl-ecs", "lpl-engine", "lpl-net")
    add_files("parity/test_aoi.cpp")
target_end()

target("test-reconciliation")
    set_kind("binary")
    set_group("tests")
    add_deps("lpl-core", "lpl-math", "lpl-ecs", "lpl-engine", "lpl-net")
    add_files("parity/test_reconciliation.cpp")
target_end()

target("test-session-identity")
    set_kind("binary")
    set_group("tests")
    add_deps("lpl-core", "lpl-math", "lpl-ecs", "lpl-engine", "lpl-net", "lpl-input")
    add_files("parity/test_session_identity.cpp")
target_end()

target("test-session-lifecycle")
    set_kind("binary")
    set_group("tests")
    add_deps("lpl-core", "lpl-math", "lpl-ecs", "lpl-engine", "lpl-net", "lpl-input")
    add_files("parity/test_session_lifecycle.cpp")
target_end()
