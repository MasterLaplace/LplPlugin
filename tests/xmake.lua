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

target("test-procgen-passes")
    set_kind("binary")
    set_group("tests")
    add_deps("lpl-core", "lpl-math", "lpl-ecs", "lpl-procgen")
    add_files("parity/test_procgen_passes.cpp")
target_end()

target("test-procgen-climate")
    set_kind("binary")
    set_group("tests")
    add_deps("lpl-core", "lpl-math", "lpl-ecs", "lpl-procgen")
    add_files("parity/test_procgen_climate.cpp")
target_end()

target("test-procgen-shapegrammar")
    set_kind("binary")
    set_group("tests")
    add_deps("lpl-core", "lpl-math", "lpl-ecs", "lpl-procgen")
    add_files("parity/test_procgen_shapegrammar.cpp")
target_end()

target("test-procgen-liminal")
    set_kind("binary")
    set_group("tests")
    add_deps("lpl-core", "lpl-math", "lpl-ecs", "lpl-procgen")
    add_files("parity/test_procgen_liminal.cpp")
target_end()

target("test-procgen-caves")
    set_kind("binary")
    set_group("tests")
    add_deps("lpl-core", "lpl-math", "lpl-ecs", "lpl-procgen")
    add_files("parity/test_procgen_caves.cpp")
target_end()

target("test-procgen-streaming")
    set_kind("binary")
    set_group("tests")
    add_deps("lpl-core", "lpl-math", "lpl-ecs", "lpl-procgen")
    add_files("parity/test_procgen_streaming.cpp")
target_end()

target("test-procgen-higen")
    set_kind("binary")
    set_group("tests")
    add_deps("lpl-core", "lpl-math", "lpl-ecs", "lpl-procgen")
    add_files("parity/test_procgen_higen.cpp")
target_end()

target("test-heightfield-collision")
    set_kind("binary")
    set_group("tests")
    add_deps("lpl-core", "lpl-math", "lpl-ecs", "lpl-procgen", "lpl-engine")
    add_files("parity/test_heightfield_collision.cpp")
target_end()

target("test-map-mesh")
    set_kind("binary")
    set_group("tests")
    add_deps("lpl-core", "lpl-math", "lpl-ecs", "lpl-procgen")
    add_files("parity/test_map_mesh.cpp")
target_end()

target("test-ai")
    set_kind("binary")
    set_group("tests")
    add_deps("lpl-core", "lpl-math", "lpl-ecs", "lpl-procgen", "lpl-ai")
    add_files("parity/test_ai.cpp")
target_end()

target("test-ecology")
    set_kind("binary")
    set_group("tests")
    add_deps("lpl-core", "lpl-math", "lpl-ecs", "lpl-procgen", "lpl-ai", "lpl-ecology")
    add_files("parity/test_ecology.cpp")
target_end()

target("test-procgen-structures")
    set_kind("binary")
    set_group("tests")
    add_deps("lpl-core", "lpl-math", "lpl-ecs", "lpl-procgen")
    add_files("parity/test_procgen_structures.cpp")
target_end()

target("test-procgen-review")
    set_kind("binary")
    set_group("tests")
    add_deps("lpl-core", "lpl-math", "lpl-ecs", "lpl-procgen")
    add_files("parity/test_procgen_review.cpp")
target_end()

target("test-world-recipe")
    set_kind("binary")
    set_group("tests")
    add_deps("lpl-core", "lpl-math", "lpl-ecs", "lpl-procgen")
    add_files("parity/test_world_recipe.cpp")
target_end()

target("test-game-document")
    set_kind("binary")
    set_group("tests")
    add_deps("lpl-core", "lpl-math", "lpl-ecs", "lpl-procgen", "lpl-editor")
    add_files("parity/test_game_document.cpp")
target_end()

target("test-command-journal")
    set_kind("binary")
    set_group("tests")
    add_deps("lpl-core", "lpl-math", "lpl-ecs", "lpl-procgen", "lpl-editor")
    add_files("parity/test_command_journal.cpp")
target_end()

target("test-game-pack")
    set_kind("binary")
    set_group("tests")
    add_deps("lpl-core", "lpl-math", "lpl-ecs", "lpl-procgen", "lpl-pack", "lpl-editor", "lpl-ai", "lpl-ecology")
    add_files("parity/test_game_pack.cpp")
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
    -- lpl-agent because the JSON-Schema emitter this test pins now lives there,
    -- where its second consumer (the GBNF grammar) is. One emitter, two callers.
    add_deps("lpl-core", "lpl-math", "lpl-ecs", "lpl-agent")
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

target("test-living-parity")
    set_kind("binary")
    set_group("tests")
    add_deps("lpl-core", "lpl-math", "lpl-ecs", "lpl-procgen", "lpl-ai", "lpl-ecology")
    add_files("parity/test_living_parity.cpp")
target_end()

target("test-prop-materialization")
    set_kind("binary")
    set_group("tests")
    add_deps("lpl-core", "lpl-math", "lpl-ecs", "lpl-procgen")
    add_files("parity/test_prop_materialization.cpp")
target_end()

target("test-draw-order")
    set_kind("binary")
    set_group("tests")
    add_deps("lpl-core", "lpl-math", "lpl-render")
    add_files("parity/test_draw_order.cpp")
target_end()

target("test-botany-parity")
    set_kind("binary")
    set_group("tests")
    add_deps("lpl-core", "lpl-math", "lpl-procgen")
    add_files("parity/test_botany_parity.cpp")
target_end()

target("test-procgen-chunking")
    set_kind("binary")
    set_group("tests")
    add_deps("lpl-core", "lpl-math", "lpl-procgen")
    add_files("parity/test_procgen_chunking.cpp")
target_end()

-- ─────────────────────────────────────────────────────────────────────────────
-- Octree: the hierarchy must agree with its own node bounds (cull invariant)
-- ─────────────────────────────────────────────────────────────────────────────
target("test-octree-cull")
    set_kind("binary")
    set_group("tests")
    add_deps("lpl-core", "lpl-math", "lpl-physics")
    add_files("parity/test_octree_cull.cpp")
target_end()

-- ─────────────────────────────────────────────────────────────────────────────
-- Diffuse image-based lighting: the ambient term must depend on the normal
-- ─────────────────────────────────────────────────────────────────────────────
target("test-irradiance")
    set_kind("binary")
    set_group("tests")
    add_deps("lpl-core", "lpl-math", "lpl-render")
    add_files("parity/test_irradiance.cpp")
target_end()

-- ─────────────────────────────────────────────────────────────────────────────
-- View profile: a world's look as content, document -> pack -> engine
-- ─────────────────────────────────────────────────────────────────────────────
target("test-view-profile")
    set_kind("binary")
    set_group("tests")
    add_deps("lpl-core", "lpl-math", "lpl-procgen", "lpl-ecology", "lpl-pack", "lpl-editor", "lpl-engine")
    add_files("parity/test_view_profile.cpp")
target_end()

-- ─────────────────────────────────────────────────────────────────────────────
-- Character controller: gravity, ground, walls, slopes, jump — and determinism
-- ─────────────────────────────────────────────────────────────────────────────
target("test-character-controller")
    set_kind("binary")
    set_group("tests")
    add_deps("lpl-core", "lpl-math", "lpl-procgen", "lpl-engine")
    add_files("parity/test_character_controller.cpp")
target_end()

-- ─────────────────────────────────────────────────────────────────────────────
-- Gates for the new organs. Same rule as everywhere else in this file: assert a
-- property that can fail, not a signature that merely pins today's arithmetic.
--
-- A gate is declared here only once its source asserts something. A target whose
-- main() is `return 0` is worse than no target: validate.sh discovers it, runs
-- it, and the absence of an "ALL PASS" line makes the whole battery red — or, if
-- it were taught to print that line, it would become a check that cannot fail,
-- the anti-pattern this repo has already paid for twice. So the scaffolding
-- sources of codec/, history/ and rosetta/ exist under parity/ without a target,
-- and each one is declared by the batch that fills it.
-- ─────────────────────────────────────────────────────────────────────────────
target("test-agent-tools")
    set_kind("binary")
    set_group("tests")
    add_deps("lpl-core", "lpl-ecs", "lpl-procgen", "lpl-editor", "lpl-agent", "lpl-image")
    add_files("parity/test_agent_tools.cpp")
target_end()

target("test-herd-scent")
    set_kind("binary")
    set_group("tests")
    add_deps("lpl-core", "lpl-math", "lpl-ecs", "lpl-procgen", "lpl-ai", "lpl-ecology", "lpl-engine")
    add_files("parity/test_herd_scent.cpp")
target_end()

target("test-agent-loop")
    set_kind("binary")
    set_group("tests")
    add_deps("lpl-core", "lpl-ecs", "lpl-procgen", "lpl-editor", "lpl-agent", "lpl-engine")
    add_files("parity/test_agent_loop.cpp")
target_end()

target("test-codec-parity")
    set_kind("binary")
    set_group("tests")
    add_deps("lpl-core", "lpl-math", "lpl-codec")
    add_files("parity/test_codec_parity.cpp")
target_end()

target("test-pack-ecc")
    set_kind("binary")
    set_group("tests")
    add_deps("lpl-core", "lpl-math", "lpl-ecs", "lpl-codec", "lpl-pack", "lpl-editor", "lpl-procgen")
    add_files("parity/test_pack_ecc.cpp")
target_end()

target("test-erasure-channel")
    set_kind("binary")
    set_group("tests")
    add_deps("lpl-core", "lpl-math", "lpl-codec")
    add_files("parity/test_erasure_channel.cpp")
target_end()

target("test-rosetta-isa")
    set_kind("binary")
    set_group("tests")
    add_deps("lpl-core", "lpl-math", "lpl-codec", "lpl-rosetta")
    add_files("parity/test_rosetta_isa.cpp")
target_end()

target("test-history-parity")
    set_kind("binary")
    set_group("tests")
    add_deps("lpl-core", "lpl-math", "lpl-ecs", "lpl-history")
    add_files("parity/test_history_parity.cpp")
target_end()
