/**
 * @file CommandBuffer.hpp
 * @brief Immutable pre-recorded command buffers + stable GPU virtual addresses,
 *        with Late-Latching of Fixed32 pose buffers at submit time.
 *
 * A command buffer is recorded ONCE (draw packets referencing stable resource
 * handles = "GPU VAs" that never relocate, mirroring the kernel HAL's
 * never-relocated mappings) and then replayed every frame. Per-draw transforms
 * are NOT baked into the packet: each packet carries a pose-buffer slot index,
 * and at submit the latest Fixed32 pose is re-fetched ("late-latched") so the
 * draw reflects the most recent simulation state without re-recording. Pose
 * authority is Fixed32; the folded packet stream + latched transforms are the
 * cross-target signature (the same recording + same poses fold identically on
 * the Linux oracle and the i686 kernel).
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-06-28
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_RENDER_COMMANDBUFFER_HPP
#    define LPL_RENDER_COMMANDBUFFER_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/math/FixedPoint.hpp>
#    include <lpl/render/RenderParity.hpp>
#    include <lpl/std/vector.hpp>

namespace lpl::render {

/** @brief One recorded draw: stable resource handles + a pose slot to latch. */
struct DrawCommand {
    core::u64 vertexBufferVA{0u}; ///< Stable GPU VA of the vertex buffer.
    core::u64 indexBufferVA{0u};  ///< Stable GPU VA of the index buffer.
    core::u32 indexCount{0u};     ///< Indices to draw.
    core::u32 instanceCount{1u};  ///< Instances.
    core::u32 poseSlot{0u};       ///< Index into the pose buffer (late-latched).
    core::u32 materialId{0u};     ///< Material/pipeline binding.
};

/** @brief A Fixed32 pose: position + uniform scale, authoritative, mutable. */
struct Pose {
    math::Fixed32 x{math::Fixed32::fromInt(0)};
    math::Fixed32 y{math::Fixed32::fromInt(0)};
    math::Fixed32 z{math::Fixed32::fromInt(0)};
    math::Fixed32 scale{math::Fixed32::fromInt(1)};
};

/**
 * @brief Pre-recorded, immutable command buffer.
 *
 * Recording appends draws; once finalize() is called the buffer is sealed and
 * subsequent record() calls are rejected. The packet list never changes per
 * frame — only the externally-owned pose buffer does.
 */
class CommandBuffer {
public:
    void record(const DrawCommand &cmd)
    {
        if (!_sealed)
            _commands.push_back(cmd);
    }

    void finalize() noexcept { _sealed = true; }
    [[nodiscard]] bool sealed() const noexcept { return _sealed; }
    [[nodiscard]] core::u32 count() const noexcept { return static_cast<core::u32>(_commands.size()); }
    [[nodiscard]] const DrawCommand &at(core::u32 i) const { return _commands[i]; }

    /** @brief FNV-1a fold of the immutable packet stream (recording identity). */
    [[nodiscard]] core::u32 recordingSignature() const noexcept
    {
        core::u32 hash = 0x811C9DC5u;
        for (core::u32 i = 0; i < count(); ++i)
        {
            const DrawCommand &c = _commands[i];
            hash = detail::fnv1aStep(hash, static_cast<core::u32>(c.vertexBufferVA));
            hash = detail::fnv1aStep(hash, static_cast<core::u32>(c.vertexBufferVA >> 32));
            hash = detail::fnv1aStep(hash, static_cast<core::u32>(c.indexBufferVA));
            hash = detail::fnv1aStep(hash, static_cast<core::u32>(c.indexBufferVA >> 32));
            hash = detail::fnv1aStep(hash, c.indexCount);
            hash = detail::fnv1aStep(hash, c.instanceCount);
            hash = detail::fnv1aStep(hash, c.poseSlot);
            hash = detail::fnv1aStep(hash, c.materialId);
        }
        return hash;
    }

private:
    pmr::vector<DrawCommand> _commands;
    bool _sealed{false};
};

/** @brief Result of submitting a command buffer against a pose buffer. */
struct SubmitResult {
    core::u32 draws{0u};            ///< Draws emitted.
    core::u32 latched_signature{0u}; ///< Fold of (packet, late-latched Fixed32 pose).
};

/**
 * @brief Submits an immutable command buffer, late-latching each draw's pose.
 *
 * For every packet the freshest Fixed32 pose at its slot is read NOW (after the
 * sim may have advanced) and folded with the packet — proving the recording is
 * reused verbatim while the transform tracks the latest authoritative state.
 *
 * @param cb     Sealed command buffer.
 * @param poses  Pose buffer (slot-indexed), authoritative Fixed32.
 * @param poseCount Number of pose slots.
 */
[[nodiscard]] inline SubmitResult submitLateLatched(const CommandBuffer &cb, const Pose *poses, core::u32 poseCount)
{
    SubmitResult out{};
    core::u32 hash = 0x811C9DC5u;
    for (core::u32 i = 0; i < cb.count(); ++i)
    {
        const DrawCommand &c = cb.at(i);
        hash = detail::fnv1aStep(hash, static_cast<core::u32>(c.vertexBufferVA));
        hash = detail::fnv1aStep(hash, c.indexCount * c.instanceCount);
        if (c.poseSlot < poseCount)
        {
            const Pose &p = poses[c.poseSlot]; // late latch: latest authoritative pose
            hash = detail::fnv1aStep(hash, static_cast<core::u32>(p.x.raw()));
            hash = detail::fnv1aStep(hash, static_cast<core::u32>(p.y.raw()));
            hash = detail::fnv1aStep(hash, static_cast<core::u32>(p.z.raw()));
            hash = detail::fnv1aStep(hash, static_cast<core::u32>(p.scale.raw()));
        }
        ++out.draws;
    }
    out.latched_signature = hash;
    return out;
}

} // namespace lpl::render

#endif // LPL_RENDER_COMMANDBUFFER_HPP
