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

    /**
     * @brief Unseals and empties the buffer, keeping every byte of its capacity.
     *
     * The header was written for a recording that outlives the frame, and that is
     * still the case it is best at. A scatter pass is the other case: WHAT is on
     * screen changes every frame, so the packet list is rebuilt — but rebuilt into
     * the same storage, so the recording path costs no allocation once warm and the
     * real-time contract survives it. Without this the only way to re-record was a
     * fresh buffer per frame, which is an allocation per frame.
     */
    void reset() noexcept
    {
        _commands.clear();
        _sealed = false;
    }
    [[nodiscard]] bool sealed() const noexcept { return _sealed; }
    [[nodiscard]] core::u32 count() const noexcept { return static_cast<core::u32>(_commands.size()); }
    [[nodiscard]] const DrawCommand &at(core::u32 i) const { return _commands[i]; }

    /** @brief FNV-1a fold of the immutable packet stream (recording identity). */
    [[nodiscard]] core::u32 recordingSignature() const noexcept
    {
        core::u32 hash = detail::kFnv1aOffsetBasis;
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
    core::u32 draws{0u};             ///< Draws emitted.
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
    core::u32 hash = detail::kFnv1aOffsetBasis;
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

/**
 * @struct DrawKey
 * @brief A sortable draw: a packed ordering key and the index it refers to.
 */
struct DrawKey {
    core::u32 key{0u};     ///< Packed (material, mesh, depth) — see @ref packDrawKey.
    core::u32 payload{0u}; ///< Index into whatever list the caller is ordering.
};

/**
 * @brief Packs a draw ordering key: material major, mesh next, depth minor.
 *
 * The order of the fields IS the policy, and it is the one every renderer
 * converges on: changing material is the expensive switch, so all draws sharing
 * one material must be adjacent; within a material, sharing a mesh saves the
 * vertex work; and within both, near-to-far lets the depth buffer reject a pixel
 * before it is shaded rather than after.
 *
 * @param materialId 0..255.
 * @param meshId     0..255.
 * @param depth      0..65535, near to far.
 */
[[nodiscard]] inline core::u32 packDrawKey(core::u32 materialId, core::u32 meshId, core::u32 depth) noexcept
{
    return ((materialId & 0xFFu) << 24) | ((meshId & 0xFFu) << 16) | (depth & 0xFFFFu);
}

/**
 * @brief Stable LSD radix sort of draw keys: four passes of eight bits.
 *
 * Chosen over a comparison sort for the reason radix always wins here: the key is
 * a fixed-width integer, so ordering it needs no comparisons at all — four counting
 * passes touch each element four times regardless of how unsorted it was. An
 * insertion sort is better on a nearly-sorted list of thirty; at a thousand draws
 * that stops being true, and a frame that submits a tree per cell has thousands.
 *
 * Stability matters and is not free: with material in the high bits, a stable sort
 * keeps draws that tie on the whole key in submission order, which is what makes
 * the resulting stream reproducible — and therefore foldable.
 *
 * @param keys    Array to sort in place.
 * @param scratch Scratch of the same length.
 * @param count   Elements.
 */
inline void radixSortDrawKeys(DrawKey *keys, DrawKey *scratch, core::u32 count) noexcept
{
    if (keys == nullptr || scratch == nullptr || count < 2u)
        return;

    DrawKey *source = keys;
    DrawKey *destination = scratch;

    for (core::u32 shift = 0u; shift < 32u; shift += 8u)
    {
        core::u32 histogram[256] = {};
        for (core::u32 i = 0u; i < count; ++i)
            ++histogram[(source[i].key >> shift) & 0xFFu];

        core::u32 running = 0u;
        for (core::u32 bucket = 0u; bucket < 256u; ++bucket)
        {
            const core::u32 here = histogram[bucket];
            histogram[bucket] = running;
            running += here;
        }

        for (core::u32 i = 0u; i < count; ++i)
            destination[histogram[(source[i].key >> shift) & 0xFFu]++] = source[i];

        DrawKey *swap = source;
        source = destination;
        destination = swap;
    }

    // Four passes is an even number, so the sorted data is back in @p keys. If
    // that ever changes, this copy is what keeps the contract true.
    if (source != keys)
        for (core::u32 i = 0u; i < count; ++i)
            keys[i] = source[i];
}

/** @brief FNV-1a fold of an ordered key stream: the order itself is the signature. */
[[nodiscard]] inline core::u32 foldDrawKeys(const DrawKey *keys, core::u32 count) noexcept
{
    core::u32 hash = detail::kFnv1aOffsetBasis;
    for (core::u32 i = 0u; i < count; ++i)
    {
        hash = detail::fnv1aStep(hash, keys[i].key);
        hash = detail::fnv1aStep(hash, keys[i].payload);
    }
    return hash;
}

} // namespace lpl::render

#endif // LPL_RENDER_COMMANDBUFFER_HPP
