#ifndef LPL_RENDER_COMMANDBUFFER_INL
#define LPL_RENDER_COMMANDBUFFER_INL

namespace lpl::render {

inline void CommandBuffer::record(const DrawCommand &cmd)
{
    if (!_sealed)
        _commands.push_back(cmd);
}

inline void CommandBuffer::reset() noexcept
{
    _commands.clear();
    _sealed = false;
}

inline core::u32 CommandBuffer::recordingSignature() const noexcept
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

inline SubmitResult submitLateLatched(const CommandBuffer &cb, const Pose *poses, core::u32 poseCount)
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
            const Pose &p = poses[c.poseSlot];
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

    if (source != keys)
        for (core::u32 i = 0u; i < count; ++i)
            keys[i] = source[i];
}

inline core::u32 foldDrawKeys(const DrawKey *keys, core::u32 count) noexcept
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

#endif // LPL_RENDER_COMMANDBUFFER_INL
