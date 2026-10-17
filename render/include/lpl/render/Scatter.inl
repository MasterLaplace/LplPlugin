#ifndef LPL_RENDER_SCATTER_INL
#define LPL_RENDER_SCATTER_INL

namespace lpl::render {

inline core::u32 scatterHash(core::i32 cellX, core::i32 cellZ, core::u32 salt) noexcept
{
    core::u32 hash = 0x811C9DC5u ^ salt;
    hash = (hash ^ static_cast<core::u32>(cellX)) * 0x01000193u;
    hash = (hash ^ 0x9E3779B9u) * 0x01000193u;
    hash = (hash ^ static_cast<core::u32>(cellZ)) * 0x01000193u;
    return hash;
}

inline core::u32 ScatterQueue::flush(const RenderTarget &rt, const math::Mat4<core::f32> &mvp, const CameraBasis &basis,
                                     const FoliageMesh *meshes, core::u32 meshCount, const FoliageStyle &style,
                                     core::f32 farDistance, core::f32 nearFullDetail, core::f32 midDetail)
{
    const core::u32 count = static_cast<core::u32>(_instances.size());
    if (count == 0u || meshes == nullptr || meshCount == 0u)
        return 0u;

    _keys.clear();
    _keys.resize(count);
    _scratch.clear();
    _scratch.resize(count);

    for (core::u32 i = 0u; i < count; ++i)
    {
        const ScatterInstance &instance = _instances[i];
        const core::f32 distance = approximateLength(instance.worldX - basis.eye.x, instance.worldZ - basis.eye.z);
        const core::f32 clamped = distance < 0.0f ? 0.0f : (distance > farDistance ? farDistance : distance);
        const core::u32 depth = farDistance > 0.0f ? static_cast<core::u32>((clamped / farDistance) * 65535.0f) : 0u;
        _keys[i].key = packDrawKey(instance.mesh, 0u, depth);
        _keys[i].payload = i;
    }

    radixSortDrawKeys(&_keys[0], &_scratch[0], count);
    _orderFold = foldDrawKeys(&_keys[0], count);

    _poses.clear();
    _poses.resize(count);
    _recording.reset();
    for (core::u32 i = 0u; i < count; ++i)
    {
        const core::u32 slot = _keys[i].payload;
        const ScatterInstance &instance = _instances[slot];
        _poses[slot].x = math::Fixed32::fromFloat(instance.worldX);
        _poses[slot].y = math::Fixed32::fromFloat(instance.worldY);
        _poses[slot].z = math::Fixed32::fromFloat(instance.worldZ);
        _poses[slot].scale = math::Fixed32::fromFloat(instance.scale);

        DrawCommand packet;
        packet.materialId = instance.mesh;
        packet.poseSlot = slot;
        packet.instanceCount = 1u;
        packet.vertexBufferVA = static_cast<core::u64>(instance.mesh) + 1u;
        packet.indexCount = 0u;
        _recording.record(packet);
    }
    _recording.finalize();
    _latched = submitLateLatched(_recording, _poses.empty() ? nullptr : &_poses[0], count);

    core::u32 triangles = 0u;
    for (core::u32 i = 0u; i < _recording.count(); ++i)
    {
        const DrawCommand &packet = _recording.at(i);
        const ScatterInstance &instance = _instances[packet.poseSlot];
        if (packet.materialId >= meshCount)
            continue;

        const Pose &pose = _poses[packet.poseSlot];
        const core::f32 poseX = pose.x.toFloat();
        const core::f32 poseY = pose.y.toFloat();
        const core::f32 poseZ = pose.z.toFloat();

        const core::f32 distance = approximateLength(poseX - basis.eye.x, poseZ - basis.eye.z);

        FoliageStyle resolved = style;
        resolved.light = instance.light;
        resolved.maxDepth = distance < nearFullDetail ? 255u : (distance < midDetail ? 2u : 1u);

        FoliageInstance transform;
        transform.x = poseX;
        transform.y = poseY;
        transform.z = poseZ;
        transform.scale = pose.scale.toFloat();
        transform.yaw = instance.yaw;
        triangles += drawFoliage(rt, mvp, basis, meshes[packet.materialId], transform, resolved);
    }
    return triangles;
}

} // namespace lpl::render

#endif // LPL_RENDER_SCATTER_INL
