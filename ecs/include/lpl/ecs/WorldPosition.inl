#ifndef LPL_ECS_WORLDPOSITION_INL
#define LPL_ECS_WORLDPOSITION_INL

namespace lpl::ecs {

constexpr void normaliseWorldPosition(WorldPosition &position, math::Fixed32 chunkSize) noexcept
{
    if (chunkSize <= math::Fixed32{})
        return;
    while (position.localX >= chunkSize)
    {
        position.localX = position.localX - chunkSize;
        ++position.chunkX;
    }
    while (position.localX < math::Fixed32{})
    {
        position.localX = position.localX + chunkSize;
        --position.chunkX;
    }
    while (position.localZ >= chunkSize)
    {
        position.localZ = position.localZ - chunkSize;
        ++position.chunkZ;
    }
    while (position.localZ < math::Fixed32{})
    {
        position.localZ = position.localZ + chunkSize;
        --position.chunkZ;
    }
}

constexpr void worldDelta(const WorldPosition &from, const WorldPosition &to, math::Fixed32 chunkSize,
                          core::i32 &outChunkX, core::i32 &outChunkZ, math::Fixed32 &outLocalX,
                          math::Fixed32 &outLocalZ) noexcept
{
    (void) chunkSize;
    outChunkX = to.chunkX - from.chunkX;
    outChunkZ = to.chunkZ - from.chunkZ;
    outLocalX = to.localX - from.localX;
    outLocalZ = to.localZ - from.localZ;
}

} // namespace lpl::ecs

#endif // LPL_ECS_WORLDPOSITION_INL
