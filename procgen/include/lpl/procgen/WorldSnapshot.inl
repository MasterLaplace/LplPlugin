#ifndef LPL_PROCGEN_WORLD_SNAPSHOT_INL
#define LPL_PROCGEN_WORLD_SNAPSHOT_INL

namespace lpl::procgen {

inline void captureSnapshot(const WorldBuilder &builder, const WalkabilityRule &rule, WorldSnapshot &out)
{
    out.gatePassed = builder.gatePassed();

    out.height = builder.heightfield();
    out.biomes = builder.biomeMap();
    out.moisture = builder.moisture();
    out.rivers = riverMask(builder.drainage(), 0.02f);
    out.settlement = builder.settlementMap();
    out.roads = builder.roadMap();

    out.width = out.height.width();
    out.depth = out.height.depth();
    (void) heightRange(out.height, out.lowest, out.highest);
    countBiomes(out.biomes, out.biomeCounts);

    out.blocked = Grid<core::u8>{out.width, out.depth, 0u};
    for (core::u32 z = 0u; z < out.depth; ++z)
        for (core::u32 x = 0u; x < out.width; ++x)
        {
            const bool drowned = out.height.at(x, z).toFloat() < rule.seaLevel;
            const bool steep = slopeAt(out.height, x, z).toFloat() > rule.maxSlope;
            out.blocked.at(x, z) = (drowned || steep) ? 1u : 0u;
        }
}

inline WorldSnapshot buildSnapshot(WorldRecipe recipe, ecs::Registry *registry,
                                   lpl::pmr::vector<ecs::EntityId> *outPropIds, const WalkabilityRule &rule,
                                   const ReliefBlend *relief)
{
    WorldSnapshot out;

    WorldBuilder builder{recipe.seed};
    const core::f32 lift = applyRecipe(builder, recipe, relief);
    WalkabilityRule lifted = rule;
    lifted.seaLevel += lift;

    if (registry != nullptr)
        out.stats = builder.materializeProps(*registry, outPropIds);

    captureSnapshot(builder, lifted, out);
    return out;
}

template <typename IsWooded, typename Emit>
core::u32 scatterVegetation(const WorldSnapshot &snapshot, core::u32 seed, core::u32 oneIn, IsWooded &&isWooded,
                            Emit &&emit)
{
    math::Random thin{seed};
    core::u32 planted = 0u;
    const core::u32 divisor = oneIn == 0u ? 1u : oneIn;
    for (core::u32 z = 0u; z < snapshot.depth; ++z)
        for (core::u32 x = 0u; x < snapshot.width; ++x)
        {
            if (!isWooded(snapshot.biomes.at(x, z)) || snapshot.blocked.at(x, z) != 0u)
                continue;
            if (divisor > 1u && thin.below(divisor) != 0u)
                continue;
            emit(static_cast<core::i32>(x), static_cast<core::i32>(z));
            ++planted;
        }
    return planted;
}

} // namespace lpl::procgen

#endif // LPL_PROCGEN_WORLD_SNAPSHOT_INL
