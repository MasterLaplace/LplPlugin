#ifndef LPL_PROCGEN_WORLD_ATLAS_INL
#define LPL_PROCGEN_WORLD_ATLAS_INL

namespace lpl::procgen {

inline void captureAtlas(const WorldBuilder &builder, const WalkabilityRule &rule, WorldAtlas &out)
{
    captureSnapshot(builder, rule, out);

    out.drainage = builder.drainage();
    out.regions = builder.regionMap();
    out.climate = builder.climateField();
    out.dungeon = builder.dungeonMap();
    out.caveSystem = builder.caves();
    out.townVolume = builder.townVolume();
    out.roadsideVolume = builder.roadsideVolume();
    out.plots = builder.plots();
}

inline WorldAtlas buildAtlas(WorldRecipe recipe, ecs::Registry *registry, lpl::pmr::vector<ecs::EntityId> *outPropIds,
                             const WalkabilityRule &rule, const ReliefBlend *relief)
{
    WorldAtlas out;

    WorldBuilder builder{recipe.seed};
    const core::f32 lift = applyRecipe(builder, recipe, relief);
    WalkabilityRule lifted = rule;
    lifted.seaLevel += lift;

    out.stats = registry != nullptr ? builder.materializeProps(*registry, outPropIds) : builder.bakeGrids();
    captureAtlas(builder, lifted, out);
    out.seaLevel = recipe.biomes.seaLevel + lift;
    return out;
}

} // namespace lpl::procgen

#endif // LPL_PROCGEN_WORLD_ATLAS_INL
