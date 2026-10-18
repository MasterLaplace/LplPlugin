#ifndef LPL_PROCGEN_LANDMARK_INL
#define LPL_PROCGEN_LANDMARK_INL

namespace lpl::procgen {

constexpr LandmarkParams caveMouthDefaults() noexcept
{
    LandmarkParams params{};
    params.cellSpan = 24u;
    params.oneIn = 2u;
    params.radius = 4u;
    params.clearanceAboveSea = 3.0f;
    params.minRelief = 2.2f;
    params.maxRelief = 1.0e9f;
    return params;
}

constexpr LandmarkParams settlementDefaults() noexcept
{
    LandmarkParams params{};
    params.cellSpan = 56u;
    params.oneIn = 2u;
    params.radius = 11u;
    params.clearanceAboveSea = 3.0f;
    params.maxHeight = 34.0f;
    params.minRelief = 0.0f;
    params.maxRelief = 4.0f;
    return params;
}

template <typename Emit>
void forEachLandmarkNear(const ChunkParams &params, const LandmarkParams &landmarks, LandmarkKind kind,
                         core::f32 seaLevel, ChunkCoord coord, Emit &&emit)
{
    if (params.size == 0u || landmarks.cellSpan == 0u)
        return;

    const core::i32 span = static_cast<core::i32>(landmarks.cellSpan);
    const core::i32 reach = static_cast<core::i32>(landmarks.radius);
    const core::i32 chunkCells = static_cast<core::i32>(params.size);
    const core::i32 minCellX = coord.x * chunkCells - reach;
    const core::i32 maxCellX = coord.x * chunkCells + chunkCells - 1 + reach;
    const core::i32 minCellZ = coord.z * chunkCells - reach;
    const core::i32 maxCellZ = coord.z * chunkCells + chunkCells - 1 + reach;

    const core::i32 fromX = floorDivChunk(minCellX, span);
    const core::i32 toX = floorDivChunk(maxCellX, span);
    const core::i32 fromZ = floorDivChunk(minCellZ, span);
    const core::i32 toZ = floorDivChunk(maxCellZ, span);

    for (core::i32 lz = fromZ; lz <= toZ; ++lz)
        for (core::i32 lx = fromX; lx <= toX; ++lx)
        {
            LandmarkSite site;
            if (!landmarkAt(params, landmarks, kind, seaLevel, lx, lz, site))
                continue;
            if (site.cellX + static_cast<core::i32>(site.radius) < minCellX ||
                site.cellX - static_cast<core::i32>(site.radius) > maxCellX ||
                site.cellZ + static_cast<core::i32>(site.radius) < minCellZ ||
                site.cellZ - static_cast<core::i32>(site.radius) > maxCellZ)
                continue;
            emit(site);
        }
}

template <typename Emit> void forEachVillageBuilding(const VillagePlan &plan, Emit &&emit)
{
    for (core::u32 i = 0u; i < plan.plots.size(); ++i)
    {
        const BuildingPlot &plot = plan.plots[i];
        if (plot.width == 0u || plot.depth == 0u)
            continue;

        LandmarkBuilding building;
        building.minX = static_cast<core::f32>(plan.originX + static_cast<core::i32>(plot.x));
        building.minZ = static_cast<core::f32>(plan.originZ + static_cast<core::i32>(plot.z));
        building.maxX = building.minX + static_cast<core::f32>(plot.width);
        building.maxZ = building.minZ + static_cast<core::f32>(plot.depth);
        building.baseY = plan.padHeight;
        const core::u32 area = plot.width * plot.depth;
        building.storeys = area >= 12u ? 1u : (area >= 6u ? 2u : 3u);
        building.height = 2.6f * static_cast<core::f32>(building.storeys);
        building.district = plot.district;
        emit(building);
    }
}

} // namespace lpl::procgen

#endif // LPL_PROCGEN_LANDMARK_INL
