#ifndef LPL_PROCGEN_MAP_MESH_INL
#define LPL_PROCGEN_MAP_MESH_INL

namespace lpl::procgen {

inline void appendQuad(MapMesh &mesh, float ax, float ay, float az, float bx, float by, float bz, float cx, float cy,
                       float cz, float dx, float dy, float dz, float nx, float ny, float nz, const Rgb &colour)
{
    const MapVertex a{ax, ay, az, nx, ny, nz, colour};
    const MapVertex b{bx, by, bz, nx, ny, nz, colour};
    const MapVertex c{cx, cy, cz, nx, ny, nz, colour};
    const MapVertex d{dx, dy, dz, nx, ny, nz, colour};
    mesh.push_back(a);
    mesh.push_back(b);
    mesh.push_back(c);
    mesh.push_back(a);
    mesh.push_back(c);
    mesh.push_back(d);
}

inline void appendTriangle(MapMesh &mesh, float ax, float ay, float az, float bx, float by, float bz, float cx,
                           float cy, float cz, float nx, float ny, float nz, const Rgb &colour)
{
    mesh.push_back(MapVertex{ax, ay, az, nx, ny, nz, colour});
    mesh.push_back(MapVertex{bx, by, bz, nx, ny, nz, colour});
    mesh.push_back(MapVertex{cx, cy, cz, nx, ny, nz, colour});
}

inline MapMesh buildSurfaceMesh(const WorldAtlas &atlas, const MapSurfaceStyle &style)
{
    MapMesh mesh;
    if (atlas.height.empty())
        return mesh;

    const core::u32 width = atlas.height.width();
    const core::u32 depth = atlas.height.depth();
    const float halfW = static_cast<float>(width) * 0.5f;
    const float halfD = static_cast<float>(depth) * 0.5f;

    lpl::pmr::vector<MapVertex> grid(static_cast<core::usize>(width) * depth);
    for (core::u32 z = 0u; z < depth; ++z)
    {
        for (core::u32 x = 0u; x < width; ++x)
        {
            const core::i32 ix = static_cast<core::i32>(x);
            const core::i32 iz = static_cast<core::i32>(z);
            const float left = atlas.height.clamped(ix - 1, iz).toFloat();
            const float right = atlas.height.clamped(ix + 1, iz).toFloat();
            const float back = atlas.height.clamped(ix, iz - 1).toFloat();
            const float front = atlas.height.clamped(ix, iz + 1).toFloat();

            float nx = left - right;
            float ny = 2.0f;
            float nz = back - front;
            const float length = lpl::pmr::sqrt(nx * nx + ny * ny + nz * nz);
            if (length > 0.0f)
            {
                nx /= length;
                ny /= length;
                nz /= length;
            }

            Rgb colour = surfaceColour(atlas, style.shading, style.climateAxis, x, z);
            if (style.rivers && !atlas.rivers.empty() && atlas.rivers.at(x, z) != 0u)
                colour = {0.16f, 0.42f, 0.72f};
            if (style.settlement && !atlas.settlement.empty())
            {
                switch (atlas.settlement.at(x, z))
                {
                case SettlementCell::Road: colour = {0.35f, 0.32f, 0.30f}; break;
                case SettlementCell::Plaza: colour = {0.55f, 0.50f, 0.44f}; break;
                case SettlementCell::Plot: colour = {0.82f, 0.55f, 0.22f}; break;
                default: break;
                }
            }
            if (style.roads && !atlas.roads.empty() && atlas.roads.at(x, z) != 0u)
                colour = {0.24f, 0.22f, 0.20f};

            grid[atlas.height.index(x, z)] = MapVertex{static_cast<float>(x) - halfW,
                                                       atlas.height.at(x, z).toFloat(),
                                                       static_cast<float>(z) - halfD,
                                                       nx,
                                                       ny,
                                                       nz,
                                                       colour};
        }
    }

    mesh.reserve(static_cast<core::usize>(width - 1u) * (depth - 1u) * 6u);
    for (core::u32 z = 0u; z + 1u < depth; ++z)
    {
        for (core::u32 x = 0u; x + 1u < width; ++x)
        {
            const MapVertex &a = grid[atlas.height.index(x, z)];
            const MapVertex &b = grid[atlas.height.index(x + 1u, z)];
            const MapVertex &c = grid[atlas.height.index(x + 1u, z + 1u)];
            const MapVertex &d = grid[atlas.height.index(x, z + 1u)];
            mesh.push_back(a);
            mesh.push_back(b);
            mesh.push_back(c);
            mesh.push_back(a);
            mesh.push_back(c);
            mesh.push_back(d);
        }
    }
    return mesh;
}

inline MapMesh buildDungeonMesh(const WorldAtlas &atlas, float depthBelow)
{
    MapMesh mesh;
    if (atlas.dungeon.empty())
        return mesh;

    const core::u32 width = atlas.dungeon.width();
    const core::u32 depth = atlas.dungeon.depth();
    const float halfW = static_cast<float>(width) * 0.5f;
    const float halfD = static_cast<float>(depth) * 0.5f;
    const float wallHeight = 1.6f;

    const Rgb floorColour{0.52f, 0.44f, 0.36f};
    const Rgb wallColour{0.30f, 0.26f, 0.24f};
    const Rgb capColour{0.66f, 0.44f, 0.20f};

    const auto solid = [&atlas](core::i32 x, core::i32 z) {
        if (!atlas.dungeon.contains(x, z))
            return true;
        return atlas.dungeon.at(static_cast<core::u32>(x), static_cast<core::u32>(z)) == DungeonCell::Wall;
    };

    for (core::u32 z = 0u; z < depth; ++z)
    {
        for (core::u32 x = 0u; x < width; ++x)
        {
            const core::i32 ix = static_cast<core::i32>(x);
            const core::i32 iz = static_cast<core::i32>(z);
            if (solid(ix, iz))
                continue;

            const float ground = atlas.height.empty() ? 0.0f : atlas.height.clamped(ix, iz).toFloat();
            const float floorY = ground - depthBelow;
            const float ceilY = floorY + wallHeight;

            const float x0 = static_cast<float>(x) - halfW;
            const float x1 = x0 + 1.0f;
            const float z0 = static_cast<float>(z) - halfD;
            const float z1 = z0 + 1.0f;

            appendQuad(mesh, x0, floorY, z0, x1, floorY, z0, x1, floorY, z1, x0, floorY, z1, 0.0f, 1.0f, 0.0f,
                       floorColour);

            if (solid(ix + 1, iz))
                appendQuad(mesh, x1, floorY, z0, x1, ceilY, z0, x1, ceilY, z1, x1, floorY, z1, 1.0f, 0.0f, 0.0f,
                           wallColour);
            if (solid(ix - 1, iz))
                appendQuad(mesh, x0, floorY, z0, x0, ceilY, z0, x0, ceilY, z1, x0, floorY, z1, -1.0f, 0.0f, 0.0f,
                           wallColour);
            if (solid(ix, iz + 1))
                appendQuad(mesh, x0, floorY, z1, x0, ceilY, z1, x1, ceilY, z1, x1, floorY, z1, 0.0f, 0.0f, 1.0f,
                           wallColour);
            if (solid(ix, iz - 1))
                appendQuad(mesh, x0, floorY, z0, x0, ceilY, z0, x1, ceilY, z0, x1, floorY, z0, 0.0f, 0.0f, -1.0f,
                           wallColour);

            if (solid(ix + 1, iz) || solid(ix - 1, iz) || solid(ix, iz + 1) || solid(ix, iz - 1))
                appendQuad(mesh, x0, ceilY, z0, x1, ceilY, z0, x1, ceilY, z1, x0, ceilY, z1, 0.0f, 1.0f, 0.0f,
                           capColour);
        }
    }
    return mesh;
}

inline MapMesh buildTownMesh(const WorldAtlas &atlas)
{
    MapMesh mesh;
    if (atlas.height.empty())
        return mesh;

    const float halfW = static_cast<float>(atlas.height.width()) * 0.5f;
    const float halfD = static_cast<float>(atlas.height.depth()) * 0.5f;

    const auto box = [&mesh](float x0, float x1, float y0, float y1, float z0, float z1, const Rgb &wall,
                             const Rgb &roof) {
        appendQuad(mesh, x0, y1, z0, x1, y1, z0, x1, y1, z1, x0, y1, z1, 0.0f, 1.0f, 0.0f, roof);
        appendQuad(mesh, x0, y0, z1, x0, y1, z1, x1, y1, z1, x1, y0, z1, 0.0f, 0.0f, 1.0f, wall);
        appendQuad(mesh, x1, y0, z0, x1, y1, z0, x0, y1, z0, x0, y0, z0, 0.0f, 0.0f, -1.0f, wall);
        appendQuad(mesh, x1, y0, z1, x1, y1, z1, x1, y1, z0, x1, y0, z0, 1.0f, 0.0f, 0.0f, wall);
        appendQuad(mesh, x0, y0, z0, x0, y1, z0, x0, y1, z1, x0, y0, z1, -1.0f, 0.0f, 0.0f, wall);
    };

    for (core::usize p = 0u; p < atlas.plots.size(); ++p)
    {
        const BuildingPlot &plot = atlas.plots[p];

        float lowest = 1.0e9f;
        for (core::u32 z = plot.z; z < plot.z + plot.depth; ++z)
            for (core::u32 x = plot.x; x < plot.x + plot.width; ++x)
            {
                const float h = atlas.height.clamped(static_cast<core::i32>(x), static_cast<core::i32>(z)).toFloat();
                if (h < lowest)
                    lowest = h;
            }
        if (lowest > 1.0e8f)
            continue;

        const core::u32 area = plot.width * plot.depth;
        const core::u32 h = ValueNoise2D::hash2(static_cast<core::i32>(plot.x), static_cast<core::i32>(plot.z),
                                                static_cast<core::u32>(plot.district) + 1u);
        const float storeys = 1.0f + static_cast<float>(area) * 0.28f + static_cast<float>(h & 0x7u) * 0.55f;

        const float x0 = static_cast<float>(plot.x) - halfW + 0.12f;
        const float x1 = static_cast<float>(plot.x + plot.width) - halfW - 0.12f;
        const float z0 = static_cast<float>(plot.z) - halfD + 0.12f;
        const float z1 = static_cast<float>(plot.z + plot.depth) - halfD - 0.12f;
        const float y0 = lowest - 0.4f;
        const float y1 = lowest + storeys;

        const float tint = static_cast<float>((h >> 8) & 0x3Fu) / 63.0f;
        const Rgb wall{0.58f + 0.22f * tint, 0.48f + 0.16f * tint, 0.38f + 0.10f * tint};
        const Rgb roof{0.34f + 0.10f * tint, 0.22f + 0.06f * tint, 0.20f};
        box(x0, x1, y0, y1, z0, z1, wall, roof);
    }
    return mesh;
}

template <typename GroundAt>
void appendVoxelFaces(MapMesh &mesh, const VoxelVolume &volume, float originX, float originZ, GroundAt &&groundAt,
                      float baseLift, const Rgb *palette, core::u32 paletteSize)
{
    if (volume.empty() || palette == nullptr || paletteSize == 0u)
        return;

    forEachVoxelFace(
        volume, VoxelWindow{}, [](core::i32, core::i32, core::i32) { return false; },
        [&](const float quad[12], float nx, float ny, float nz, core::u8 material, core::u32 x, core::u32,
            core::u32 z) {
            const Rgb colour = palette[material < paletteSize ? material : 0u];
            const float lift = groundAt(x, z) + baseLift;
            float world[12];
            for (core::u32 v = 0u; v < 4u; ++v)
            {
                world[v * 3u] = originX + quad[v * 3u];
                world[v * 3u + 1u] = lift + quad[v * 3u + 1u];
                world[v * 3u + 2u] = originZ + quad[v * 3u + 2u];
            }
            appendQuad(mesh, world[0], world[1], world[2], world[3], world[4], world[5], world[6], world[7], world[8],
                       world[9], world[10], world[11], nx, ny, nz, colour);
        });
}

inline MapMesh buildVoxelMesh(const VoxelVolume &volume, const WorldAtlas &atlas, float baseLift, const Rgb *palette,
                              core::u32 paletteSize, const float *datum, core::usize datumCount)
{
    MapMesh mesh;
    if (volume.empty() || atlas.height.empty())
        return mesh;

    const float halfW = static_cast<float>(atlas.height.width()) * 0.5f;
    const float halfD = static_cast<float>(atlas.height.depth()) * 0.5f;

    appendVoxelFaces(
        mesh, volume, -halfW, -halfD,
        [&](core::u32 x, core::u32 z) {
            const core::usize index = static_cast<core::usize>(z) * volume.width + x;
            return datum != nullptr && index < datumCount ?
                       datum[index] :
                       atlas.height.clamped(static_cast<core::i32>(x), static_cast<core::i32>(z)).toFloat();
        },
        baseLift, palette, paletteSize);
    return mesh;
}

inline MapMesh buildCaveSystemMesh(const WorldAtlas &atlas, float topDepth, float layerSpacing)
{
    MapMesh mesh;
    const CaveSystem &system = atlas.caveSystem;
    if (system.layerCount == 0u || atlas.height.empty())
        return mesh;

    const float halfW = static_cast<float>(atlas.height.width()) * 0.5f;
    const float halfD = static_cast<float>(atlas.height.depth()) * 0.5f;

    for (core::u32 layer = 0u; layer < system.layerCount; ++layer)
    {
        const DungeonMap &plan = system.layer[layer];
        if (plan.empty())
            continue;

        const float shade = 1.0f - 0.22f * static_cast<float>(layer);
        const Rgb floorColour{0.52f * shade, 0.44f * shade, 0.36f * shade};
        const Rgb wallColour{0.30f * shade, 0.26f * shade, 0.24f * shade};

        const auto rock = [&plan](core::i32 x, core::i32 z) {
            if (!plan.contains(x, z))
                return true;
            return !isWalkable(plan.at(static_cast<core::u32>(x), static_cast<core::u32>(z)));
        };

        for (core::u32 z = 0u; z < plan.depth(); ++z)
            for (core::u32 x = 0u; x < plan.width(); ++x)
            {
                const core::i32 ix = static_cast<core::i32>(x);
                const core::i32 iz = static_cast<core::i32>(z);
                if (rock(ix, iz))
                    continue;

                const float ground = atlas.height.clamped(ix, iz).toFloat();
                const float floorY = ground - topDepth - static_cast<float>(layer) * layerSpacing;
                const float ceilY = floorY + 1.6f;
                const float x0 = static_cast<float>(x) - halfW;
                const float x1 = x0 + 1.0f;
                const float z0 = static_cast<float>(z) - halfD;
                const float z1 = z0 + 1.0f;

                appendQuad(mesh, x0, floorY, z0, x1, floorY, z0, x1, floorY, z1, x0, floorY, z1, 0.0f, 1.0f, 0.0f,
                           floorColour);
                if (rock(ix + 1, iz))
                    appendQuad(mesh, x1, floorY, z0, x1, ceilY, z0, x1, ceilY, z1, x1, floorY, z1, 1.0f, 0.0f, 0.0f,
                               wallColour);
                if (rock(ix - 1, iz))
                    appendQuad(mesh, x0, floorY, z0, x0, ceilY, z0, x0, ceilY, z1, x0, floorY, z1, -1.0f, 0.0f, 0.0f,
                               wallColour);
                if (rock(ix, iz + 1))
                    appendQuad(mesh, x0, floorY, z1, x0, ceilY, z1, x1, ceilY, z1, x1, floorY, z1, 0.0f, 0.0f, 1.0f,
                               wallColour);
                if (rock(ix, iz - 1))
                    appendQuad(mesh, x0, floorY, z0, x0, ceilY, z0, x1, ceilY, z0, x1, floorY, z0, 0.0f, 0.0f, -1.0f,
                               wallColour);
            }
    }

    for (core::u32 i = 0u; i < system.shafts.size(); ++i)
    {
        const CaveShaft &shaft = system.shafts[i];
        const float ground =
            atlas.height.clamped(static_cast<core::i32>(shaft.x), static_cast<core::i32>(shaft.z)).toFloat();
        const float upper =
            shaft.surface ? ground + 0.4f : ground - topDepth - static_cast<float>(shaft.upperLayer) * layerSpacing;
        const float lower =
            ground - topDepth - static_cast<float>(shaft.surface ? shaft.upperLayer : shaft.lowerLayer) * layerSpacing;
        const Rgb colour = shaft.surface ? Rgb{0.95f, 0.62f, 0.14f} : Rgb{0.40f, 0.34f, 0.30f};

        const float x0 = static_cast<float>(shaft.x) - halfW + 0.25f;
        const float x1 = x0 + 0.5f;
        const float z0 = static_cast<float>(shaft.z) - halfD + 0.25f;
        const float z1 = z0 + 0.5f;
        appendQuad(mesh, x0, lower, z0, x0, upper, z0, x1, upper, z0, x1, lower, z0, 0.0f, 0.0f, -1.0f, colour);
        appendQuad(mesh, x1, lower, z1, x1, upper, z1, x0, upper, z1, x0, lower, z1, 0.0f, 0.0f, 1.0f, colour);
        appendQuad(mesh, x1, lower, z0, x1, upper, z0, x1, upper, z1, x1, lower, z1, 1.0f, 0.0f, 0.0f, colour);
        appendQuad(mesh, x0, lower, z1, x0, upper, z1, x0, upper, z0, x0, lower, z0, -1.0f, 0.0f, 0.0f, colour);
    }
    return mesh;
}

inline Rgb liminalZoneColour(LiminalZone zone) noexcept
{
    switch (zone)
    {
    case LiminalZone::Corridor: return {0.76f, 0.72f, 0.55f};
    case LiminalZone::Office: return {0.82f, 0.79f, 0.62f};
    case LiminalZone::Hall: return {0.70f, 0.68f, 0.58f};
    case LiminalZone::Pool: return {0.55f, 0.72f, 0.74f};
    case LiminalZone::Count: break;
    }
    return {0.7f, 0.7f, 0.7f};
}

inline MapMesh buildLiminalMesh(const LiminalSpace &space)
{
    MapMesh mesh;
    if (space.map.empty())
        return mesh;

    const core::u32 width = space.map.width();
    const core::u32 depth = space.map.depth();
    const float halfW = static_cast<float>(width) * 0.5f;
    const float halfD = static_cast<float>(depth) * 0.5f;
    const float wallHeight = 2.6f;

    const auto rock = [&space](core::i32 x, core::i32 z) {
        if (!space.map.contains(x, z))
            return true;
        return !isWalkable(space.map.at(static_cast<core::u32>(x), static_cast<core::u32>(z)));
    };

    for (core::u32 z = 0u; z < depth; ++z)
        for (core::u32 x = 0u; x < width; ++x)
        {
            const core::i32 ix = static_cast<core::i32>(x);
            const core::i32 iz = static_cast<core::i32>(z);
            if (rock(ix, iz))
                continue;
            const Rgb floorColour = liminalZoneColour(space.zones.at(x, z));
            const Rgb wallColour{floorColour.r * 0.78f, floorColour.g * 0.78f, floorColour.b * 0.70f};

            const float x0 = static_cast<float>(x) - halfW;
            const float x1 = x0 + 1.0f;
            const float z0 = static_cast<float>(z) - halfD;
            const float z1 = z0 + 1.0f;

            appendQuad(mesh, x0, 0.0f, z0, x1, 0.0f, z0, x1, 0.0f, z1, x0, 0.0f, z1, 0.0f, 1.0f, 0.0f, floorColour);
            if (rock(ix + 1, iz))
                appendQuad(mesh, x1, 0.0f, z0, x1, wallHeight, z0, x1, wallHeight, z1, x1, 0.0f, z1, 1.0f, 0.0f, 0.0f,
                           wallColour);
            if (rock(ix - 1, iz))
                appendQuad(mesh, x0, 0.0f, z0, x0, wallHeight, z0, x0, wallHeight, z1, x0, 0.0f, z1, -1.0f, 0.0f, 0.0f,
                           wallColour);
            if (rock(ix, iz + 1))
                appendQuad(mesh, x0, 0.0f, z1, x0, wallHeight, z1, x1, wallHeight, z1, x1, 0.0f, z1, 0.0f, 0.0f, 1.0f,
                           wallColour);
            if (rock(ix, iz - 1))
                appendQuad(mesh, x0, 0.0f, z0, x0, wallHeight, z0, x1, wallHeight, z0, x1, 0.0f, z0, 0.0f, 0.0f, -1.0f,
                           wallColour);
        }
    return mesh;
}

inline Rgb canopyColour(BiomeId biome) noexcept
{
    switch (biome)
    {
    case BiomeId::Taiga: return {0.13f, 0.33f, 0.25f};
    case BiomeId::Rainforest: return {0.09f, 0.42f, 0.17f};
    case BiomeId::Savanna: return {0.50f, 0.47f, 0.21f};
    case BiomeId::Desert: return {0.30f, 0.50f, 0.28f};
    case BiomeId::Marsh: return {0.29f, 0.46f, 0.27f};
    default: break;
    }
    return {0.16f, 0.42f, 0.20f};
}

inline void appendPlant(MapMesh &mesh, const WorldAtlas &atlas, float cx, float cy, float cz, float half)
{
    const float trunkHalf = half * 0.22f;
    const float trunkTop = cy + half * 1.1f;
    const float base = cy - half;

    Rgb canopy = canopyColour(BiomeId::Count);
    if (!atlas.biomes.empty() && !atlas.height.empty())
    {
        const core::i32 gx = static_cast<core::i32>(cx + static_cast<float>(atlas.height.width()) * 0.5f);
        const core::i32 gz = static_cast<core::i32>(cz + static_cast<float>(atlas.height.depth()) * 0.5f);
        if (atlas.biomes.contains(gx, gz))
            canopy = canopyColour(atlas.biomes.at(static_cast<core::u32>(gx), static_cast<core::u32>(gz)));
    }

    const Rgb bark{0.33f, 0.23f, 0.15f};
    appendQuad(mesh, cx - trunkHalf, base, cz, cx - trunkHalf, trunkTop, cz, cx + trunkHalf, trunkTop, cz,
               cx + trunkHalf, base, cz, 0.0f, 0.0f, 1.0f, bark);
    appendQuad(mesh, cx, base, cz - trunkHalf, cx, trunkTop, cz - trunkHalf, cx, trunkTop, cz + trunkHalf, cx, base,
               cz + trunkHalf, 1.0f, 0.0f, 0.0f, bark);

    const float spread = half * 1.15f;
    const float apex = trunkTop + half * 2.1f;
    appendTriangle(mesh, cx, apex, cz, cx - spread, trunkTop, cz - spread, cx + spread, trunkTop, cz - spread, 0.0f,
                   0.45f, -0.89f, canopy);
    appendTriangle(mesh, cx, apex, cz, cx + spread, trunkTop, cz - spread, cx + spread, trunkTop, cz + spread, 0.89f,
                   0.45f, 0.0f, canopy);
    appendTriangle(mesh, cx, apex, cz, cx + spread, trunkTop, cz + spread, cx - spread, trunkTop, cz + spread, 0.0f,
                   0.45f, 0.89f, canopy);
    appendTriangle(mesh, cx, apex, cz, cx - spread, trunkTop, cz + spread, cx - spread, trunkTop, cz - spread, -0.89f,
                   0.45f, 0.0f, canopy);
}

inline core::u32 foldMapMesh(const MapMesh &mesh) noexcept
{
    core::u32 hash = 0x811C9DC5u;
    const auto fold = [&hash](float value) {
        core::u32 bits = 0u;
        static_assert(sizeof(bits) == sizeof(value), "a float is four bytes here");
        __builtin_memcpy(&bits, &value, sizeof(bits));
        hash = (hash ^ bits) * 0x01000193u;
    };
    for (core::usize i = 0u; i < mesh.size(); ++i)
    {
        const MapVertex &v = mesh[i];
        fold(v.x);
        fold(v.y);
        fold(v.z);
        fold(v.nx);
        fold(v.ny);
        fold(v.nz);
        fold(v.colour.r);
        fold(v.colour.g);
        fold(v.colour.b);
    }
    return hash;
}

} // namespace lpl::procgen

#endif // LPL_PROCGEN_MAP_MESH_INL
