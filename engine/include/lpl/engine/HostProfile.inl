#ifndef LPL_ENGINE_HOST_PROFILE_INL
#define LPL_ENGINE_HOST_PROFILE_INL

namespace lpl::engine {

inline Config::Builder &applyHostProfile(Config::Builder &builder, HostProfile profile)
{
    switch (profile)
    {
    case HostProfile::Ring0Client:
        builder.arenaSize(256u * 1024u)
            .worldArenaSize(512u * 1024u)
            .worldCellCapacity(1024u)
            .maxResidentChunks(56u)
            .enablePhysics(true)
            .enableNetworking(false)
            .enableRendering(false)
            .enableGpu(false)
            .enableBci(false)
            .enableRealTimeGuard(true)
            .enableStreaming(true)
            .headless(false)
            .serverMode(false)
            .lodRings(3u)
            .viewDistance(70.0f)
            .enableTerrainShadows(true)
            .shadowChunksPerTick(1u)
            .enablePerPixelSurface(true)
            .enablePbrSurface(false)
            .enableWaterReflection(true)
            .waterTessellation(6u)
            .caveDrawRadius(10u)
            .skyBlockSize(1u);
        break;

    case HostProfile::Ring0Server:
        builder.arenaSize(256u * 1024u)
            .worldArenaSize(512u * 1024u)
            .worldCellCapacity(1024u)
            .maxResidentChunks(24u)
            .enablePhysics(true)
            .enableNetworking(false)
            .enableRendering(false)
            .enableGpu(false)
            .enableBci(false)
            .enableRealTimeGuard(true)
            .headless(true)
            .serverMode(false)
            .lodRings(1u)
            .viewDistance(0.0f)
            .enableTerrainShadows(false)
            .shadowChunksPerTick(0u)
            .enablePerPixelSurface(false)
            .enablePbrSurface(false)
            .enableWaterReflection(false)
            .waterTessellation(0u)
            .caveDrawRadius(0u)
            .skyBlockSize(4u);
        break;

    case HostProfile::DesktopClient:
        builder.arenaSize(64u * 1024u * 1024u)
            .worldArenaSize(64u * 1024u * 1024u)
            .maxResidentChunks(256u)
            .enablePhysics(true)
            .enableNetworking(false)
            .enableRendering(true)
            .enableRealTimeGuard(false)
            .enableStreaming(true)
            .headless(false)
            .serverMode(false)
            .lodRings(4u)
            .viewDistance(220.0f)
            .enableTerrainShadows(true)
            .shadowChunksPerTick(4u)
            .enablePerPixelSurface(true)
            .enablePbrSurface(true)
            .enableWaterReflection(true)
            .waterTessellation(14u)
            .caveDrawRadius(24u)
            .skyBlockSize(1u);
        break;

    case HostProfile::DedicatedServer:
        builder.arenaSize(64u * 1024u * 1024u)
            .worldArenaSize(64u * 1024u * 1024u)
            .maxResidentChunks(512u)
            .enablePhysics(true)
            .enableNetworking(true)
            .enableRendering(false)
            .enableRealTimeGuard(false)
            .headless(true)
            .serverMode(true)
            .lodRings(1u)
            .viewDistance(0.0f)
            .enableTerrainShadows(false)
            .shadowChunksPerTick(0u)
            .enablePerPixelSurface(false)
            .enablePbrSurface(false)
            .enableWaterReflection(false)
            .waterTessellation(0u)
            .caveDrawRadius(0u)
            .skyBlockSize(4u);
        break;
    }
    return builder;
}

} // namespace lpl::engine

#endif // LPL_ENGINE_HOST_PROFILE_INL
