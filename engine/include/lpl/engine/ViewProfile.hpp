/**
 * @file ViewProfile.hpp
 * @brief What a world LOOKS like, as content a document can carry.
 *
 * The half of presentation that belongs to the place rather than to the machine.
 * A `.lplscene` could already specify how many hydraulic erosion iterations to run
 * and had no way to say the world was at dusk; the sky's colour, the tint of its
 * water and the palette of its biomes were compiled into the host, so every world
 * the format could express came out the same blue.
 *
 * The line drawn here is the one @c TerrainSurfaceParams already draws internally,
 * promoted to the document:
 *
 *  - CONTENT, and therefore here: where the sea is, how dense the haze, the hour of
 *    the day, the colour of a forest. Two machines rendering the same cartridge
 *    must agree about these or they are not showing the same world.
 *  - BUDGET, and therefore engine::HostProfile's: whether the fog is computed per
 *    pixel, how many chunks stay resident, how many shadow chunks a tick may bake.
 *    Two machines rendering the same cartridge SHOULD differ about these.
 *
 * Putting a budget in a cartridge is how a phone gets told to render like a
 * workstation. Putting the palette in the host is how every world looks the same.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-31
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ENGINE_VIEW_PROFILE_HPP
#    define LPL_ENGINE_VIEW_PROFILE_HPP

#    include <lpl/engine/TerrainRenderer.hpp>
#    include <lpl/engine/TerrainSurface.hpp>
#    include <lpl/pack/GamePack.hpp>

namespace lpl::engine {

/**
 * @struct ViewProfile
 * @brief The decoded view profile: the engine structs a cartridge fills in.
 */
struct ViewProfile {
    render::SkyParams sky{};
    render::WaterParams water{};
    TerrainSurfaceParams surface{};
    core::f32 dayFraction{0.32f};

    core::u32 grazerTint{0x00D0A852u};
    core::u32 hunterTint{0x00C03028u};
    core::f32 bodyScale{0.35f};

    /**
     * @brief Biome colours, indexed by procgen::BiomeId.
     *
     * @c paletteCount of zero means "the world declined to say", and a caller must
     * then keep its own colours. That is not the same as a palette of zero entries,
     * which would be a world painted black — a distinction worth a field, because
     * the zero-initialised case is the common one.
     */
    core::u32 palette[pack::kWireBiomeColours]{};
    core::u32 paletteCount{0u};

    /** @brief Looks up a biome colour, or falls back to what the caller uses. */
    [[nodiscard]] core::u32 colourFor(core::u32 biome, core::u32 fallback) const noexcept
    {
        return biome < paletteCount ? palette[biome] : fallback;
    }
};

/**
 * @brief Decodes a wire view profile into the engine structs it describes.
 *
 * Field by field and by name, as the recipe codec is: a rename or a reordering on
 * either side of the fence has to be a compile error rather than a world that
 * silently comes out the wrong colour.
 */
[[nodiscard]] inline ViewProfile toEngineView(const pack::ViewV1 &wire) noexcept
{
    ViewProfile out{};

    out.sky.zenithR = wire.zenithR;
    out.sky.zenithG = wire.zenithG;
    out.sky.zenithB = wire.zenithB;
    out.sky.horizonR = wire.horizonR;
    out.sky.horizonG = wire.horizonG;
    out.sky.horizonB = wire.horizonB;
    out.sky.duskR = wire.duskR;
    out.sky.duskG = wire.duskG;
    out.sky.duskB = wire.duskB;
    out.sky.groundR = wire.groundR;
    out.sky.groundG = wire.groundG;
    out.sky.groundB = wire.groundB;
    out.sky.sunSize = wire.sunSize;
    out.sky.mieStrength = wire.mieStrength;
    out.sky.mieSharpness = wire.mieSharpness;
    out.sky.nightFloor = wire.nightFloor;

    out.dayFraction = wire.dayFraction;

    out.surface.seaLevel = wire.seaLevel;
    out.surface.fogDensity = wire.fogDensity;
    out.surface.ambient = wire.ambient;
    out.surface.grainTiles = wire.grainTiles;
    out.surface.shadowSteps = wire.shadowSteps;

    out.water.shallow = wire.waterShallow;
    out.water.deep = wire.waterDeep;
    out.water.rippleScale = wire.rippleScale;
    out.water.rippleAmplitude = wire.rippleAmplitude;
    out.water.glintPower = wire.glintPower;
    out.water.depthScale = wire.depthScale;
    // phase is not carried: it is where the ripples HAPPEN to be, which is state,
    // not content. A cartridge that pinned it would restart every load mid-swell.

    out.grazerTint = wire.grazerTint;
    out.hunterTint = wire.hunterTint;
    out.bodyScale = wire.bodyScale;

    if ((wire.flags & pack::kViewFlagOverridePalette) != 0u)
    {
        const core::u32 count =
            wire.biomeColourCount > pack::kWireBiomeColours ? pack::kWireBiomeColours : wire.biomeColourCount;
        for (core::u32 i = 0u; i < count; ++i)
            out.palette[i] = wire.biomeColour[i];
        out.paletteCount = count;
    }
    return out;
}

/** @brief Encodes an engine view profile back to wire form (the baker's half). */
[[nodiscard]] inline pack::ViewV1 toWireView(const ViewProfile &profile) noexcept
{
    pack::ViewV1 wire{};

    wire.zenithR = profile.sky.zenithR;
    wire.zenithG = profile.sky.zenithG;
    wire.zenithB = profile.sky.zenithB;
    wire.horizonR = profile.sky.horizonR;
    wire.horizonG = profile.sky.horizonG;
    wire.horizonB = profile.sky.horizonB;
    wire.duskR = profile.sky.duskR;
    wire.duskG = profile.sky.duskG;
    wire.duskB = profile.sky.duskB;
    wire.groundR = profile.sky.groundR;
    wire.groundG = profile.sky.groundG;
    wire.groundB = profile.sky.groundB;
    wire.sunSize = profile.sky.sunSize;
    wire.mieStrength = profile.sky.mieStrength;
    wire.mieSharpness = profile.sky.mieSharpness;
    wire.nightFloor = profile.sky.nightFloor;

    wire.dayFraction = profile.dayFraction;

    wire.seaLevel = profile.surface.seaLevel;
    wire.fogDensity = profile.surface.fogDensity;
    wire.ambient = profile.surface.ambient;
    wire.grainTiles = profile.surface.grainTiles;
    wire.shadowSteps = profile.surface.shadowSteps;

    wire.waterShallow = profile.water.shallow;
    wire.waterDeep = profile.water.deep;
    wire.rippleScale = profile.water.rippleScale;
    wire.rippleAmplitude = profile.water.rippleAmplitude;
    wire.glintPower = profile.water.glintPower;
    wire.depthScale = profile.water.depthScale;

    wire.grazerTint = profile.grazerTint;
    wire.hunterTint = profile.hunterTint;
    wire.bodyScale = profile.bodyScale;

    if (profile.paletteCount != 0u)
    {
        const core::u32 count =
            profile.paletteCount > pack::kWireBiomeColours ? pack::kWireBiomeColours : profile.paletteCount;
        for (core::u32 i = 0u; i < count; ++i)
            wire.biomeColour[i] = profile.palette[i];
        wire.biomeColourCount = count;
        wire.flags |= pack::kViewFlagOverridePalette;
    }
    return wire;
}

} // namespace lpl::engine

#endif // LPL_ENGINE_VIEW_PROFILE_HPP
