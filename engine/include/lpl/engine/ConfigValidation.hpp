/**
 * @file ConfigValidation.hpp
 * @brief Catches contradictory engine configurations before they bite.
 *
 * The networking knobs (AOI, network LOD, precision LOD, acked baseline,
 * bandwidth budget) interact: a switch is inert, or actively broken, unless the
 * switches it depends on are also set. Rather than let a misconfiguration fail
 * silently (an AOI server with physics off dereferences a null spatial index) or
 * waste bandwidth (precision LOD with no far ring), this reports every such
 * conflict as a short message.
 *
 * Header-only and allocation-free: it hands each warning to a caller-supplied
 * callback as a string literal, so it costs nothing to call and pulls in no
 * container. The apps print them; a test asserts on them.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-24
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ENGINE_CONFIGVALIDATION_HPP
#    define LPL_ENGINE_CONFIGVALIDATION_HPP

#    include <lpl/engine/Config.hpp>
#    include <lpl/math/FixedPoint.hpp>

namespace lpl::engine {

/**
 * @brief Calls @p emit(const char*) once per configuration conflict found.
 *
 * @tparam Emit Callable accepting a @c const @c char*.
 * @param config The configuration to check.
 * @param emit   Receives each warning message (a string literal).
 * @return The number of warnings emitted (0 means the config is self-consistent).
 */
template <typename Emit> core::u32 forEachConfigWarning(const Config &config, Emit &&emit)
{
    core::u32 count = 0;
    const auto warn = [&](const char *message) {
        emit(message);
        ++count;
    };

    const bool aoi = config.interestRadius() > math::Fixed32::zero();
    const bool lod = config.lodNearRadius() > math::Fixed32::zero();
    const bool precision = config.worldExtent() > math::Fixed32::zero();

    // ── Server sanity ──────────────────────────────────────────────────────── //
    if (config.serverMode() && !config.enableNetworking())
        warn("serverMode is on but networking is off: a server cannot receive or "
             "broadcast — enable networking or run a solo client.");

    // ── Area of interest ───────────────────────────────────────────────────── //
    if (aoi && !config.enableNetworking())
        warn("interestRadius is set but networking is off: AOI is the broadcast "
             "path, so it has nothing to run over.");
    if (aoi && !config.enablePhysics())
        warn("interestRadius is set but physics is off: AOI queries the spatial "
             "partition, which only exists when physics is enabled (it would "
             "dereference a null index).");
    if (aoi && !config.serverMode())
        warn("interestRadius is set on a client: AOI is a server-side broadcast "
             "and is ignored here — a client reconciles, it does not broadcast.");

    // ── Network LOD (cadence) ──────────────────────────────────────────────── //
    if (lod && !aoi)
        warn("lodNearRadius is set but interestRadius is 0: network LOD lives "
             "inside the interest radius; with no AOI it does nothing.");
    if (lod && aoi && config.lodNearRadius() >= config.interestRadius())
        warn("lodNearRadius >= interestRadius: the near ring covers the whole "
             "interest area, so the far ring is empty and LOD never fires.");

    // ── Precision LOD (quantization) ───────────────────────────────────────── //
    if (precision && !lod)
        warn("worldExtent is set but lodNearRadius is 0: position quantization "
             "applies only to the far ring, which does not exist without LOD.");
    if (precision && (config.lodFarPosBits() == 0 || config.lodFarPosBits() > 32 || config.lodFarPosBits() % 8 != 0))
        warn("lodFarPosBits must be a non-zero multiple of 8 (<=32); the codec "
             "will fall back to 16 to keep the wire byte-aligned.");

    // ── Reliable baseline ──────────────────────────────────────────────────── //
    if (config.reliableBaseline() && !aoi)
        warn("reliableBaseline is on but AOI is off: the acked-baseline model only "
             "governs the AOI delta stream.");
    if (config.reliableBaseline() && config.keyframeInterval() > 0 && config.keyframeInterval() < 1000)
        warn("reliableBaseline is on together with a short keyframeInterval: acks "
             "already guarantee delivery, so periodic keyframes are redundant "
             "bandwidth — raise keyframeInterval or leave the optimistic model.");

    // ── Presentation against the host it runs on ───────────────────────────── //
    //
    // These catch the combinations that produce no error and no picture — the kind
    // that costs an afternoon because everything "works".
    if (config.headless() && config.enablePerPixelSurface())
        warn("headless is on together with per-pixel surface shading: there is no "
             "surface to look at, and the shading still costs a tick.");
    if (config.headless() && config.enableTerrainShadows())
        warn("headless is on together with terrain shadows: the shadow masks are "
             "computed and never drawn.");
    if (config.enablePbrSurface() && !config.enablePerPixelSurface())
        warn("enablePbrSurface without enablePerPixelSurface: the physically based "
             "path IS a per-pixel path, so it cannot run on the flat one.");
    if (config.enableTerrainShadows() && config.shadowChunksPerTick() == 0)
        warn("terrain shadows are on but shadowChunksPerTick is 0: no mask is ever "
             "refreshed, so the shadows freeze at whatever the first tick produced.");
    if (config.lodRings() == 0)
        warn("lodRings is 0: no ring is ever drawn, so a streamed world renders "
             "nothing at all.");
    if (config.maxResidentChunks() != 0 && config.maxResidentChunks() < 9)
        warn("maxResidentChunks below 9: a 3x3 neighbourhood is the minimum a "
             "walker needs, and below it chunks are released as fast as they are "
             "built.");
    if (config.skyBlockSize() > 8)
        warn("skyBlockSize above 8: the sky is evaluated once per block, and past "
             "8x8 the horizon becomes visible steps rather than a gradient.");

    return count;
}

} // namespace lpl::engine

#endif // LPL_ENGINE_CONFIGVALIDATION_HPP
