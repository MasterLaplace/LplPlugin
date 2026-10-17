/**
 * @file GameProfile.hpp
 * @brief One-call netcode presets per game genre (book chapter 6).
 *
 * Chapter 6 lays out a spectrum of synchronisation models rather than a single
 * one, and the engine keeps each as an option. This turns a genre into a
 * self-consistent set of those options, so an app says what KIND of game it is
 * and gets a configuration that passes @ref forEachConfigWarning with no
 * conflicts. The default and best-supported target is the MMORPG / FullDive
 * build; the others exist because "rien n'empêche d'avoir des options plus
 * généralistes".
 *
 * A profile only sets the networking/replication knobs; tick rate, entity
 * budget, GPU, BCI and address stay the app's call.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-24
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ENGINE_GAMEPROFILE_HPP
#    define LPL_ENGINE_GAMEPROFILE_HPP

#    include <lpl/engine/Config.hpp>
#    include <lpl/math/FixedPoint.hpp>

#    include <cstring>

namespace lpl::engine {

/**
 * @enum GameProfile
 * @brief The genre a preset targets, each mapping to a point on chapter 6's
 *        spectrum of synchronisation models.
 */
enum class GameProfile {
    Mmorpg,   /**< Many players, huge world: AOI + network LOD + precision LOD (default). */
    Fps,      /**< Few players, low latency, precise hits: full state, no LOD, reliable. */
    Rts,      /**< Many units, wide view: AOI with a large radius, LOD on the far units. */
    Fighting, /**< Two players, everything visible, competitive integrity: reliable, no LOD. */
    CoOp      /**< A handful of players: full broadcast, no LOD, simple and cheap. */
};

/**
 * @brief Applies the networking/replication preset for @p profile to @p builder.
 *
 * Returns the same builder for chaining, so an app writes the genre-independent
 * knobs around it: `applyGameProfile(Builder{}.tickRate(144), profile).build()`.
 *
 * What each preset sets, and why:
 *
 * - Mmorpg, the scaling target: interest management breaks the O(N²) broadcast, LOD
 *   thins the far ring in cadence and precision, a keyframe self-heals lost deltas,
 *   and a bandwidth budget caps each client. Optimistic baseline (no per-client ack
 *   traffic at MMO scale).
 * - Fps: small arena, hit registration must be exact: full precision everywhere (no
 *   LOD, full-float positions), reliable acked baseline for competitive integrity, so
 *   acks rather than keyframes carry reliability, and no bandwidth budget, which would
 *   starve a shot's target. A modest interest radius still culls players across a
 *   large map.
 * - Rts: hundreds of units, the player sees a wide slice of the map: a large interest
 *   radius, LOD to thin distant units, generous budget.
 * - Fighting: two players, both see the whole stage: full broadcast (everyone sees
 *   everything), no culling, full precision, and no acked baseline, since full
 *   broadcast has no per-client delta baseline. (Rollback, §6.2.2, is the other option
 *   for this genre; this preset is the authoritative path.)
 * - CoOp: a handful of players against the world: full broadcast is cheap enough, no
 *   LOD, keep it simple.
 *
 * @param builder The builder to configure (mutated in place).
 * @param profile The genre preset to apply.
 * @return Reference to @p builder.
 */
inline Config::Builder &applyGameProfile(Config::Builder &builder, GameProfile profile);

/**
 * @brief Parses a profile name (case-insensitive) for a CLI @c --game argument.
 * @param name    The text: "mmorpg", "fps", "rts", "fighting", "coop".
 * @param outProfile Set on success.
 * @return false if @p name matches no profile.
 */
[[nodiscard]] inline bool parseGameProfile(const char *name, GameProfile &outProfile);

/**
 * @brief Human-readable name of a profile, for logging.
 */
[[nodiscard]] inline const char *gameProfileName(GameProfile profile) noexcept;

} // namespace lpl::engine

#    include "GameProfile.inl"

#endif // LPL_ENGINE_GAMEPROFILE_HPP
