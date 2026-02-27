/**
 * @file Constants.hpp
 * @brief Engine-wide compile-time constants.
 *
 * All tunable parameters that affect the simulation, networking, and
 * memory budgets are centralised here so that a single header controls
 * the engine's fundamental operating parameters.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */
#pragma once

#ifndef LPL_CORE_CONSTANTS_HPP
    #define LPL_CORE_CONSTANTS_HPP

    #include "Types.hpp"

namespace lpl::core {

inline constexpr u32   kTickRate              = 144;
inline constexpr f64   kFixedDeltaTime        = 1.0 / static_cast<f64>(kTickRate);

inline constexpr u32   kMaxEntities           = 10'000;
inline constexpr u32   kMaxEntityIdSpace      = 1'000'000;
inline constexpr u32   kMaxChunks             = 65'536;
inline constexpr u32   kMaxEntitiesPerChunk   = 1'000;
inline constexpr u32   kMaxSystems            = 64;

inline constexpr u16   kDefaultPort           = 7777;
inline constexpr u32   kRingSlots             = 4096;
inline constexpr u32   kMaxPacketSize         = 256;
inline constexpr u32   kMaxSessions           = 256;

inline constexpr u32   kOctreeMaxDepth        = 8;
inline constexpr u32   kOctreeLeafCapacity    = 32;
inline constexpr u32   kBroadphaseN2Threshold = 32;
inline constexpr u32   kCollisionSolverIter   = 4;
inline constexpr f64   kCollisionRestitution  = 0.5;

inline constexpr u32   kSleepFrameThreshold   = 30;
inline constexpr f64   kSleepVelocitySqThresh = 0.01;
inline constexpr f64   kFrictionLinear         = 0.995;

inline constexpr i32   kMortonBias            = 1 << 20;
inline constexpr u32   kGenerationBits        = 18;
inline constexpr u32   kSlotBits              = 14;

inline constexpr usize kFrameArenaSize        = 16 * 1024 * 1024;

} // namespace lpl::core

#endif // LPL_CORE_CONSTANTS_HPP
