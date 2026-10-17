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
 * The checks cover, in order, server sanity, the area of interest, network LOD
 * (cadence), precision LOD (quantization), the reliable baseline, and presentation
 * against the host it runs on. The last ones catch the combinations that produce no
 * error and no picture — the kind that costs an afternoon because everything "works".
 *
 * @tparam Emit Callable accepting a @c const @c char*.
 * @param config The configuration to check.
 * @param emit   Receives each warning message (a string literal).
 * @return The number of warnings emitted (0 means the config is self-consistent).
 */
template <typename Emit> core::u32 forEachConfigWarning(const Config &config, Emit &&emit);

} // namespace lpl::engine

#    include "ConfigValidation.inl"

#endif // LPL_ENGINE_CONFIGVALIDATION_HPP
