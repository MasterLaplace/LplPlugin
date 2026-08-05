/**
 * @file ModelSection.hpp
 * @brief Descriptor for the inference weights that accompany a world.
 *
 * Not the weights themselves — a descriptor: identity, quantisation, expected
 * arena size, and the hash. The blob arrives as its own boot module; this section
 * is how a world declares which demon it expects.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_LPL_PACK_MODELSECTION_HPP
#    define LPL_LPL_PACK_MODELSECTION_HPP

#    include <lpl/core/Types.hpp>

namespace lpl::pack {

// TODO(lot 8): declarations only — no implementation yet.

} // namespace lpl::pack

#endif // LPL_LPL_PACK_MODELSECTION_HPP
