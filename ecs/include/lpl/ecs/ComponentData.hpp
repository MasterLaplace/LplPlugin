/**
 * @file ComponentData.hpp
 * @brief The C++ shapes of component payloads, in one place.
 *
 * @warning **One place, because two are how a component gets read as something it is not.** A component
 * has three descriptions: the size `defaultLayout` declares, the fields `kSchemas` describes for
 * the baker and the editor, and the struct every system reads it AS. The first two are asserted
 * against each other in `ComponentReflection.hpp`; this file is where the third lives so it can
 * be asserted too, and so it stops being declared next to whichever system happened to need it
 * first -- which is where `HistoricalBody` was, in a header the ECS could not include without
 * depending on the module that owned it.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ECS_COMPONENTDATA_HPP
#    define LPL_ECS_COMPONENTDATA_HPP

#    include <lpl/core/Types.hpp>
#    include <lpl/ecs/Component.hpp>

namespace lpl::ecs {

/**
 * @struct HistoricalBody
 * @brief The `ComponentId::Historical` payload: who this entity is, in a corpus.
 */
struct HistoricalBody {
    core::u32 subject{0u}; ///< Identifier the corpus knows them by.
    core::i32 bornDay{0};  ///< As a source dates it; zero when none does.
    core::i32 diedDay{0};  ///< Zero when no source says.
};

/**
 * @struct WorldCellData
 * @brief The `ComponentId::WorldCell` payload: which chunk a Position is relative to.
 */
struct WorldCellData {
    core::i32 chunkX{0};
    core::i32 chunkZ{0};
};

// @warning The struct must be exactly what the layout declares. Nothing else ties the two together:
// a field added to one and not the other is a component that serialises to what no system wrote.
static_assert(sizeof(HistoricalBody) == defaultLayout(ComponentId::Historical).size,
              "HistoricalBody does not match the layout ComponentId::Historical declares");
static_assert(sizeof(WorldCellData) == defaultLayout(ComponentId::WorldCell).size,
              "WorldCellData does not match the layout ComponentId::WorldCell declares");

} // namespace lpl::ecs

#endif // LPL_ECS_COMPONENTDATA_HPP
