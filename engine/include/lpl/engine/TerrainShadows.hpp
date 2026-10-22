/**
 * @file TerrainShadows.hpp
 * @brief The shadows of a streamed world's chunks, cast by the ground its resident chunks hold.
 *
 * Where TerrainStreamer and TerrainSurface meet. The surface marches the rays and knows the sun; the
 * streamer knows where the ground is. Neither includes the other: the surface takes the ground as a
 * callback, and the streamer knows nothing of how the world looks.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-10-22
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ENGINE_TERRAIN_SHADOWS_HPP
#    define LPL_ENGINE_TERRAIN_SHADOWS_HPP

#    include <lpl/engine/TerrainStreamer.hpp>
#    include <lpl/engine/TerrainSurface.hpp>

namespace lpl::engine {

/**
 * @brief Fills a resident chunk's shadow mask: the ground over itself and its neighbours, then its props.
 *
 * The rays read @ref TerrainStreamer::groundAt, the field the world is drawn from: eroded, carved by
 * the rivers and lifted by the measured relief. A ridge in the next resident chunk casts here too, and
 * past the resident ring the world's height answers.
 *
 * @param streamer    The streamer that holds @p chunk.
 * @param surface     The surface whose sun and shadow settings cast.
 * @param chunk       The chunk whose `shade` is written.
 * @param forEachProp (emit) with emit(cellX, cellZ, height, spread) for every prop of @p chunk that
 *                    casts, as TerrainSurface::fillShadowMask takes it.
 */
template <typename ForEachProp>
void castChunkShadows(const TerrainStreamer &streamer, const TerrainSurface &surface, TerrainChunk &chunk,
                      ForEachProp &&forEachProp);

} // namespace lpl::engine

#    include "TerrainShadows.inl"

#endif // LPL_ENGINE_TERRAIN_SHADOWS_HPP
