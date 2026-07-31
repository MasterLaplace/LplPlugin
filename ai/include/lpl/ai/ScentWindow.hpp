/**
 * @file ScentWindow.hpp
 * @brief A stigmergy field that follows a walker across a world with no corner.
 *
 * @ref StigmergyField is a grid, and a grid has a corner. An endless world does
 * not, so the field becomes a WINDOW anchored on world coordinates: when the walker
 * leaves its middle third the window recentres, and what was outside is gone.
 *
 * That loss is real and worth naming rather than hiding: a scent laid down two
 * windows ago has been dropped, so a streamed world has short-term memory where a
 * bounded one has a whole map's worth. It is also what makes it affordable — a field
 * that grew with the world would be a grid the size of the world, which is the one
 * thing streaming exists to avoid. Chunked pheromones with per-chunk evaporation
 * would keep the history, and they are their own piece of work.
 *
 * The hysteresis matters as much as the window: recentring the moment the walker
 * leaves the exact middle would clear the field on every other step while they
 * pace, and a trail that is deleted continuously is not a trail.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-28
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_AI_SCENT_WINDOW_HPP
#    define LPL_AI_SCENT_WINDOW_HPP

#    include <lpl/ai/StigmergyField.hpp>
#    include <lpl/core/Types.hpp>

namespace lpl::ai {

/**
 * @class ScentWindow
 * @brief A stigmergy field on absolute world coordinates, recentred with slack.
 */
class ScentWindow {
public:
    /**
     * @brief Opens a window of @p span cells with @p layers pheromone channels.
     *
     * @param slackDivisor The walker may drift span/slackDivisor from the middle
     *                     before the window moves. Six is a third of the span on
     *                     either side, which is far enough that pacing does not
     *                     clear the field and near enough that the walker never
     *                     reaches the edge.
     */
    void open(core::u32 span, core::u32 layers, core::u32 slackDivisor = 6u);

    [[nodiscard]] StigmergyField &field() noexcept { return _field; }
    [[nodiscard]] const StigmergyField &field() const noexcept { return _field; }
    [[nodiscard]] core::u32 span() const noexcept { return _span; }
    [[nodiscard]] core::u32 recentres() const noexcept { return _recentres; }
    [[nodiscard]] core::i32 originX() const noexcept { return _originX; }
    [[nodiscard]] core::i32 originZ() const noexcept { return _originZ; }

    /** @brief Centres the window on a world cell, dropping what falls outside. */
    void centreOn(core::i32 worldX, core::i32 worldZ);

    /**
     * @brief Recentres the window if the focus has drifted past the slack.
     * @return True when the window moved, and therefore when trails were dropped.
     */
    bool follow(core::i32 focusX, core::i32 focusZ);

    /** @brief World cell to a cell of the window, when it is inside it. */
    [[nodiscard]] bool toWindow(core::i32 worldX, core::i32 worldZ, core::u32 &outX,
                                core::u32 &outZ) const noexcept;

private:
    StigmergyField _field{};
    core::u32 _span{64u};
    core::u32 _slackDivisor{6u};
    core::i32 _originX{0};
    core::i32 _originZ{0};
    core::u32 _recentres{0u};
};

} // namespace lpl::ai

// Out-of-line definitions: the window is consumed header-only, the freestanding
// kernel included, so they live in a .inl rather than a .cpp that neither kernel
// build path lists.
#    include <lpl/ai/ScentWindow.inl>

#endif // LPL_AI_SCENT_WINDOW_HPP
