/**
 * @file ScentWindow.inl
 * @brief Out-of-line definitions for ai::ScentWindow.
 *
 * Included at the end of ScentWindow.hpp. The window is consumed header-only, in
 * the freestanding kernel as well as on the host, so its definitions go here rather
 * than into a translation unit neither kernel build path lists.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-28
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_AI_SCENT_WINDOW_INL
#    define LPL_AI_SCENT_WINDOW_INL

namespace lpl::ai {

inline void ScentWindow::open(core::u32 span, core::u32 layers, core::u32 slackDivisor)
{
    _span = span;
    _slackDivisor = slackDivisor == 0u ? 1u : slackDivisor;
    _field = StigmergyField{span, span, layers};
    _recentres = 0u;
}

inline void ScentWindow::centreOn(core::i32 worldX, core::i32 worldZ)
{
    _originX = worldX - static_cast<core::i32>(_span / 2u);
    _originZ = worldZ - static_cast<core::i32>(_span / 2u);
    _field.clear();
}

inline bool ScentWindow::follow(core::i32 focusX, core::i32 focusZ)
{
    const core::i32 halfSpan = static_cast<core::i32>(_span / 2u);
    const core::i32 driftX = focusX - (_originX + halfSpan);
    const core::i32 driftZ = focusZ - (_originZ + halfSpan);
    const core::i32 slack = static_cast<core::i32>(_span / _slackDivisor);

    if (driftX > -slack && driftX < slack && driftZ > -slack && driftZ < slack)
        return false;

    centreOn(focusX, focusZ);
    ++_recentres;
    return true;
}

inline bool ScentWindow::toWindow(core::i32 worldX, core::i32 worldZ, core::u32 &outX, core::u32 &outZ) const noexcept
{
    const core::i32 localX = worldX - _originX;
    const core::i32 localZ = worldZ - _originZ;
    if (localX < 0 || localZ < 0 || localX >= static_cast<core::i32>(_span) ||
        localZ >= static_cast<core::i32>(_span))
        return false;
    outX = static_cast<core::u32>(localX);
    outZ = static_cast<core::u32>(localZ);
    return true;
}

} // namespace lpl::ai

#endif // LPL_AI_SCENT_WINDOW_INL
