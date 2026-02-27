/**
 * @file InputState.hpp
 * @brief Snapshot of classical (non-BCI) input at a single tick.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_INPUT_INPUTSTATE_HPP
    #define LPL_INPUT_INPUTSTATE_HPP

#include <lpl/math/Vec3.hpp>
#include <lpl/math/Quat.hpp>
#include <lpl/math/FixedPoint.hpp>
#include <lpl/core/Types.hpp>

#include <bitset>

namespace lpl::input {

/**
 * @enum ButtonId
 * @brief Enumeration of all supported buttons / actions.
 */
enum class ButtonId : core::u8
{
    MoveForward = 0,
    MoveBack,
    MoveLeft,
    MoveRight,
    Jump,
    Crouch,
    Sprint,
    PrimaryAction,
    SecondaryAction,
    Interact,

    Count
};

/**
 * @struct InputState
 * @brief Deterministic input snapshot — one per tick per player.
 *
 * All values are Fixed32 for determinism. Look direction is a quaternion.
 */
struct InputState
{
    std::bitset<static_cast<core::usize>(ButtonId::Count)> buttons{};
    math::Vec3<math::Fixed32>   moveAxis{};
    math::Quat<math::Fixed32>   lookOrientation{};
    core::u32                   sequence{0};
};

} // namespace lpl::input

#endif // LPL_INPUT_INPUTSTATE_HPP
