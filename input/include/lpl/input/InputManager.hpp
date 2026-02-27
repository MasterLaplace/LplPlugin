/**
 * @file InputManager.hpp
 * @brief Per-entity input aggregation with neural speed modulation.
 *
 * Ported from legacy engine/InputManager.hpp.
 * Features: per-entity InputState map, neural speed modulation
 * (concentration → velocity scale), rising-edge blink detection,
 * grounded-state tracking for jump-gating.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_INPUT_INPUTMANAGER_HPP
    #define LPL_INPUT_INPUTMANAGER_HPP

#include <lpl/input/IInputSource.hpp>
#include <lpl/input/InputState.hpp>
#include <lpl/input/NeuralInputState.hpp>
#include <lpl/math/Vec3.hpp>
#include <lpl/math/FixedPoint.hpp>
#include <lpl/core/Types.hpp>
#include <lpl/core/Expected.hpp>
#include <lpl/core/NonCopyable.hpp>

#include <memory>
#include <vector>

namespace lpl::input {

/**
 * @struct NeuralControl
 * @brief Legacy-compatible neural control data per entity.
 *
 * Used by the movement system to modulate speed and trigger blink-based jumps.
 */
struct NeuralControl
{
    float alpha          = 0.0f;
    float beta           = 0.0f;
    float concentration  = 0.0f;
};

/**
 * @struct PerEntityInput
 * @brief Complete input snapshot for a single entity (keyboard + neural).
 *
 * Mirrors the legacy InputState struct from engine/InputManager.hpp but uses
 * the new Fixed32-based types where possible.
 */
struct PerEntityInput
{
    static constexpr core::u32 kMaxKeys = 512;
    static constexpr core::u32 kMaxAxes = 16;

    bool   keys[kMaxKeys]  = {};
    float  axes[kMaxAxes]  = {};

    NeuralControl neural{};

    /** @brief Previous blink state for rising-edge detection. */
    bool blinkPrev    = false;
    /** @brief Whether the entity is currently on the ground. */
    bool isGrounded   = false;

    /** @brief Gets key state. */
    [[nodiscard]] bool getKey(core::u16 key) const noexcept { return key < kMaxKeys && keys[key]; }
    /** @brief Sets key state. */
    void setKey(core::u16 key, bool pressed) noexcept { if (key < kMaxKeys) keys[key] = pressed; }
    /** @brief Gets axis value. */
    [[nodiscard]] float getAxis(core::u8 axisId) const noexcept { return axisId < kMaxAxes ? axes[axisId] : 0.f; }
    /** @brief Sets axis value. */
    void setAxis(core::u8 axisId, float value) noexcept { if (axisId < kMaxAxes) axes[axisId] = value; }
};

/**
 * @class InputManager
 * @brief Per-entity input manager with neural speed modulation.
 *
 * Ported from legacy engine/InputManager.hpp. Stores per-entity input state
 * and provides WASD movement computation with neural concentration modulation
 * and blink-based jumping.
 *
 * Neural speed modulation: concentration [0..1] → speed scale [0.70x..1.30x].
 * Blink detection uses rising-edge (previous=false, current=true) to prevent
 * repeated jumps. Grounded state prevents double-jump.
 */
class InputManager final : public core::NonCopyable<InputManager>
{
public:
    InputManager();
    ~InputManager();

    // --------------------------------------------------------------------- //
    //  Source management                                                      //
    // --------------------------------------------------------------------- //

    /** @brief Registers a new input source. */
    void addSource(std::unique_ptr<IInputSource> source);

    /** @brief Initializes all registered sources. */
    [[nodiscard]] core::Expected<void> init();

    /** @brief Polls all sources and produces the current-tick snapshot. */
    [[nodiscard]] core::Expected<void> poll();

    /** @brief Shuts down all sources. */
    void shutdown();

    // --------------------------------------------------------------------- //
    //  Per-entity state (ported from legacy)                                  //
    // --------------------------------------------------------------------- //

    /** @brief Sets key state for an entity. */
    void setKeyState(core::u32 entityId, core::u16 key, bool pressed);

    /** @brief Sets axis value for an entity. */
    void setAxis(core::u32 entityId, core::u8 axisId, float value);

    /**
     * @brief Sets neural control data for an entity.
     * @param entityId Entity to update.
     * @param alpha    Alpha band power.
     * @param beta     Beta band power.
     * @param concentration Concentration level [0..1].
     * @param blink    True if blink detected this tick.
     */
    void setNeural(core::u32 entityId, float alpha, float beta,
                   float concentration, bool blink);

    /** @brief Returns input state for an entity (nullptr if not found). */
    [[nodiscard]] const PerEntityInput *getState(core::u32 entityId) const;

    /** @brief Returns mutable input state for an entity (nullptr if not found). */
    [[nodiscard]] PerEntityInput *getStateMut(core::u32 entityId);

    /** @brief Returns or creates input state for an entity. */
    [[nodiscard]] PerEntityInput &getOrCreate(core::u32 entityId);

    /** @brief Removes an entity from the input manager. */
    void removeEntity(core::u32 entityId);

    /** @brief Tests whether an entity has input state. */
    [[nodiscard]] bool hasEntity(core::u32 entityId) const;

    /**
     * @brief Updates grounded state based on vertical velocity.
     * @param entityId Entity to check.
     * @param currentVelY Vertical velocity component.
     * @param threshold Velocity threshold for grounded detection (default 0.5f).
     */
    void updateGroundedState(core::u32 entityId, float currentVelY,
                             float threshold = 0.5f);

    /**
     * @brief Computes movement velocity from WASD + neural modulation.
     *
     * Applies neural concentration modulation:
     * - concentration [0..1] → speed scale [0.70x..1.30x]
     * - Blink detection (rising-edge) triggers jump if grounded
     *
     * @param entityId  Entity to compute for.
     * @param currentVel Current velocity (Y preserved for jump).
     * @param speed     Base movement speed (default 50).
     * @return Modified velocity vector.
     */
    [[nodiscard]] math::Vec3<float> computeMovementVelocity(
        core::u32 entityId,
        math::Vec3<float> currentVel,
        float speed = 50.f);

    // --------------------------------------------------------------------- //
    //  Global state                                                           //
    // --------------------------------------------------------------------- //

    /** @brief Returns the latest classical input state (global, not per-entity). */
    [[nodiscard]] const InputState& currentState() const noexcept;

    /** @brief Returns the latest neural input state (global, not per-entity). */
    [[nodiscard]] const NeuralInputState& currentNeuralState() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace lpl::input

#endif // LPL_INPUT_INPUTMANAGER_HPP
