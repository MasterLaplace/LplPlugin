/**
 * @file InputManager.hpp
 * @brief Gestionnaire d'inputs unifié (clavier, axes, neural) par entité.
 *
 * Stocke l'état courant de tous les inputs pour chaque entité contrôlée.
 * Utilisé côté serveur (état autoritaire depuis les paquets réseau) et
 * côté client (état local depuis GLFW/BCI + client-side prediction).
 *
 * Fournit une logique partagée de calcul de vélocité (WASD + modulation neurale)
 * via computeMovementVelocity() pour éviter la duplication serveur/client.
 *
 * @author MasterLaplace
 */

#pragma once

#include <cstdint>
#include <algorithm>
#include <unordered_map>
#include "Math.hpp"

// ─── Neural Input State ──────────────────────────────────────

/**
 * @brief État neural d'un joueur pour le calcul de mouvement.
 */
struct NeuralInputState {
    float alpha = 0.0f;
    float beta = 0.0f;
    float concentration = 0.5f;
    bool blinkDetected = false;
    bool blinkPrev = false; ///< Pour le rising-edge detection du blink
    bool isGrounded = true; ///< Vrai si l'entité est sur le sol (peut sauter)
};

// ─── Input State ─────────────────────────────────────────────

/**
 * @brief État complet des inputs d'une entité.
 *
 * Combine clavier (WASD), axes (joystick/souris), et données neurales.
 * L'état est idempotent : des paquets répétés ne cumulent pas.
 */
struct InputState {
    // ─── Keys ────────────────────────────────────────────────
    static constexpr uint16_t MAX_KEYS = 512u;
    bool keys[MAX_KEYS] = {};

    // ─── Axes ────────────────────────────────────────────────
    static constexpr uint8_t MAX_AXES = 16u;
    float axes[MAX_AXES] = {};

    // ─── Neural ──────────────────────────────────────────────
    NeuralInputState neural;

    // ─── Helpers ─────────────────────────────────────────────

    [[nodiscard]] bool getKey(uint16_t key) const noexcept
    {
        return key < MAX_KEYS && keys[key];
    }

    void setKey(uint16_t key, bool pressed) noexcept
    {
        if (key < MAX_KEYS)
            keys[key] = pressed;
    }

    [[nodiscard]] float getAxis(uint8_t axisId) const noexcept
    {
        return axisId < MAX_AXES ? axes[axisId] : 0.0f;
    }

    void setAxis(uint8_t axisId, float value) noexcept
    {
        if (axisId < MAX_AXES)
            axes[axisId] = value;
    }
};

// ─── Input Manager ───────────────────────────────────────────

/**
 * @brief Gestionnaire centralisé des inputs par entité.
 *
 * Chaque entité contrôlée (joueur) a un InputState.
 * Les systèmes écrivent dans l'InputManager (Network, BCI, GLFW callbacks)
 * et lisent depuis l'InputManager (MovementSystem, InputSendSystem).
 *
 * La méthode computeMovementVelocity() factorise la logique de calcul
 * de vélocité partagée entre le serveur autoritaire et le client-side prediction.
 */
class InputManager {
public:
    // ─── Key/Axis/Neural setters ─────────────────────────────

    void setKeyState(uint32_t entityId, uint16_t key, bool pressed)
    {
        _states[entityId].setKey(key, pressed);
    }

    void setAxis(uint32_t entityId, uint8_t axisId, float value)
    {
        _states[entityId].setAxis(axisId, value);
    }

    void setNeural(uint32_t entityId, float alpha, float beta, float concentration, bool blink)
    {
        auto &state = _states[entityId];
        state.neural.alpha = alpha;
        state.neural.beta = beta;
        state.neural.concentration = std::clamp(concentration, 0.0f, 1.0f);
        state.neural.blinkDetected = blink;
    }

    // ─── Getters ─────────────────────────────────────────────

    [[nodiscard]] const InputState *getState(uint32_t entityId) const
    {
        auto it = _states.find(entityId);
        return it != _states.end() ? &it->second : nullptr;
    }

    [[nodiscard]] InputState *getStateMut(uint32_t entityId)
    {
        auto it = _states.find(entityId);
        return it != _states.end() ? &it->second : nullptr;
    }

    /**
     * @brief Crée un InputState pour une entité (si absent).
     */
    InputState &getOrCreate(uint32_t entityId)
    {
        return _states[entityId];
    }

    /**
     * @brief Supprime l'InputState d'une entité (déconnexion).
     */
    void removeEntity(uint32_t entityId)
    {
        _states.erase(entityId);
    }

    /**
     * @brief Vérifie si une entité a un InputState enregistré.
     */
    [[nodiscard]] bool hasEntity(uint32_t entityId) const
    {
        return _states.count(entityId) > 0;
    }

    // ─── Movement Computation (shared server/client) ─────────

    /**
     * Standard WASD key codes (ASCII / GLFW compatible).
     * W=87, A=65, S=83, D=68
     */
    static constexpr uint16_t KEY_W = 87u;
    static constexpr uint16_t KEY_A = 65u;
    static constexpr uint16_t KEY_S = 83u;
    static constexpr uint16_t KEY_D = 68u;

    /**
     * @brief Met à jour l'état "grounded" basé sur la vélocité Y.
     *
     * Si la vélocité Y est petite et négative ou proche de 0, l'entité vient
     * de toucher le sol. Cela permet au saut suivant de se déclencher.
     *
     * @param entityId    ID de l'entité.
     * @param currentVelY Composante Y actuelle de la vélocité.
     * @param threshold   Seuil de vélocité Y pour considérer l'entité sur le sol (ex: 0.5).
     */
    void updateGroundedState(uint32_t entityId, float currentVelY, float threshold = 0.5f)
    {
        auto *state = getStateMut(entityId);
        if (!state)
            return;
        // Grounded si on descend lentement ou pas du tout (pas en saut/chute)
        state->neural.isGrounded = std::abs(currentVelY) < threshold;
    }

    /**
     * @brief Calcule la vélocité de déplacement à partir des inputs WASD + modulation neurale.
     *
     * Cette formule est identique côté serveur (autoritaire) et client (prediction).
     * La vélocité Y existante est préservée (gravité), seuls X et Z sont recalculés.
     *
     * @param entityId     ID de l'entité.
     * @param currentVel   Vélocité actuelle (pour préserver Y).
     * @param speed        Vitesse de déplacement de base (ex: 50.0f).
     * @return Nouvelle vélocité avec composantes X/Z recalculées.
     */
    Vec3 computeMovementVelocity(uint32_t entityId, Vec3 currentVel, float speed = 50.0f)
    {
        auto *state = getStateMut(entityId);
        if (!state)
            return currentVel;

        Vec3 newVel = currentVel;

        // Neural speed modulation: concentration [0..1] → scale [0.70x .. 1.30x]
        const float concentration = std::clamp(state->neural.concentration, 0.0f, 1.0f);
        const float neuralSpeedScale = 0.70f + concentration * 0.60f;

        // WASD → velocity
        const float moveRight = (state->getKey(KEY_D) ? 1.0f : 0.0f) - (state->getKey(KEY_A) ? 1.0f : 0.0f);
        const float moveForward = (state->getKey(KEY_W) ? 1.0f : 0.0f) - (state->getKey(KEY_S) ? 1.0f : 0.0f);

        newVel.x = moveRight * speed * neuralSpeedScale;
        newVel.z = moveForward * speed * neuralSpeedScale;

        // Blink jump — rising edge only (idempotent)
        // NE PEUT SAUTER QUE SI ON EST SUR LE SOL
        if (state->neural.isGrounded && state->neural.blinkDetected && !state->neural.blinkPrev)
            newVel.y += 10.0f;
        state->neural.blinkPrev = state->neural.blinkDetected;

        return newVel;
    }

private:
    std::unordered_map<uint32_t, InputState> _states;
};
