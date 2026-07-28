/**
 * @file AbstractWorld.hpp
 * @brief Creatures that keep living when nobody is looking, for almost nothing.
 *
 * Two independent studies arrive at the same architecture, which is a good reason
 * to trust it. Rain World splits every creature into an *abstract* form — a node
 * on a graph, no physics, no collision — and a *realised* one with a body. UE's
 * Mass framework calls the same thing LOD 2: off-screen, an entity is a row of
 * data that still eats and breeds and costs nothing to draw.
 *
 * They are one mechanism, so this is one mechanism. An abstract creature is an
 * id, a room, and a personality derived from that id; realising it means giving
 * it a body; abstracting it means taking the body away and keeping everything
 * that matters.
 *
 * The interesting part is the **budget**, because "realise what is near the
 * player" is not a policy — it is a hope. The policy is a score per realised
 * room, and the highest score is the one that goes:
 *
 * @f[ S = t_{\text{since visit}} + k \cdot n_{\text{room changes}} - B_{\text{adjacent}}
 *       - B_{\text{predicted}} + P_{\text{unlikely}} @f]
 *
 * A room next door to the player is protected, so a threat at the door does not
 * evaporate the moment it stops being watched. The room the player is *probably*
 * about to enter is protected harder, so walking through a door does not stutter.
 * Everything else ages out.
 *
 * @warning The budget is counted in **rooms**, never in milliseconds. A wall
 *          clock would make two machines abstract different rooms and the
 *          simulations would diverge — which is the failure a seed-shipping
 *          architecture cannot absorb.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_AI_ABSTRACTWORLD_HPP
#    define LPL_AI_ABSTRACTWORLD_HPP

#    include <lpl/ai/Personality.hpp>
#    include <lpl/core/Types.hpp>
#    include <lpl/std/vector.hpp>

namespace lpl::ai {

/// A room's index in the abstract graph.
using RoomId = core::u32;

/// Value meaning "no room".
inline constexpr RoomId kNoRoom = 0xFFFFFFFFu;

/**
 * @struct AbstractCreature
 * @brief A creature reduced to what survives being forgotten.
 *
 * No position, no velocity, no body. Its personality is not stored because it is
 * a function of @c id (see @ref personalityOf), which is what makes abstraction
 * lossless rather than lossy.
 */
struct AbstractCreature {
    core::u32 id{0u};       ///< Identity, and the seed of everything derived.
    core::u32 species{0u};  ///< Species salt.
    RoomId room{kNoRoom};   ///< Where it currently is.
    core::u32 energy{100u}; ///< Coarse condition, ticked even while abstract.
    bool realised{false};   ///< Whether a body currently exists for it.
};

/**
 * @struct AbstractRoom
 * @brief A node of the world graph, and what the budget needs to know about it.
 */
struct AbstractRoom {
    lpl::pmr::vector<RoomId> exits; ///< Adjacent rooms.
    core::u32 lastVisitTick{0u};    ///< When a focus was last here.
    core::u32 changesSinceVisit{0u};///< Room transitions elsewhere since then.
    bool realised{false};           ///< Whether this room's creatures have bodies.
};

/**
 * @struct RealizationBudget
 * @brief The constants of the deletion score, and the cap it enforces.
 */
struct RealizationBudget {
    core::u32 maxRealisedRooms{6u};    ///< Rooms allowed a body at once.
    core::u32 changeWeight{4u};        ///< @f$k@f$: weight of room changes since the visit.
    core::u32 adjacentBonus{100u};     ///< Protection for a room next to the focus.
    core::u32 predictedBonus{200u};    ///< Protection for the room the focus is heading to.
    core::u32 unlikelyPenalty{20u};    ///< Extra ageing for a room the focus is moving away from.
};

/**
 * @class AbstractWorld
 * @brief The room graph, its creatures, and the realisation policy.
 */
class AbstractWorld {
public:
    /**
     * @brief Adds a room.
     * @return Its id.
     */
    RoomId addRoom();

    /**
     * @brief Joins two rooms, both ways.
     * @param a First room.
     * @param b Second room.
     */
    void connect(RoomId a, RoomId b);

    /**
     * @brief Adds a creature to a room.
     * @param id      Identity; also the seed of its personality.
     * @param species Species salt.
     * @param room    Starting room.
     * @return Index into @ref creatures.
     */
    core::u32 addCreature(core::u32 id, core::u32 species, RoomId room);

    /// @brief Moves a creature, and ages every room's "changes since visit".
    void moveCreature(core::u32 index, RoomId destination);

    /**
     * @brief Records that a focus (a player, a camera) is in a room this tick.
     * @param room       Where the focus is.
     * @param predicted  Where it is likely to go next; @ref kNoRoom when unknown.
     * @param tick       The current tick.
     */
    void observe(RoomId room, RoomId predicted, core::u32 tick);

    /**
     * @brief The deletion score of a realised room. Higher means "drop this one".
     * @param room   Room to score.
     * @param budget The constants.
     * @param tick   Current tick.
     * @return The score.
     */
    [[nodiscard]] core::i32 deletionScore(RoomId room, const RealizationBudget &budget, core::u32 tick) const;

    /**
     * @brief Realises what should have a body and abstracts what should not.
     *
     * Realises the focus's room and its neighbours first, then, while over
     * budget, abstracts whichever realised room scores highest. Both directions
     * in one call, because doing them separately lets the count overshoot between
     * them.
     *
     * @param budget The constants and the cap.
     * @param tick   Current tick.
     * @return Number of rooms whose realisation state changed.
     */
    core::u32 enforceBudget(const RealizationBudget &budget, core::u32 tick);

    /**
     * @brief Ticks every abstract creature: the cheap simulation.
     *
     * This is the whole point of the split. An abstract creature still loses
     * condition and still migrates, so a region left alone for an hour has
     * genuinely changed when the player returns — the illusion of a world that
     * does not wait.
     *
     * @param tick Current tick.
     * @return Number of creatures that moved.
     */
    core::u32 tickAbstract(core::u32 tick);

    [[nodiscard]] const lpl::pmr::vector<AbstractCreature> &creatures() const noexcept { return _creatures; }
    [[nodiscard]] const lpl::pmr::vector<AbstractRoom> &rooms() const noexcept { return _rooms; }
    [[nodiscard]] core::u32 realisedRoomCount() const noexcept;
    [[nodiscard]] core::u32 realisedCreatureCount() const noexcept;

    /// @brief The personality of a creature, derived rather than stored.
    [[nodiscard]] PersonalityTraits personality(core::u32 index) const;

    /// @brief FNV-1a fold of every creature's state, for determinism checks.
    [[nodiscard]] core::u32 fold() const;

private:
    lpl::pmr::vector<AbstractRoom> _rooms;
    lpl::pmr::vector<AbstractCreature> _creatures;
    RoomId _focus{kNoRoom};
    RoomId _predicted{kNoRoom};
};

} // namespace lpl::ai

#endif // LPL_AI_ABSTRACTWORLD_HPP
