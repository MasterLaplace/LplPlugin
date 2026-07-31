/**
 * @file Herd.hpp
 * @brief A population of animals walking a world: flocking, scent, food, census.
 *
 * The fourth thing written inside a sample that belongs to a module. An animal in a
 * simulated world does the same four things whatever the game is: it moves with its
 * kind, it follows a gradient of something it wants, it refuses to walk where it
 * cannot stand, and it eats. What differs is where the ground is and what counts as
 * food — two callbacks.
 *
 * Two decisions in here are load-bearing and were each learned from a picture:
 *
 *  - the flock's own integration is DISCARDED. The boid rules decide where an animal
 *    wants to go; the terrain decides where it may stand. Taking the position back
 *    from the flock put bodies inside rock before any check could refuse them.
 *  - the axes are tested SEPARATELY. An animal that cannot move diagonally into a
 *    corner can still slide along the wall, which is what stops a herd from piling
 *    up against a cliff and vibrating.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-28
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ECOLOGY_HERD_HPP
#    define LPL_ECOLOGY_HERD_HPP

#    include <lpl/ai/Personality.hpp>
#    include <lpl/ai/StigmergyField.hpp>
#    include <lpl/ai/Swarm.hpp>
#    include <lpl/ecology/Genome.hpp>
#    include <lpl/ecology/Populations.hpp>
#    include <lpl/procgen/FixedMath.hpp>
#    include <lpl/procgen/Random.hpp>
#    include <lpl/std/vector.hpp>

namespace lpl::ecology {

/**
 * @struct HerdMember
 * @brief One animal: a body the flock moves, a genome, and who it is.
 */
struct HerdMember {
    ai::Boid body{};
    Genome genome{};
    core::u32 id{0u};
    core::u32 species{0u};                       ///< Index into the caller's species table.
    math::Fixed32 heading{math::Fixed32::one()}; ///< Unit facing, X — for drawing.
    math::Fixed32 headingZ{};                    ///< Unit facing, Z.
};

/**
 * @struct HerdParams
 * @brief How a species moves, per trophic role.
 */
struct HerdParams {
    core::u32 speciesCount{2u};
    core::f32 separationHunter{1.1f};
    core::f32 separationGrazer{0.9f};
    core::f32 alignmentHunter{0.5f};
    core::f32 alignmentGrazer{0.8f};
    core::f32 cohesionHunter{0.25f};
    core::f32 cohesionGrazer{0.5f};
    core::i32 neighbourHunter{10};
    core::i32 neighbourGrazer{6};
    /// How hard a scent gradient pulls, before personality scales it.
    math::Fixed32 scentPull{math::Fixed32::fromFloat(0.06f)};
    math::Fixed32 step{math::Fixed32::fromRaw(1092)}; ///< One tick, in seconds (60 Hz).
};

/**
 * @class Herd
 * @brief The animals, and one step of what they do.
 */
class Herd {
public:
    [[nodiscard]] core::u32 size() const noexcept { return static_cast<core::u32>(_members.size()); }
    [[nodiscard]] bool empty() const noexcept { return _members.empty(); }
    [[nodiscard]] HerdMember &at(core::u32 index) noexcept { return _members[index]; }
    [[nodiscard]] const HerdMember &at(core::u32 index) const noexcept { return _members[index]; }
    void clear() noexcept { _members.clear(); }
    void add(const HerdMember &member) { _members.push_back(member); }

    /** @brief Animals of one species, for a census or a HUD. */
    [[nodiscard]] core::u32 countSpecies(core::u32 species) const noexcept;

    /**
     * @brief Removes one animal of a species; used to reconcile with the census.
     * @return True when one was found and removed.
     */
    bool removeOne(core::u32 species) noexcept;

    /**
     * @brief One tick: flock, follow the scent, eat, then move where allowed.
     *
     * @param field     The stigmergy field both roles read. A grazer climbs it toward
     *                  pasture; a hunter climbs the SAME field because it leads to
     *                  grazers. Nothing tells the hunter where the herd is.
     * @param toFieldCell (worldX, worldZ, outX, outZ) -> bool; false when the animal
     *                    is outside the field's window (a streamed world's window
     *                    follows the player, so this happens constantly).
     * @param walkable  (x, z) -> bool, in world units.
     * @param graze     (worldX, worldZ) -> void, called where a grazer stands.
     */
    template <typename ToFieldCell, typename Walkable, typename Graze>
    void step(const HerdParams &params, const ai::StigmergyField &field, ToFieldCell &&toFieldCell,
              Walkable &&walkable, Graze &&graze)
    {
        if (_members.empty())
            return;

        for (core::u32 species = 0u; species < params.speciesCount; ++species)
        {
            _flock.clear();
            for (core::u32 i = 0u; i < _members.size(); ++i)
                if (_members[i].species == species)
                    _flock.push_back(_members[i].body);
            if (_flock.empty())
                continue;

            const bool hunter = species == 1u;
            ai::BoidParams boids;
            boids.separationWeight = hunter ? params.separationHunter : params.separationGrazer;
            boids.alignmentWeight = hunter ? params.alignmentHunter : params.alignmentGrazer;
            boids.cohesionWeight = hunter ? params.cohesionHunter : params.cohesionGrazer;
            boids.neighbourRadius =
                math::Fixed32::fromInt(hunter ? params.neighbourHunter : params.neighbourGrazer);

            // dt is explicit, and the integration the flock performed is thrown away:
            // only the velocities are taken back. See the file comment.
            ai::stepBoids(&_flock[0], static_cast<core::u32>(_flock.size()), boids, params.step);

            core::u32 cursor = 0u;
            for (core::u32 i = 0u; i < _members.size(); ++i)
                if (_members[i].species == species)
                {
                    _members[i].body.vx = _flock[cursor].vx;
                    _members[i].body.vz = _flock[cursor].vz;
                    ++cursor;
                }
        }

        for (core::u32 i = 0u; i < _members.size(); ++i)
        {
            HerdMember &member = _members[i];
            const ai::PersonalityTraits traits = ai::personalityOf(member.id, member.species);

            const core::i32 worldX = member.body.x.toInt();
            const core::i32 worldZ = member.body.z.toInt();

            core::u32 cellX = 0u;
            core::u32 cellZ = 0u;
            if (toFieldCell(worldX, worldZ, cellX, cellZ))
            {
                const core::u32 direction = field.gradientDirection(1u, cellX, cellZ, true);
                if (direction != ai::StigmergyField::kNoDirection)
                {
                    const math::Fixed32 pull = params.scentPull * (math::Fixed32::half() + traits.energy);
                    member.body.vx = member.body.vx + math::Fixed32::fromInt(procgen::kNeighbor8X[direction]) * pull;
                    member.body.vz = member.body.vz + math::Fixed32::fromInt(procgen::kNeighbor8Z[direction]) * pull;
                }
                if (member.species == 0u)
                    graze(worldX, worldZ);
            }

            // Heading from the velocity, and the PACE from the genome and the
            // personality — not from the velocity's magnitude. That separation is
            // what keeps a chain of scent impulses from accumulating into a bolt:
            // the flock and the scent decide the DIRECTION, the genome decides how
            // fast this animal can possibly travel.
            const math::Fixed32 lengthSquared =
                member.body.vx * member.body.vx + member.body.vz * member.body.vz;
            const math::Fixed32 length = procgen::fixedSqrt(lengthSquared);
            if (length.raw() > 256)
            {
                member.heading = member.body.vx / length;
                member.headingZ = member.body.vz / length;
            }

            const math::Fixed32 pace = member.genome.maxSpeed * params.step *
                                       (math::Fixed32::fromFloat(0.7f) + traits.energy * math::Fixed32::half());
            const math::Fixed32 tryX = member.body.x + member.heading * pace;
            const math::Fixed32 tryZ = member.body.z + member.headingZ * pace;

            // Axes tested separately: an animal blocked diagonally slides along the
            // wall instead of stopping dead against it.
            const bool freeX = walkable(tryX, member.body.z);
            const bool freeZ = walkable(member.body.x, tryZ);
            if (freeX && freeZ && walkable(tryX, tryZ))
            {
                member.body.x = tryX;
                member.body.z = tryZ;
            }
            else if (freeX)
                member.body.x = tryX;
            else if (freeZ)
                member.body.z = tryZ;
            else
            {
                // Cornered: turn around rather than freeze. A frozen animal in a
                // corner reads as a bug even when it is a correct refusal to move.
                member.heading = math::Fixed32{} - member.heading;
                member.headingZ = math::Fixed32{} - member.headingZ;
            }
        }
    }

    /**
     * @brief Brings the number of bodies in line with what a census says exists.
     *
     * The population model is the authority — it integrates births and deaths — and
     * the bodies are its view. Letting the bodies drift from it is how a species goes
     * extinct in the numbers while still walking around on screen.
     *
     * @param spawn (species) -> bool, creating one animal; false when it could not.
     */
    template <typename CountFor, typename Spawn>
    void reconcile(core::u32 speciesCount, CountFor &&countFor, Spawn &&spawn)
    {
        for (core::u32 species = 0u; species < speciesCount; ++species)
        {
            const core::u32 wanted = countFor(species);
            core::u32 present = countSpecies(species);
            while (present < wanted && spawn(species))
                ++present;
            while (present > wanted && removeOne(species))
                --present;
        }
    }

private:
    lpl::pmr::vector<HerdMember> _members;
    lpl::pmr::vector<ai::Boid> _flock;
};

} // namespace lpl::ecology

// Out-of-line definitions: consumed header-only, the freestanding kernel included.
#    include <lpl/ecology/Herd.inl>

#endif // LPL_ECOLOGY_HERD_HPP
