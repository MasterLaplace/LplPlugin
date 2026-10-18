#include <lpl/ai/StigmergyField.hpp>
#include <lpl/ecology/Genome.hpp>
#include <lpl/ecology/Herd.hpp>
#include <lpl/ecs/Archetype.hpp>
#include <lpl/ecs/Partition.hpp>
#include <lpl/ecs/Registry.hpp>
#include <lpl/engine/systems/CreatureSystems.hpp>
#include <lpl/math/FixedPoint.hpp>
#include <lpl/math/Vec3.hpp>

#include <cmath>
#include <cstdio>
#include <string>

static int failures = 0;

static void check(bool ok, const std::string &what)
{
    std::printf("  %s: %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok)
        ++failures;
}

namespace {

constexpr lpl::core::u32 kSize = 33u;
constexpr lpl::core::u32 kChannels = 6u;
constexpr lpl::core::u32 kPreyX = 16u;
constexpr lpl::core::u32 kPreyZ = 16u;

/**
 * @brief Flat, endless ground: everything is standable and nothing grows on it.
 *
 * @details This is the reason engine::ITerrainQuery is an interface and not a lambda the owner hands down: a test
 *          can BE the terrain, so the locomotion system is measurable without generating a heightfield.
 */
struct OpenGround final : public lpl::engine::ITerrainQuery {
    [[nodiscard]] bool standable(lpl::math::Fixed32, lpl::math::Fixed32) const override { return true; }
    bool consumePlantAt(lpl::core::i32, lpl::core::i32) override { return false; }
};

/**
 * @brief Ground with a wall: everything east of @c kWallX is rock, and there is grass.
 */
struct WalledMeadow final : public lpl::engine::ITerrainQuery {
    static constexpr lpl::core::i32 kWallX = 20;

    [[nodiscard]] bool standable(lpl::math::Fixed32 x, lpl::math::Fixed32) const override { return x.toInt() < kWallX; }
    bool consumePlantAt(lpl::core::i32, lpl::core::i32) override
    {
        ++meals;
        return true;
    }

    lpl::core::u32 meals{0u};
};

/**
 * @brief The archetype a creature carries now that its body is an entity.
 *
 * @details Facing is a component now: it is what a walker advances along, so the state that decides the next
 *          position lives in the registry with the body.
 */
lpl::ecs::Archetype creatureArchetype()
{
    lpl::ecs::Archetype archetype;
    archetype.add(lpl::ecs::ComponentId::Position);
    archetype.add(lpl::ecs::ComponentId::Velocity);
    archetype.add(lpl::ecs::ComponentId::Genome);
    archetype.add(lpl::ecs::ComponentId::Creature);
    archetype.add(lpl::ecs::ComponentId::Heading);
    return archetype;
}

/**
 * @brief Writes one creature's components.
 *
 * @return false when the entity is unreachable.
 */
bool writeCreature(lpl::ecs::Registry &registry, lpl::ecs::EntityId id, lpl::core::u32 species, lpl::core::i32 x,
                   lpl::core::i32 z)
{
    auto ref = registry.resolve(id);
    if (!ref.has_value())
        return false;
    for (const auto &part : registry.partitions())
    {
        if (!part || !part->archetype().has(lpl::ecs::ComponentId::Creature))
            continue;
        const auto &chunks = part->chunks();
        if (ref.value().chunkIndex >= static_cast<lpl::core::u32>(chunks.size()) || !chunks[ref.value().chunkIndex])
            continue;
        auto &chunk = *chunks[ref.value().chunkIndex];
        const auto entities = chunk.entities();
        if (ref.value().localIndex >= entities.size() || entities[ref.value().localIndex] != id)
            continue;
        const lpl::core::u32 slot = ref.value().localIndex;
        auto *p =
            static_cast<lpl::math::Vec3<lpl::math::Fixed32> *>(chunk.writeComponent(lpl::ecs::ComponentId::Position));
        auto *c = static_cast<lpl::core::u32 *>(chunk.writeComponent(lpl::ecs::ComponentId::Creature));
        auto *g = static_cast<lpl::ecology::Genome *>(chunk.writeComponent(lpl::ecs::ComponentId::Genome));
        auto *h = static_cast<lpl::math::Fixed32 *>(chunk.writeComponent(lpl::ecs::ComponentId::Heading));
        if (p == nullptr || c == nullptr || g == nullptr || h == nullptr)
            return false;
        p[slot] = {lpl::math::Fixed32::fromInt(x), lpl::math::Fixed32{}, lpl::math::Fixed32::fromInt(z)};
        c[slot * 2u] = species;
        c[slot * 2u + 1u] = 1u + slot;
        g[slot] = lpl::ecology::Genome{};
        g[slot].maxSpeed = lpl::math::Fixed32::fromInt(4);
        g[slot].size = lpl::math::Fixed32::one();
        h[slot * 2u] = lpl::math::Fixed32::one();
        h[slot * 2u + 1u] = lpl::math::Fixed32{};
        return true;
    }
    return false;
}

/**
 * @brief Reads a creature's position back out of the registry.
 *
 * @details From the WRITE side, like every creature system: the buffer swap is a phase callback the Engine installs
 *          after Physics, so a world whose systems are all PrePhysics never swaps, and its front buffer holds
 *          whatever a component was born with. Reading the front side here would report every animal at its spawn.
 */
lpl::math::Vec3<lpl::math::Fixed32> positionOf(lpl::ecs::Registry &registry, lpl::ecs::EntityId id)
{
    auto ref = registry.resolve(id);
    if (!ref.has_value())
        return {};
    for (const auto &part : registry.partitions())
    {
        if (!part || !part->archetype().has(lpl::ecs::ComponentId::Creature))
            continue;
        const auto &chunks = part->chunks();
        if (ref.value().chunkIndex >= static_cast<lpl::core::u32>(chunks.size()) || !chunks[ref.value().chunkIndex])
            continue;
        auto &chunk = *chunks[ref.value().chunkIndex];
        const auto entities = chunk.entities();
        if (ref.value().localIndex >= entities.size() || entities[ref.value().localIndex] != id)
            continue;
        const auto *p = static_cast<const lpl::math::Vec3<lpl::math::Fixed32> *>(
            chunk.latestComponent(lpl::ecs::ComponentId::Position));
        return p != nullptr ? p[ref.value().localIndex] : lpl::math::Vec3<lpl::math::Fixed32>{};
    }
    return {};
}

/**
 * @brief Places @p count hunters on one arc, all approaching the prey from the same side.
 *
 * @details A tight clump due south of the prey: if nothing pushed them apart they would stay a clump all the way
 *          in.
 */
void seedPack(lpl::ecs::Registry &registry, lpl::ecology::Herd &herd, lpl::core::u32 count)
{
    const lpl::ecs::Archetype archetype = creatureArchetype();
    for (lpl::core::u32 i = 0u; i < count; ++i)
    {
        auto created = registry.createEntity(archetype);
        if (!created.has_value())
            continue;
        (void) writeCreature(registry, created.value(), 1u, static_cast<lpl::core::i32>(kPreyX + i) - 2,
                             static_cast<lpl::core::i32>(kPreyZ) + 8);
        herd.add(created.value());
    }
}

/**
 * @brief How widely the pack is distributed in angle around the prey, in radians.
 *
 * @details Circular standard deviation rather than a plain spread: an angle wraps, and a pack straddling due north
 *          would otherwise look maximally spread when it is tightly clumped. The mean resultant length R near 1
 *          means "all pointing the same way"; near 0 means "spread out".
 */
double angularSpread(lpl::ecs::Registry &registry, const lpl::ecology::Herd &herd)
{
    double sumSin = 0.0;
    double sumCos = 0.0;
    lpl::core::u32 n = 0u;
    for (lpl::core::u32 i = 0u; i < herd.size(); ++i)
    {
        const lpl::ecs::EntityId body = herd.at(i);
        const lpl::core::u32 *creature = lpl::ecology::Herd::creatureOf(registry, body);
        if (creature == nullptr || creature[0] != 1u)
            continue;
        const lpl::math::Vec3<lpl::math::Fixed32> at = positionOf(registry, body);
        const double dx = at.x.toFloat() - static_cast<double>(kPreyX);
        const double dz = at.z.toFloat() - static_cast<double>(kPreyZ);
        if (dx == 0.0 && dz == 0.0)
            continue;
        const double angle = std::atan2(dz, dx);
        sumSin += std::sin(angle);
        sumCos += std::cos(angle);
        ++n;
    }
    if (n == 0u)
        return 0.0;
    const double resultant = std::sqrt(sumSin * sumSin + sumCos * sumCos) / static_cast<double>(n);
    return std::sqrt(-2.0 * std::log(resultant < 1e-9 ? 1e-9 : resultant));
}

/**
 * @brief How many of the four quadrants around the prey hold at least one hunter.
 *
 * @details The structural reading of "encirclement": a spread in radians can grow because the pack fanned out on
 *          ONE side, which is not surrounding anything. Counting occupied quadrants cannot be fooled that way, and
 *          it needs no threshold — four quadrants is the whole scale.
 */
lpl::core::u32 quadrantsOccupied(lpl::ecs::Registry &registry, const lpl::ecology::Herd &herd)
{
    bool seen[4]{};
    for (lpl::core::u32 i = 0u; i < herd.size(); ++i)
    {
        const lpl::ecs::EntityId body = herd.at(i);
        const lpl::core::u32 *creature = lpl::ecology::Herd::creatureOf(registry, body);
        if (creature == nullptr || creature[0] != 1u)
            continue;
        const lpl::math::Vec3<lpl::math::Fixed32> at = positionOf(registry, body);
        const bool east = at.x.toFloat() >= static_cast<double>(kPreyX);
        const bool south = at.z.toFloat() >= static_cast<double>(kPreyZ);
        seen[(east ? 1u : 0u) | (south ? 2u : 0u)] = true;
    }
    return static_cast<lpl::core::u32>(seen[0]) + seen[1] + seen[2] + seen[3];
}

/**
 * @brief Runs a hunt at a given kin-repulsion strength.
 *
 * @details One term varies and nothing else does: attraction to the prey is fixed, and the pack's repulsion from
 *          its own scent is the independent variable. The scent is long-range: it has to cross a field. The prey
 *          stands still and smells of prey; keeping it fixed isolates the pack's geometry from the chase.
 *
 * The whole tick runs as systems, in the order the scheduler derives: mark, forget, steer, flock, walk. Nothing
 * about an animal is left in a container's loop, and no buffer swap is needed — every creature system reads the
 * write side, which is what makes them work in a world whose systems are all PrePhysics and which therefore never
 * swaps.
 *
 * @return How spread the pack got.
 */
double hunt(float kinWeight, lpl::core::u32 ticks, lpl::core::u32 *outQuadrants = nullptr)
{
    lpl::ai::StigmergyField field{kSize, kSize, kChannels};

    lpl::ecology::HerdParams params;
    lpl::ecology::applyDefaultScents(params);
    params.scentPull = lpl::math::Fixed32::fromFloat(0.5f);
    params.scent[1].palate = lpl::ai::ScentPalate{};
    params.scent[1].palate.add(lpl::ai::ScentChannel::Herbivore, lpl::math::Fixed32::one());
    if (kinWeight != 0.0f)
        params.scent[1].palate.add(lpl::ai::ScentChannel::Carnivore, lpl::math::Fixed32::fromFloat(kinWeight));

    lpl::ecs::Registry registry;
    lpl::ecology::Herd herd;
    herd.bind(registry);
    seedPack(registry, herd, 6u);

    lpl::ai::StigmergyParams physics;
    physics.evaporation = 0.995f;
    physics.diffusion = 0.22f;
    physics.floor = 0.0005f;

    const lpl::engine::systems::CreatureFieldView window{0, 0, kSize, kSize};
    OpenGround ground;
    lpl::engine::systems::ScentDepositSystem deposit{registry, field, params, window};
    lpl::engine::systems::ScentFieldSystem evaporate{field, physics};
    lpl::engine::systems::ScentSteeringSystem steer{registry, field, params, window};
    lpl::engine::systems::FlockingSystem flock{registry, params};
    lpl::engine::systems::LocomotionSystem walk{registry, params, ground};

    for (lpl::core::u32 tick = 0u; tick < ticks; ++tick)
    {
        field.deposit(static_cast<lpl::core::u32>(lpl::ai::ScentChannel::Herbivore), kPreyX, kPreyZ,
                      lpl::math::Fixed32::fromInt(40));
        deposit.execute(0.016f);
        evaporate.execute(0.016f);
        steer.execute(0.016f);
        flock.execute(0.016f);
        walk.execute(0.016f);
    }
    if (outQuadrants != nullptr)
        *outQuadrants = quadrantsOccupied(registry, herd);
    return angularSpread(registry, herd);
}

/**
 * @brief A channel means something: a palate reads signs, not one channel.
 *
 * @details Food lies to the east, a predator to the west. The predator's own palate must not send it into its own
 *          scent.
 */
void checkAChannelMeansSomething()
{
    std::printf("-- a palate reads signs, not one channel --\n");
    lpl::ai::StigmergyField field{9u, 9u, kChannels};
    field.deposit(static_cast<lpl::core::u32>(lpl::ai::ScentChannel::Plant), 8u, 4u, lpl::math::Fixed32::fromInt(50));
    field.deposit(static_cast<lpl::core::u32>(lpl::ai::ScentChannel::Carnivore), 0u, 4u,
                  lpl::math::Fixed32::fromInt(50));
    lpl::ai::StigmergyParams physics;
    physics.evaporation = 0.999f;
    physics.diffusion = 0.25f;
    physics.floor = 0.0001f;
    for (lpl::core::u32 i = 0u; i < 60u; ++i)
        field.step(physics);

    const lpl::ecology::SpeciesScent grazer = lpl::ecology::defaultGrazerScent();
    const lpl::core::u32 move = field.palateDirection(grazer.palate, 4u, 4u);
    check(move != lpl::ai::StigmergyField::kNoDirection, "a grazer between food and a predator moves");
    if (move != lpl::ai::StigmergyField::kNoDirection)
        check(lpl::procgen::kNeighbor8X[move] > 0, "and it moves AWAY from the predator, toward the food");

    const lpl::ecology::SpeciesScent hunter = lpl::ecology::defaultHunterScent();
    check(hunter.depositChannel == static_cast<lpl::core::u32>(lpl::ai::ScentChannel::Carnivore),
          "a hunter marks the carnivore channel");
    check(grazer.depositChannel == static_cast<lpl::core::u32>(lpl::ai::ScentChannel::Herbivore),
          "a grazer marks the herbivore channel");
    check(grazer.depositChannel != hunter.depositChannel, "so the two do not smell alike");
}

/**
 * @brief An empty palate is not a silent no-op: it stands still.
 */
void checkAnEmptyPalateIsNotASilentNoOp()
{
    std::printf("\n-- an empty palate stands still --\n");
    lpl::ai::StigmergyField field{5u, 5u, kChannels};
    field.deposit(1u, 4u, 2u, lpl::math::Fixed32::fromInt(10));
    const lpl::ai::ScentPalate nothing;
    check(field.palateDirection(nothing, 2u, 2u) == lpl::ai::StigmergyField::kNoDirection,
          "an animal that cares about nothing does not wander");
}

/**
 * @brief Encirclement, measured: it emerges, unscripted.
 *
 * @details The assertion is MONOTONICITY, not a threshold. A fixed margin would be a number chosen to make today's
 *          tuning pass, and would say nothing about whether the mechanism works; "push harder and they spread
 *          further" is the actual claim, it can fail, and it stays true if the defaults are retuned.
 *
 * The SURROUND is measured on a longer run than the spread, and the reason is a measurement, not a preference: at
 * 200 ticks both packs sit in 2 quadrants out of 4, because 200 ticks is the APPROACH — the pack is still walking
 * in from the south and cannot be on the far side of something it has not reached. Surrounding requires arriving
 * first. At 600 the same parameters give 3 quadrants with kin repulsion and 2 without.
 *
 * SURROUNDING, not merely fanning out. The previous version of this check was `firm > none * 1.5` — a multiplier
 * picked so the tuning of the day passed, which is trap number one in this repository's own list, and it duly
 * failed the first time the boid separation radius was set properly. Quadrants have no threshold to tune: four is
 * the whole scale, and a pack spread across one side of the prey cannot score three.
 */
void checkEncirclementEmerges()
{
    std::printf("\n-- encirclement emerges, unscripted --\n");
    lpl::core::u32 quadrantsNone = 0u;
    lpl::core::u32 quadrantsFirm = 0u;
    const double none = hunt(0.0f, 200u);
    const double mild = hunt(-0.6f, 200u);
    const double firm = hunt(-2.0f, 200u);
    const double firmLong = hunt(-2.0f, 600u, &quadrantsFirm);
    (void) hunt(0.0f, 600u, &quadrantsNone);
    std::printf("  angular spread around the prey: none %.3f, mild %.3f, firm %.3f rad\n", none, mild, firm);
    std::printf("  after 600 ticks: firm spread %.3f rad, quadrants none %u/4, firm %u/4\n", firmLong, quadrantsNone,
                quadrantsFirm);
    check(mild > none, "any kin repulsion spreads the pack");
    check(firm > mild, "and more of it spreads them further");
    check(quadrantsFirm > quadrantsNone, "and they end up on more SIDES of the prey, not just further apart");
}

/**
 * @brief Determinism, because this feeds an authoritative simulation: the same hunt twice ends identically.
 */
void checkTheSameHuntTwiceEndsIdentically()
{
    std::printf("\n-- the same hunt twice --\n");
    check(hunt(-0.6f, 90u) == hunt(-0.6f, 90u), "two identical hunts end identically");
}

/**
 * @brief The same behaviour, as systems over entities.
 *
 * @details The point of the migration: a creature is an ENTITY, its genome is a COMPONENT, and what it does is a
 *          SYSTEM. Sixteen systems existed before these and not one was about anything alive — every animal's
 *          behaviour sat in a 1952-line sample where nothing could query it.
 *
 * A grazer and a hunter stand on the SAME cell. Under the old hard-coded channel they would have moved the same
 * way; they read the same field and must now move differently, because they weigh it differently. They are written
 * through the chunk, the way a system sees them: the registry has no per-entity component accessor, and inventing
 * one here would be a second path into storage that the systems do not use.
 *
 * No publish, deliberately. The systems read the WRITE side, so setup written there is what they see — and that is
 * not a convenience, it is the fix for a live bug: the swap is a phase callback the Engine installs after Physics,
 * so a world whose systems are all PrePhysics never swaps at all. While these systems read the front buffer they
 * saw every creature at (0,0,0), and "both creatures marked the field" passed green because the origin happens to
 * be inside the window.
 *
 * Grass to the EAST, a herd of prey to the WEST. Chosen so the two palates disagree: the grazer wants the grass,
 * the hunter wants the prey. The first version of this put the grass east and a rival predator west, and BOTH
 * animals moved east — the deer fleeing the predator, the wolf repelled by a rival. Correct behaviour, and a
 * scenario that discriminated nothing. A test where two different rules give the same answer proves only that
 * something moved. Both sources are the SAME distance from the pair, so neither wins by being nearer, and both
 * PERSISTENT: a patch of grass and a herd of prey are standing facts, and one deposit does not survive ten cells of
 * diffusion.
 *
 * The environment diffuses WITHOUT the creatures marking it. They stand on the same cell, so their own marks build
 * a local mound that drowns anything ten cells away — and a test that measured that would be measuring the deposit,
 * not the steering. The two are checked apart.
 *
 * The field system is what fixes the bug this slice found: evaporation used to live in one branch of TerrainWorld's
 * tick, so a bounded map never forgot anything. A system cannot be forgotten in one branch.
 */
void checkCreaturesAreEntitiesAndBehaviourIsSystems()
{
    std::printf("\n-- creatures as entities, behaviour as systems --\n");
    lpl::ecs::Registry registry;
    lpl::ai::StigmergyField field{kSize, kSize, kChannels};

    lpl::ecology::HerdParams params;
    lpl::ecology::applyDefaultScents(params);
    params.scentPull = lpl::math::Fixed32::fromFloat(0.5f);

    lpl::engine::systems::CreatureFieldView view{0, 0, kSize, kSize};

    lpl::ecs::Archetype archetype;
    archetype.add(lpl::ecs::ComponentId::Position);
    archetype.add(lpl::ecs::ComponentId::Velocity);
    archetype.add(lpl::ecs::ComponentId::Genome);
    archetype.add(lpl::ecs::ComponentId::Creature);
    auto first = registry.createEntity(archetype);
    auto second = registry.createEntity(archetype);
    check(first.has_value() && second.has_value(), "two creature entities exist");

    constexpr lpl::core::u32 kGrazer = 0u;
    constexpr lpl::core::u32 kHunter = 1u;
    for (const auto &part : registry.partitions())
    {
        if (!part || !part->archetype().has(lpl::ecs::ComponentId::Creature))
            continue;
        for (const auto &chunk : part->chunks())
        {
            if (!chunk)
                continue;
            auto *positions = static_cast<lpl::math::Vec3<lpl::math::Fixed32> *>(
                chunk->writeComponent(lpl::ecs::ComponentId::Position));
            auto *creature = static_cast<lpl::core::u32 *>(chunk->writeComponent(lpl::ecs::ComponentId::Creature));
            for (lpl::core::u32 i = 0u; i < chunk->count(); ++i)
            {
                positions[i] = {lpl::math::Fixed32::fromInt(16), lpl::math::Fixed32{}, lpl::math::Fixed32::fromInt(16)};
                creature[i * 2u] = (i == 0u) ? kGrazer : kHunter;
                creature[i * 2u + 1u] = i + 1u;
            }
        }
    }

    constexpr lpl::core::u32 kGrassX = 26u;
    constexpr lpl::core::u32 kPreyX = 6u;
    lpl::ai::StigmergyParams physics;
    physics.evaporation = 0.999f;
    physics.diffusion = 0.25f;
    physics.floor = 0.0001f;

    lpl::engine::systems::ScentDepositSystem deposit{registry, field, params, view};
    lpl::engine::systems::ScentFieldSystem evaporate{field, physics};
    lpl::engine::systems::ScentSteeringSystem steer{registry, field, params, view};

    for (lpl::core::u32 i = 0u; i < 200u; ++i)
    {
        field.deposit(static_cast<lpl::core::u32>(lpl::ai::ScentChannel::Plant), kGrassX, 16u,
                      lpl::math::Fixed32::fromInt(40));
        field.deposit(static_cast<lpl::core::u32>(lpl::ai::ScentChannel::Herbivore), kPreyX, 16u,
                      lpl::math::Fixed32::fromInt(40));
        evaporate.execute(0.016f);
    }

    std::printf("  field at the pair: plant=%d herb=%d | west herb=%d east plant=%d (raw)\n",
                field.value(static_cast<lpl::core::u32>(lpl::ai::ScentChannel::Plant), 16u, 16u).raw(),
                field.value(static_cast<lpl::core::u32>(lpl::ai::ScentChannel::Herbivore), 16u, 16u).raw(),
                field.value(static_cast<lpl::core::u32>(lpl::ai::ScentChannel::Herbivore), 15u, 16u).raw(),
                field.value(static_cast<lpl::core::u32>(lpl::ai::ScentChannel::Plant), 17u, 16u).raw());
    steer.execute(0.016f);
    check(steer.steered() == 2u, "both creatures steered");

    deposit.execute(0.016f);
    check(deposit.deposits() == 2u, "and both marked the field, each on its own channel");

    lpl::math::Fixed32 deerX{};
    lpl::math::Fixed32 wolfX{};
    for (const auto &part : registry.partitions())
    {
        if (!part || !part->archetype().has(lpl::ecs::ComponentId::Creature))
            continue;
        for (const auto &chunk : part->chunks())
        {
            if (!chunk || chunk->count() < 2u)
                continue;
            const auto *v = static_cast<const lpl::math::Vec3<lpl::math::Fixed32> *>(
                chunk->latestComponent(lpl::ecs::ComponentId::Velocity));
            deerX = v[0].x;
            wolfX = v[1].x;
        }
    }
    std::printf("  from one cell: grazer vx=%d raw, hunter vx=%d raw\n", deerX.raw(), wolfX.raw());
    check(deerX.raw() > 0, "the grazer leans EAST, toward the grass");
    check(wolfX.raw() < 0, "the hunter leans WEST, toward the prey");
    check(deerX.raw() != wolfX.raw(), "two creatures on one cell, one field, opposite moves");

    const lpl::core::u32 before = field.fold();
    evaporate.execute(0.016f);
    check(field.fold() != before, "the field forgets, every tick, unconditionally");
}

/**
 * @brief The terrain answers, and the answer is obeyed: terrain refuses, and a forager eats.
 *
 * @details The half of an animal's tick that had no test at all, because it needed a terrain and a terrain was
 *          private state of a 2000-line sample. With engine::ITerrainQuery the test IS the terrain, so both claims
 *          are checkable: a body does not walk into rock, and a forager eats where it stands.
 *
 * One grazer walks due east, straight at the wall, from well before it. A hunter on the same ground must not
 * forage: which species eats is read off its scent declaration, not off a species index.
 */
void checkTheTerrainAnswersAndTheAnswerIsObeyed()
{
    std::printf("\n-- terrain refuses, and a forager eats --\n");
    lpl::ecs::Registry registry;
    lpl::ecology::Herd herd;
    herd.bind(registry);
    WalledMeadow ground;

    lpl::ecology::HerdParams params;
    lpl::ecology::applyDefaultScents(params);

    auto created = registry.createEntity(creatureArchetype());
    check(created.has_value(), "a grazer exists");
    (void) writeCreature(registry, created.value(), 0u, 10, 8);
    herd.add(created.value());

    lpl::engine::systems::GrazingSystem graze{registry, params, ground};
    lpl::engine::systems::LocomotionSystem walk{registry, params, ground};

    for (lpl::core::u32 tick = 0u; tick < 400u; ++tick)
    {
        graze.execute(0.016f);
        walk.execute(0.016f);
    }

    const lpl::math::Vec3<lpl::math::Fixed32> at = positionOf(registry, created.value());
    std::printf("  grazer stopped at x=%.2f (wall at %d), meals=%u\n", at.x.toFloat(), WalledMeadow::kWallX,
                ground.meals);
    check(at.x.toInt() > 10, "it walked");
    check(at.x.toInt() < WalledMeadow::kWallX, "and the terrain refused to let it into the rock");
    check(ground.meals == 400u, "a forager eats on every tick it stands on food");
    check(graze.meals() == 1u, "and the system reports the LAST tick's meals, not a running total");

    auto hunter = registry.createEntity(creatureArchetype());
    check(hunter.has_value(), "a hunter exists");
    (void) writeCreature(registry, hunter.value(), 1u, 12, 8);
    const lpl::core::u32 mealsBefore = ground.meals;
    graze.execute(0.016f);
    check(ground.meals == mealsBefore + 1u, "still exactly one forager, though there are two animals");
}

/**
 * @brief A census counts the registry, and removal destroys the body: the roster owns entities, not copies of them.
 *
 * @details The bug this replaced: the roster entry was popped and the ENTITY stayed. Since the renderer draws what
 *          is in the world rather than what a container remembers, an animal removed from the census kept being
 *          drawn for good.
 */
void checkACensusCountsTheRegistryAndRemovalDestroysTheBody()
{
    std::printf("\n-- the roster owns entities, not copies of them --\n");
    lpl::ecs::Registry registry;
    lpl::ecology::Herd herd;
    herd.bind(registry);
    for (lpl::core::u32 i = 0u; i < 4u; ++i)
    {
        auto body = registry.createEntity(creatureArchetype());
        if (!body.has_value())
            continue;
        (void) writeCreature(registry, body.value(), i < 3u ? 0u : 1u, static_cast<lpl::core::i32>(i), 0);
        herd.add(body.value());
    }
    check(herd.countSpecies(0u) == 3u, "three grazers, counted from the Creature component");
    check(herd.countSpecies(1u) == 1u, "and one hunter");

    const lpl::core::u32 aliveBefore = registry.liveCount();
    check(herd.removeOne(0u), "one grazer can be removed");
    check(herd.countSpecies(0u) == 2u, "the census follows");
    check(registry.liveCount() == aliveBefore - 1u, "and the entity is destroyed, not orphaned");

    herd.clear();
    check(herd.empty() && registry.liveCount() == 0u, "clearing the herd destroys every body");
}

} // namespace

/**
 * @brief Olfaction: a herd steers by what a channel MEANS, and encirclement emerges.
 *
 * Herd had no test at all, which is how it came to read one hard-coded channel
 * uphill for every animal — a deer climbing the deer scent, attracted to itself
 * instead of fleeing the wolf hunting it. Four of the six named channels were
 * never read by anything.
 *
 * The claim worth measuring is the second one. Nothing in the engine says
 * "flank". Hunters are pulled toward the prey's scent and pushed off each other's,
 * and the flanking is what those two terms do when there are several hunters. So
 * the test measures ANGULAR SPREAD around the prey: with kin repulsion the pack
 * distributes around it, without it they stack on one side.
 */
int main()
{
    std::printf("== herd olfaction ==\n\n");

    checkAChannelMeansSomething();

    checkAnEmptyPalateIsNotASilentNoOp();

    checkEncirclementEmerges();

    checkTheSameHuntTwiceEndsIdentically();

    checkCreaturesAreEntitiesAndBehaviourIsSystems();

    checkTheTerrainAnswersAndTheAnswerIsObeyed();

    checkACensusCountsTheRegistryAndRemovalDestroysTheBody();

    std::printf("\n%s (%d failures)\n", failures == 0 ? "ALL PASS" : "FAILURES", failures);
    return failures == 0 ? 0 : 1;
}
