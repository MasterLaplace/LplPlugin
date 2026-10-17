#include <lpl/ai/AbstractWorld.hpp>

#include <lpl/math/Random.hpp>

namespace lpl::ai {

namespace {

constexpr core::u32 kFnv1aOffsetBasis = 0x811C9DC5u;
constexpr core::u32 kFnv1aPrime = 0x01000193u;

} // namespace

RoomId AbstractWorld::addRoom()
{
    _rooms.push_back(AbstractRoom{});
    return static_cast<RoomId>(_rooms.size()) - 1u;
}

void AbstractWorld::connect(RoomId a, RoomId b)
{
    if (a >= _rooms.size() || b >= _rooms.size() || a == b)
        return;
    for (core::u32 i = 0u; i < _rooms[a].exits.size(); ++i)
        if (_rooms[a].exits[i] == b)
            return;
    _rooms[a].exits.push_back(b);
    _rooms[b].exits.push_back(a);
}

core::u32 AbstractWorld::addCreature(core::u32 id, core::u32 species, RoomId room)
{
    AbstractCreature creature;
    creature.id = id;
    creature.species = species;
    creature.room = room;
    _creatures.push_back(creature);
    return static_cast<core::u32>(_creatures.size()) - 1u;
}

void AbstractWorld::moveCreature(core::u32 index, RoomId destination)
{
    if (index >= _creatures.size() || destination >= _rooms.size())
        return;
    _creatures[index].room = destination;

    for (core::u32 r = 0u; r < _rooms.size(); ++r)
        ++_rooms[r].changesSinceVisit;
}

void AbstractWorld::observe(RoomId room, RoomId predicted, core::u32 tick)
{
    _focus = room;
    _predicted = predicted;
    if (room < _rooms.size())
    {
        _rooms[room].lastVisitTick = tick;
        _rooms[room].changesSinceVisit = 0u;
    }
}

core::i32 AbstractWorld::deletionScore(RoomId room, const RealizationBudget &budget, core::u32 tick) const
{
    if (room >= _rooms.size())
        return 0;
    const AbstractRoom &node = _rooms[room];

    core::i32 score = static_cast<core::i32>(tick - node.lastVisitTick);
    score += static_cast<core::i32>(budget.changeWeight * node.changesSinceVisit);

    if (_focus < _rooms.size())
    {
        if (room == _focus)
            return -1000000;

        for (core::u32 i = 0u; i < _rooms[_focus].exits.size(); ++i)
            if (_rooms[_focus].exits[i] == room)
            {
                score -= static_cast<core::i32>(budget.adjacentBonus);
                break;
            }
    }

    if (room == _predicted)
        score -= static_cast<core::i32>(budget.predictedBonus);
    else if (_predicted < _rooms.size() && _focus < _rooms.size())
    {
        score += static_cast<core::i32>(budget.unlikelyPenalty);
    }
    return score;
}

core::u32 AbstractWorld::enforceBudget(const RealizationBudget &budget, core::u32 tick)
{
    core::u32 changes = 0u;

    const auto realise = [&](RoomId room) {
        if (room >= _rooms.size() || _rooms[room].realised)
            return;
        _rooms[room].realised = true;
        ++changes;
    };

    if (_focus < _rooms.size())
    {
        realise(_focus);
        for (core::u32 i = 0u; i < _rooms[_focus].exits.size(); ++i)
            realise(_rooms[_focus].exits[i]);
    }
    if (_predicted < _rooms.size())
        realise(_predicted);

    for (;;)
    {
        core::u32 realised = realisedRoomCount();
        if (realised <= budget.maxRealisedRooms)
            break;

        RoomId worst = kNoRoom;
        core::i32 worstScore = 0;
        for (RoomId r = 0u; r < _rooms.size(); ++r)
        {
            if (!_rooms[r].realised || r == _focus)
                continue;
            const core::i32 score = deletionScore(r, budget, tick);
            if (worst == kNoRoom || score > worstScore)
            {
                worst = r;
                worstScore = score;
            }
        }
        if (worst == kNoRoom)
            break;

        _rooms[worst].realised = false;
        ++changes;
    }

    for (core::u32 i = 0u; i < _creatures.size(); ++i)
    {
        const AbstractCreature &creature = _creatures[i];
        const bool shouldExist = creature.room < _rooms.size() && _rooms[creature.room].realised;
        if (_creatures[i].realised != shouldExist)
            _creatures[i].realised = shouldExist;
    }
    return changes;
}

core::u32 AbstractWorld::tickAbstract(core::u32 tick)
{
    core::u32 moved = 0u;
    for (core::u32 i = 0u; i < _creatures.size(); ++i)
    {
        AbstractCreature &creature = _creatures[i];
        if (creature.room >= _rooms.size())
            continue;

        if (creature.energy > 0u)
            --creature.energy;

        if (_rooms[creature.room].realised)
            continue;

        const PersonalityTraits traits = personalityOf(creature.id, creature.species);
        math::Random random{creature.id ^ (tick * 0x9E3779B1u)};

        const core::u32 restlessness =
            static_cast<core::u32>((traits.energy * math::Fixed32::fromInt(64)).toInt()) + 8u;
        if (random.below(1024u) >= restlessness)
            continue;

        const lpl::pmr::vector<RoomId> &exits = _rooms[creature.room].exits;
        if (exits.empty())
            continue;
        creature.room = exits[random.below(static_cast<core::u32>(exits.size()))];
        ++moved;
    }
    return moved;
}

core::u32 AbstractWorld::realisedRoomCount() const noexcept
{
    core::u32 count = 0u;
    for (core::u32 i = 0u; i < _rooms.size(); ++i)
        count += _rooms[i].realised ? 1u : 0u;
    return count;
}

core::u32 AbstractWorld::realisedCreatureCount() const noexcept
{
    core::u32 count = 0u;
    for (core::u32 i = 0u; i < _creatures.size(); ++i)
        count += _creatures[i].realised ? 1u : 0u;
    return count;
}

PersonalityTraits AbstractWorld::personality(core::u32 index) const
{
    if (index >= _creatures.size())
        return PersonalityTraits{};
    return personalityOf(_creatures[index].id, _creatures[index].species);
}

core::u32 AbstractWorld::fold() const
{
    core::u32 hash = kFnv1aOffsetBasis;
    for (core::u32 i = 0u; i < _creatures.size(); ++i)
    {
        const AbstractCreature &creature = _creatures[i];
        hash = (hash ^ creature.id) * kFnv1aPrime;
        hash = (hash ^ creature.room) * kFnv1aPrime;
        hash = (hash ^ creature.energy) * kFnv1aPrime;
        hash = (hash ^ (creature.realised ? 1u : 0u)) * kFnv1aPrime;
    }
    return hash;
}

} // namespace lpl::ai
