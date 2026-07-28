/**
 * @file Social.cpp
 * @brief Implementation of asymmetric memory and faction reputation.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/ai/Social.hpp>

namespace lpl::ai {

namespace {

constexpr core::u32 kFnv1aOffsetBasis = 0x811C9DC5u;
constexpr core::u32 kFnv1aPrime = 0x01000193u;

/// Below this confidence a model is worthless and is dropped.
constexpr core::i32 kForgetThreshold = 0x0666; // ~0.025 in Q16.16

} // namespace

void RelationshipTracker::observe(core::u32 observer, core::u32 subject, core::u32 cell, math::Fixed32 vx,
                                  math::Fixed32 vz, Attitude attitude)
{
    for (core::u32 i = 0u; i < _models.size(); ++i)
    {
        if (_models[i].observer != observer || _models[i].opinion.subject != subject)
            continue;
        Opinion &existing = _models[i].opinion;
        existing.lastSeenCell = cell;
        existing.lastVx = vx;
        existing.lastVz = vz;
        existing.confidence = math::Fixed32::one();

        // Seeing something again reinforces the opinion rather than replacing it.
        // Replacing would make a creature's view of another swing on a single
        // glance, which is exactly the memoryless behaviour this exists to avoid.
        if (existing.attitude == attitude)
        {
            existing.intensity =
                existing.intensity + (math::Fixed32::one() - existing.intensity) / math::Fixed32::fromInt(4);
        }
        else
        {
            existing.intensity = existing.intensity - existing.intensity / math::Fixed32::fromInt(3);
            if (existing.intensity < math::Fixed32::fromRaw(0x2000))
                existing.attitude = attitude;
        }
        return;
    }

    Model model;
    model.observer = observer;
    model.opinion.subject = subject;
    model.opinion.attitude = attitude;
    model.opinion.intensity = math::Fixed32::half();
    model.opinion.confidence = math::Fixed32::one();
    model.opinion.lastSeenCell = cell;
    model.opinion.lastVx = vx;
    model.opinion.lastVz = vz;
    _models.push_back(model);
}

core::u32 RelationshipTracker::tick(core::u32 width, core::u32 depth, math::Fixed32 decay)
{
    if (width == 0u || depth == 0u)
        return 0u;

    core::u32 forgotten = 0u;
    core::u32 write = 0u;

    for (core::u32 read = 0u; read < _models.size(); ++read)
    {
        Model model = _models[read];
        Opinion &opinion = model.opinion;

        opinion.confidence = opinion.confidence * decay;
        if (opinion.confidence.raw() < kForgetThreshold)
        {
            ++forgotten;
            continue;
        }

        // Extrapolate the remembered position along the last seen velocity. The
        // creature will search where the target WOULD be if it had not turned —
        // and the fact that it usually did turn is the behaviour, not a defect.
        const core::i32 x = static_cast<core::i32>(opinion.lastSeenCell % width) + opinion.lastVx.toInt();
        const core::i32 z = static_cast<core::i32>(opinion.lastSeenCell / width) + opinion.lastVz.toInt();
        const core::i32 cx = x < 0 ? 0 : (x >= static_cast<core::i32>(width) ? static_cast<core::i32>(width) - 1 : x);
        const core::i32 cz = z < 0 ? 0 : (z >= static_cast<core::i32>(depth) ? static_cast<core::i32>(depth) - 1 : z);
        opinion.lastSeenCell = static_cast<core::u32>(cz) * width + static_cast<core::u32>(cx);

        _models[write++] = model;
    }

    while (_models.size() > write)
        _models.pop_back();
    return forgotten;
}

bool RelationshipTracker::opinion(core::u32 observer, core::u32 subject, Opinion &out) const
{
    for (core::u32 i = 0u; i < _models.size(); ++i)
        if (_models[i].observer == observer && _models[i].opinion.subject == subject)
        {
            out = _models[i].opinion;
            return true;
        }
    return false;
}

void RelationshipTracker::recordAggression(core::u32 faction, core::u32 attacker, math::Fixed32 severity)
{
    for (core::u32 i = 0u; i < _reputations.size(); ++i)
    {
        if (_reputations[i].faction != faction || _reputations[i].subject != attacker)
            continue;
        math::Fixed32 value = _reputations[i].value - severity;
        if (value < -math::Fixed32::one())
            value = -math::Fixed32::one();
        _reputations[i].value = value;
        return;
    }
    _reputations.push_back(Reputation{faction, attacker, -severity});
}

math::Fixed32 RelationshipTracker::reputation(core::u32 faction, core::u32 subject) const
{
    for (core::u32 i = 0u; i < _reputations.size(); ++i)
        if (_reputations[i].faction == faction && _reputations[i].subject == subject)
            return _reputations[i].value;
    return math::Fixed32::zero();
}

Attitude RelationshipTracker::effectiveAttitude(Attitude remembered, const PersonalityTraits &traits,
                                                math::Fixed32 intensity)
{
    // Personality modulates, never overrides. A creature that remembers you as
    // prey does not become afraid of you because it is timid — it becomes
    // reluctant, which is `Antagonises`.
    switch (remembered)
    {
    case Attitude::Antagonises:
        if (traits.aggression > math::Fixed32::fromRaw(0xB333) && intensity > math::Fixed32::half())
            return Attitude::Attacks;
        if (traits.sympathy > math::Fixed32::fromRaw(0xB333))
            return Attitude::Ignores;
        return remembered;

    case Attitude::Eats:
        // A timid predator hesitates rather than charging. `Antagonises` is the
        // honest name for "wants it, will not commit".
        if (traits.bravery < math::Fixed32::fromRaw(0x4CCC))
            return Attitude::Antagonises;
        return remembered;

    case Attitude::Afraid:
        // Bravery does not remove fear, it raises the threshold at which fear
        // wins. A brave creature stands its ground against a weak impression.
        if (traits.bravery > math::Fixed32::fromRaw(0xCCCC) && intensity < math::Fixed32::half())
            return Attitude::Antagonises;
        return remembered;

    default: return remembered;
    }
}

core::u32 RelationshipTracker::fold() const
{
    core::u32 hash = kFnv1aOffsetBasis;
    for (core::u32 i = 0u; i < _models.size(); ++i)
    {
        const Model &model = _models[i];
        hash = (hash ^ model.observer) * kFnv1aPrime;
        hash = (hash ^ model.opinion.subject) * kFnv1aPrime;
        hash = (hash ^ static_cast<core::u32>(model.opinion.attitude)) * kFnv1aPrime;
        hash = (hash ^ static_cast<core::u32>(model.opinion.intensity.raw())) * kFnv1aPrime;
        hash = (hash ^ model.opinion.lastSeenCell) * kFnv1aPrime;
    }
    for (core::u32 i = 0u; i < _reputations.size(); ++i)
    {
        hash = (hash ^ _reputations[i].faction) * kFnv1aPrime;
        hash = (hash ^ _reputations[i].subject) * kFnv1aPrime;
        hash = (hash ^ static_cast<core::u32>(_reputations[i].value.raw())) * kFnv1aPrime;
    }
    return hash;
}

} // namespace lpl::ai
