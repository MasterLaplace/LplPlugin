#ifndef LPL_ENGINE_LIVING_LAYER_INL
#define LPL_ENGINE_LIVING_LAYER_INL

namespace lpl::engine {

inline void LivingLayer::configure(const LivingLayerParams &params, const ecology::LivingRecipe &recipe, core::u32 seed)
{
    _params = params;
    _recipe = recipe;
    _seed = seed;
    _heredity = seed ^ 0xA57E22u;
    _nextId = 1u;
    _grazed = 0u;
    _herd.clear();
    _herdParams.speciesCount = params.speciesCount;
    ecology::applyDefaultScents(_herdParams);
}

inline void LivingLayer::seedWeb(core::u32 standingPlants)
{
    _web = ecology::TrophicWeb{};
    const core::u32 declared =
        _recipe.speciesCount < ecology::kMaxLivingSpecies ? _recipe.speciesCount : ecology::kMaxLivingSpecies;
    for (core::u32 i = 0u; i < declared; ++i)
    {
        ecology::SpeciesParams params = _recipe.species[i].params;
        math::Fixed32 initial = _recipe.species[i].initial;
        if (standingPlants != 0u && params.level == ecology::TrophicLevel::Producer)
        {
            params.capacity = math::Fixed32::fromInt(static_cast<core::i32>(standingPlants + 1u));
            initial = params.capacity;
        }
        (void) _web.add(params, initial, _recipe.species[i].preyIndex);
    }
}

inline core::u32 LivingLayer::webIndexOfBodied(core::u32 bodied) const noexcept
{
    core::u32 seen = 0u;
    for (core::u32 i = 0u; i < _web.species.size(); ++i)
    {
        if (_web.species[i].params.level == ecology::TrophicLevel::Producer)
            continue;
        if (seen == bodied)
            return i;
        ++seen;
    }
    return kNoSpecies;
}

inline core::u32 LivingLayer::rawBodiesFor(core::u32 webIndex) const noexcept
{
    if (webIndex >= _web.species.size())
        return 0u;
    const ecology::Species &species = _web.species[webIndex];
    if (species.population <= species.params.refuge)
        return 0u;

    const core::u32 perBody = _recipe.headPerBody != 0u ? _recipe.headPerBody : 2u;
    core::i32 wanted = species.population.toInt() / static_cast<core::i32>(perBody);
    if (wanted < 1)
        wanted = 1;
    return static_cast<core::u32>(wanted);
}

inline core::u32 LivingLayer::bodiesFor(core::u32 species) const noexcept
{
    if (species >= _params.speciesCount)
        return 0u;
    const core::u32 index = webIndexOfBodied(species);
    if (index == kNoSpecies)
        return 0u;

    const core::u32 mine = rawBodiesFor(index);
    core::u32 total = 0u;
    for (core::u32 i = 0u; i < _params.speciesCount; ++i)
    {
        const core::u32 other = webIndexOfBodied(i);
        if (other != kNoSpecies)
            total += rawBodiesFor(other);
    }
    if (total <= _params.maxBodies || total == 0u)
        return mine;

    const core::u32 scaled = (mine * _params.maxBodies) / total;
    return scaled == 0u && mine != 0u ? 1u : scaled;
}

inline systems::CreatureFieldView LivingLayer::fieldView() const noexcept
{
    return systems::CreatureFieldView{_scent.originX(), _scent.originZ(), _scent.field().width(),
                                      _scent.field().depth()};
}

template <typename PlaceAt> inline bool LivingLayer::spawn(math::Random &random, core::u32 species, PlaceAt &&placeAt)
{
    if (_herd.size() >= _params.maxBodies || _herd.registry() == nullptr)
        return false;

    const core::u32 identity = _nextId++;

    ecology::Genome stock{};
    if (species == 1u)
    {
        stock.size = math::Fixed32::fromFloat(1.4f);
        stock.maxSpeed = math::Fixed32::fromFloat(5.0f);
    }
    else
    {
        stock.size = math::Fixed32::fromFloat(0.9f);
        stock.maxSpeed = math::Fixed32::fromFloat(3.5f);
    }
    const ecology::Genome genome = ecology::mutate(stock, 8u, 0.18f, _heredity);

    for (core::u32 attempt = 0u; attempt < 24u; ++attempt)
    {
        math::Fixed32 x{};
        math::Fixed32 z{};
        if (!placeAt(random, attempt, x, z))
            continue;
        ecs::Archetype archetype;
        archetype.add(ecs::ComponentId::Position);
        archetype.add(ecs::ComponentId::Velocity);
        archetype.add(ecs::ComponentId::AABB);
        archetype.add(ecs::ComponentId::Genome);
        archetype.add(ecs::ComponentId::Creature);
        archetype.add(ecs::ComponentId::Heading);

        ecs::Registry &registry = *_herd.registry();
        auto created = registry.createEntity(archetype);
        if (!created.has_value())
            return false;
        const ecs::EntityId entity = created.value();

        const math::Fixed32 vx = random.unit() - math::Fixed32::half();
        const math::Fixed32 vz = random.unit() - math::Fixed32::half();

        if (!writeBody(registry, entity, species, identity, genome, x, z, vx, vz))
        {
            (void) registry.destroyEntity(entity);
            return false;
        }
        _herd.add(entity);
        return true;
    }
    return false;
}

template <typename PlaceAt> inline void LivingLayer::reconcile(core::u32 tick, PlaceAt &&placeAt)
{
    math::Random stock{_seed ^ (0xB0D533u + tick)};
    _herd.reconcile(
        _params.speciesCount, [this](core::u32 species) { return bodiesFor(species); },
        [this, &stock, &placeAt](core::u32 species) { return spawn(stock, species, placeAt); });
}

inline bool LivingLayer::writeBody(ecs::Registry &registry, ecs::EntityId entity, core::u32 species, core::u32 identity,
                                   const ecology::Genome &genome, math::Fixed32 x, math::Fixed32 z, math::Fixed32 vx,
                                   math::Fixed32 vz)
{
    auto ref = registry.resolve(entity);
    if (!ref.has_value())
        return false;
    for (const auto &part : registry.partitions())
    {
        if (!part || !part->archetype().has(ecs::ComponentId::Creature))
            continue;
        const auto &chunks = part->chunks();
        if (ref.value().chunkIndex >= static_cast<core::u32>(chunks.size()) || !chunks[ref.value().chunkIndex])
            continue;
        auto &chunk = *chunks[ref.value().chunkIndex];
        const auto entities = chunk.entities();
        if (ref.value().localIndex >= entities.size() || entities[ref.value().localIndex] != entity)
            continue;
        const core::u32 slot = ref.value().localIndex;

        if (auto *p = static_cast<math::Vec3<math::Fixed32> *>(chunk.writeComponent(ecs::ComponentId::Position)))
            p[slot] = {x, math::Fixed32{}, z};
        if (auto *v = static_cast<math::Vec3<math::Fixed32> *>(chunk.writeComponent(ecs::ComponentId::Velocity)))
            v[slot] = {vx, math::Fixed32{}, vz};
        if (auto *b = static_cast<math::Vec3<math::Fixed32> *>(chunk.writeComponent(ecs::ComponentId::AABB)))
            b[slot] = {genome.size * math::Fixed32::half(), genome.size, genome.size * math::Fixed32::half()};
        if (auto *g = static_cast<ecology::Genome *>(chunk.writeComponent(ecs::ComponentId::Genome)))
            g[slot] = genome;
        if (auto *c = static_cast<core::u32 *>(chunk.writeComponent(ecs::ComponentId::Creature)))
        {
            c[slot * 2u] = species;
            c[slot * 2u + 1u] = identity;
        }
        if (auto *h = static_cast<math::Fixed32 *>(chunk.writeComponent(ecs::ComponentId::Heading)))
        {
            h[slot * 2u] = vx;
            h[slot * 2u + 1u] = vz;
        }
        return true;
    }
    return false;
}

inline void LivingLayer::setProducerPopulation(core::u32 standing) noexcept
{
    if (!_web.species.empty() && standing != 0u)
        _web.species[0].population = math::Fixed32::fromInt(static_cast<core::i32>(standing));
}

} // namespace lpl::engine

#endif // LPL_ENGINE_LIVING_LAYER_INL
