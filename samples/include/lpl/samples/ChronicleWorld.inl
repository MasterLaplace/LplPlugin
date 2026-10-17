#ifndef LPL_SAMPLES_CHRONICLEWORLD_INL
#define LPL_SAMPLES_CHRONICLEWORLD_INL

namespace lpl::samples {

inline void ManiGazetteer::bind(const math::ReliefField &field, core::u32 side) noexcept
{
    _side = side;
    for (core::u32 i = 0u; i < kManiPlaceCount; ++i)
    {
        const ChroniclePlace &place = kManiPlaces[i];
        const core::i32 cellX = field.projection.cellX(math::Fixed32::fromFloat(place.longitude).raw());
        const core::i32 cellZ = field.projection.cellZ(math::Fixed32::fromFloat(place.latitude).raw());
        _cellX[i] = cellX;
        _cellZ[i] = cellZ;
        _worldX[i] = static_cast<core::f32>(cellX) - static_cast<core::f32>(side / 2u);
        _worldZ[i] = static_cast<core::f32>(cellZ) - static_cast<core::f32>(side / 2u);
    }
}

inline bool ManiGazetteer::resolve(core::u32 id, history::Place &out) const
{
    for (core::u32 i = 0u; i < kManiPlaceCount; ++i)
    {
        if (kManiPlaces[i].id != id)
            continue;
        out.id = id;
        out.x = math::Fixed32::fromFloat(_worldX[i]);
        out.z = math::Fixed32::fromFloat(_worldZ[i]);
        out.minYear = kChronicleFirstYear;
        out.maxYear = kChronicleLastYear;
        out.located = true;
        return true;
    }
    return false;
}

inline core::u32 ManiGazetteer::linkedPlaces(core::u32 id, core::u32 *out, core::u32 capacity) const
{
    static constexpr core::u32 kLinks[][2] = {
        {kChroniclePlaceKalamata,  kChroniclePlaceKardamyli},
        {kChroniclePlaceKardamyli, kChroniclePlaceOitylo   },
        {kChroniclePlaceOitylo,    kChroniclePlaceAreopoli },
        {kChroniclePlaceAreopoli,  kChroniclePlaceTainaron },
        {kChroniclePlaceOitylo,    kChroniclePlaceGytheio  },
        {kChroniclePlaceGytheio,   kChroniclePlaceSparta   },
    };
    core::u32 written = 0u;
    for (const auto &link : kLinks)
    {
        if (written >= capacity)
            break;
        if (link[0] == id)
            out[written++] = link[1];
        else if (link[1] == id)
            out[written++] = link[0];
    }
    return written;
}

inline bool ManiGazetteer::mapCell(core::u32 index, core::i32 &outX, core::i32 &outZ) const noexcept
{
    if (index >= kManiPlaceCount)
        return false;
    outX = _cellX[index];
    outZ = _cellZ[index];
    return outX >= 0 && outZ >= 0 && outX < static_cast<core::i32>(_side) && outZ < static_cast<core::i32>(_side);
}

inline core::Expected<void> ChronicleWorld::onInit(engine::WorldContext &context)
{
    _recipe.width = kSide;
    _recipe.depth = kSide;

    const math::ReliefField field = makeRealReliefField(_recipe, _recipe.terrain.amplitude);
    _relief = field;
    _mosaic = math::ReliefMosaic{};
    (void) _mosaic.add(&_relief);

    procgen::ReliefBlend blend{};
    blend.mosaic = &_mosaic;
    blend.detail = _recipe.terrain;
    blend.detail.amplitude = _recipe.terrain.amplitude * 0.25f;

    procgen::WalkabilityRule rule{};
    rule.seaLevel = _recipe.biomes.seaLevel;
    _snapshot = procgen::buildSnapshot(_recipe, nullptr, nullptr, rule, &blend);

    _gazetteer.bind(_relief, kSide);

    engine::systems::TerrainRouteParams routeParams;
    routeParams.cellSize = 1u;
    routeParams.cost.waterLevel = _recipe.biomes.seaLevel;
    routeParams.cost.waterPenalty = 40.0f;
    routeParams.cost.slopePenalty = 6.0f;
    routeParams.coarseRatio = 8u;
    _routes.bind(_snapshot.height, _gazetteer, routeParams);

    core::u32 order[kManiPlaceCount];
    for (core::u32 i = 0u; i < kManiPlaceCount; ++i)
        order[i] = kManiPlaces[i].id;
    _pavedCells = _routes.paveAttested(order, kManiPlaceCount);

    buildCorpus();
    history::FusionReport report;
    history::WorldView view;
    _timeline = history::buildTimeline(_corpus, view, report);
    _era = history::Era::ofYears(kFirstYear, kLastYear, kDaysPerTick);

    auto constraints = pmr::make_unique<history::HistorySystem>(_timeline, _era, _chronicle);
    _constraints = constraints.get();
    ecs::Archetype archetype;
    archetype.add(ecs::ComponentId::Position);
    archetype.add(ecs::ComponentId::Historical);
    _constraints->bindWorld(registry(), _gazetteer, archetype);
    if (auto registered = scheduler().registerSystem(std::move(constraints)); !registered)
        return registered;

    for (core::u32 i = 0u; i < kManiPlaceCount; ++i)
        _placeIds[i] = kManiPlaces[i].id;

    engine::systems::JourneyParams walk;
    walk.pacePerYear = math::Fixed32::fromFloat(6.0f);
    walk.arrivalRadius = math::Fixed32::fromFloat(1.2f);
    walk.horizon = math::Fixed32::fromFloat(200.0f);
    auto journey = pmr::make_unique<engine::systems::JourneySystem>(registry(), _gazetteer, _placeIds, kManiPlaceCount,
                                                                    _era, _chronicle, walk);
    _journey = journey.get();
    _journey->useRoutes(_routes);
    _journey->useTerrain(*this);
    _journey->placeBodyAt(kChronicleSubjectMerchant, kChroniclePlaceKardamyli);
    _journey->placeBodyAt(kChronicleSubjectPilgrim, kChroniclePlaceGytheio);
    if (auto registered = scheduler().registerSystem(std::move(journey)); !registered)
        return registered;

    _hasSurface = context.platform.display().querySurface(_surface) && _surface.buffer != nullptr;
    if (_hasSurface)
    {
        _zoom = _surface.width / kSide;
        const core::u32 verticalZoom = _surface.height / kSide;
        _zoom = _zoom < verticalZoom ? _zoom : verticalZoom;
        if (_zoom == 0u)
            _zoom = 1u;
    }

    char census[96];
    formatLine(census, sizeof(census), "ChronicleWorld: biome grid ", static_cast<core::i32>(_snapshot.biomes.width()),
               "x", static_cast<core::i32>(_snapshot.biomes.depth()), ", distinct biomes ",
               static_cast<core::i32>(distinctBiomes()));
    core::Log::info(census);

    char banner[96];
    formatLine(banner, sizeof(banner), "ChronicleWorld: road cells ", static_cast<core::i32>(_pavedCells), ", pairs ",
               static_cast<core::i32>(_routes.pairs()), ", coarse expansions ",
               static_cast<core::i32>(_routes.coarseExpanded()));
    core::Log::info(banner);
    return {};
}

inline void ChronicleWorld::onFixedStep(core::f32 dt)
{
    tick(dt);
    ++_ticks;
}

inline void ChronicleWorld::onRender(engine::WorldContext &context, core::f64 /*alpha*/)
{
    if (!_hasSurface)
        return;
    drawMap();
    drawRoads();
    drawPlaces();
    drawTravellers();
    drawHud();
    context.platform.display().present();
}

inline bool ChronicleWorld::standable(math::Fixed32 x, math::Fixed32 z) const
{
    core::i32 cellX = 0;
    core::i32 cellZ = 0;
    if (!toCell(x, z, cellX, cellZ))
        return false;
    return _snapshot.blocked.at(static_cast<core::u32>(cellX), static_cast<core::u32>(cellZ)) == 0u;
}

inline bool ChronicleWorld::toCell(math::Fixed32 x, math::Fixed32 z, core::i32 &outX, core::i32 &outZ) noexcept
{
    outX = (x.raw() >> 16) + static_cast<core::i32>(kSide / 2u);
    outZ = (z.raw() >> 16) + static_cast<core::i32>(kSide / 2u);
    return outX >= 0 && outZ >= 0 && outX < static_cast<core::i32>(kSide) && outZ < static_cast<core::i32>(kSide);
}

inline void ChronicleWorld::buildCorpus()
{
    history::SourceProfile register_;
    register_.id = kChronicleSourceRegister;
    register_.kind = history::SourceKind::Notarial;
    register_.yearsAfterEvent = 0u;
    _corpus.sources.push_back(register_);

    history::SourceProfile periplus;
    periplus.id = kChronicleSourcePeriplus;
    periplus.kind = history::SourceKind::Notarial;
    periplus.yearsAfterEvent = 0u;
    _corpus.sources.push_back(periplus);

    addBirth(kChronicleSubjectMerchant, kChroniclePlaceKardamyli, kChronicleSourceRegister);
    addBirth(kChronicleSubjectPilgrim, kChroniclePlaceGytheio, kChronicleSourcePeriplus);

    addJourney(kChronicleSubjectMerchant, kChroniclePlaceTainaron, -480, -460);
    addJourney(kChronicleSubjectPilgrim, kChroniclePlaceKalamata, -470, -440);
}

inline void ChronicleWorld::addBirth(core::u32 subject, core::u32 place, core::u32 source)
{
    history::Fact born;
    born.subject = subject;
    born.predicate = static_cast<core::u32>(history::Predicate::BornAt);
    born.object = place;
    born.fromDay = history::firstDayOfYear(kFirstYear);
    born.toDay = history::lastDayOfYear(kFirstYear);
    born.source = source;
    born.sigma = math::Fixed32::fromFloat(0.97f);
    _corpus.facts.push_back(born);
}

inline void ChronicleWorld::addJourney(core::u32 subject, core::u32 place, core::i32 fromYear, core::i32 toYear)
{
    history::Fact went;
    went.subject = subject;
    went.predicate = static_cast<core::u32>(history::Predicate::TravelledTo);
    went.object = place;
    went.fromDay = history::firstDayOfYear(fromYear);
    went.toDay = history::lastDayOfYear(toYear);
    went.source = kChronicleSourceRegister;
    went.sigma = math::Fixed32::fromFloat(0.3f);
    _corpus.facts.push_back(went);
}

inline core::u32 ChronicleWorld::offSurveyPlaces() const noexcept
{
    core::u32 off = 0u;
    for (core::u32 i = 0u; i < kManiPlaceCount; ++i)
    {
        core::i32 x = 0;
        core::i32 z = 0;
        if (!_gazetteer.mapCell(i, x, z))
            ++off;
    }
    return off;
}

inline core::u32 ChronicleWorld::distinctBiomes() const noexcept
{
    core::u32 seen = 0u;
    for (core::u32 b = 0u; b < static_cast<core::u32>(procgen::BiomeId::Count); ++b)
        if (_snapshot.biomeCounts[b] != 0u)
            ++seen;
    return seen;
}

inline core::u32 ChronicleWorld::pack(procgen::Rgb colour) noexcept
{
    const auto channel = [](float value) {
        const float scaled = value * 255.0f;
        const float clamped = scaled < 0.0f ? 0.0f : scaled > 255.0f ? 255.0f : scaled;
        return static_cast<core::u32>(clamped);
    };
    return (channel(colour.r) << 16) | (channel(colour.g) << 8) | channel(colour.b);
}

inline void ChronicleWorld::put(core::i32 cellX, core::i32 cellZ, procgen::Rgb colour) noexcept
{
    if (cellX < 0 || cellZ < 0 || cellX >= static_cast<core::i32>(kSide) || cellZ >= static_cast<core::i32>(kSide))
        return;
    const core::u32 packed = pack(colour);
    const core::u32 pitch = _surface.pitch / 4u;
    for (core::u32 dz = 0u; dz < _zoom; ++dz)
    {
        const core::u32 y = static_cast<core::u32>(cellZ) * _zoom + dz;
        if (y >= _surface.height)
            return;
        for (core::u32 dx = 0u; dx < _zoom; ++dx)
        {
            const core::u32 x = static_cast<core::u32>(cellX) * _zoom + dx;
            if (x < _surface.width)
                _surface.buffer[y * pitch + x] = packed;
        }
    }
}

inline void ChronicleWorld::drawMap() noexcept
{
    const core::u32 pitch = _surface.pitch / 4u;
    for (core::u32 y = 0u; y < _surface.height; ++y)
        for (core::u32 x = 0u; x < _surface.width; ++x)
            _surface.buffer[y * pitch + x] = 0x00060810u;

    for (core::u32 z = 0u; z < kSide; ++z)
    {
        for (core::u32 x = 0u; x < kSide; ++x)
        {
            procgen::Rgb colour = procgen::biomeColour(_snapshot.biomes.at(x, z));
            const core::f32 slope = procgen::slopeAt(_snapshot.height, x, z).toFloat();
            const core::f32 lit = slope > 2.0f ? 0.62f : slope > 0.8f ? 0.82f : 1.0f;
            colour.r *= lit;
            colour.g *= lit;
            colour.b *= lit;
            put(static_cast<core::i32>(x), static_cast<core::i32>(z), colour);
        }
    }
}

inline void ChronicleWorld::drawRoads() noexcept
{
    const procgen::Rgb road{0.84f, 0.60f, 0.23f};
    for (core::u32 z = 0u; z < kSide; ++z)
        for (core::u32 x = 0u; x < kSide; ++x)
            if (_routes.roads().at(x, z) != 0u)
                put(static_cast<core::i32>(x), static_cast<core::i32>(z), road);
}

inline void ChronicleWorld::drawPlaces() noexcept
{
    const procgen::Rgb town{0.97f, 0.93f, 0.82f};
    for (core::u32 i = 0u; i < kManiPlaceCount; ++i)
    {
        core::i32 cellX = 0;
        core::i32 cellZ = 0;
        if (!_gazetteer.mapCell(i, cellX, cellZ))
            continue;
        for (core::i32 dz = -1; dz <= 1; ++dz)
            for (core::i32 dx = -1; dx <= 1; ++dx)
                put(cellX + dx, cellZ + dz, town);
    }
}

inline void ChronicleWorld::drawTravellers() noexcept
{
    const procgen::Rgb walker{0.93f, 0.33f, 0.24f};
    _bodiesDrawn = 0u;
    for (const auto &partition : registry().partitions())
    {
        if (partition == nullptr || !partition->archetype().has(ecs::ComponentId::Historical))
            continue;
        for (const auto &chunkPtr : partition->chunks())
        {
            if (chunkPtr == nullptr)
                continue;
            const auto *positions =
                static_cast<const math::Vec3<math::Fixed32> *>(chunkPtr->latestComponent(ecs::ComponentId::Position));
            if (positions == nullptr)
                continue;
            for (core::u32 row = 0u; row < chunkPtr->count(); ++row)
            {
                core::i32 cellX = 0;
                core::i32 cellZ = 0;
                if (!toCell(positions[row].x, positions[row].z, cellX, cellZ))
                    continue;
                for (core::i32 d = -1; d <= 1; ++d)
                {
                    put(cellX + d, cellZ, walker);
                    put(cellX, cellZ + d, walker);
                }
                ++_bodiesDrawn;
            }
        }
    }
}

inline void ChronicleWorld::drawHud() noexcept
{
    const core::u32 pitch = _surface.pitch / 4u;
    char line[96];
    core::u32 y = 4u;
    const auto write = [&](const char *text) {
        render::drawShadowedText8x16(_surface.buffer, pitch, 6u, y, text, 0x00F4EFE4u, 0x00101418u);
        y += 17u;
    };

    write("LPLKERNEL CHRONICLE  the Mani, -500 to -400");

    const core::i32 year = kFirstYear + static_cast<core::i32>((_ticks * kDaysPerTick) / 365u);
    formatLine(line, sizeof(line), "year ", year, "   bodies ", static_cast<core::i32>(_bodiesDrawn), "   arrivals ",
               static_cast<core::i32>(_journey != nullptr ? _journey->arrivals() : 0u));
    write(line);

    formatLine(line, sizeof(line), "road cells ", static_cast<core::i32>(_pavedCells), "   pairs ",
               static_cast<core::i32>(_routes.pairs()), "   unreachable ",
               static_cast<core::i32>(_routes.unreachable()));
    write(line);

    formatLine(line, sizeof(line), "coarse ", static_cast<core::i32>(_routes.coarseExpanded()), "   corridor ",
               static_cast<core::i32>(_routes.corridorCells()), "   fine ", static_cast<core::i32>(_routes.expanded()));
    write(line);

    formatLine(line, sizeof(line), "chronicle ", static_cast<core::i32>(_chronicle.size()), "   seeded ",
               static_cast<core::i32>(_constraints != nullptr ? _constraints->seeded() : 0u), "   walkers ",
               static_cast<core::i32>(_journey != nullptr ? _journey->walkers() : 0u));
    write(line);

    const core::u32 off = offSurveyPlaces();
    formatLine(line, sizeof(line), "places ", static_cast<core::i32>(kManiPlaceCount), "  on this ground ",
               static_cast<core::i32>(kManiPlaceCount - off), "  located elsewhere ", static_cast<core::i32>(off));
    write(line);
}

inline void ChronicleWorld::formatLine(char *out, core::u32 capacity, const char *a, core::i32 av, const char *b,
                                       core::i32 bv, const char *c, core::i32 cv) noexcept
{
    core::u32 at = 0u;
    const auto append = [&](const char *text) {
        while (*text != '\0' && at + 1u < capacity)
            out[at++] = *text++;
    };
    const auto appendInt = [&](core::i32 value) {
        char digits[12];
        core::u32 count = 0u;
        const bool negative = value < 0;
        core::u32 magnitude = negative ? static_cast<core::u32>(-value) : static_cast<core::u32>(value);
        do
        {
            digits[count++] = static_cast<char>('0' + (magnitude % 10u));
            magnitude /= 10u;
        } while (magnitude != 0u && count < sizeof(digits));
        if (negative && at + 1u < capacity)
            out[at++] = '-';
        while (count != 0u && at + 1u < capacity)
            out[at++] = digits[--count];
    };
    append(a);
    appendInt(av);
    append(b);
    appendInt(bv);
    append(c);
    appendInt(cv);
    out[at] = '\0';
}

} // namespace lpl::samples

#endif // LPL_SAMPLES_CHRONICLEWORLD_INL
