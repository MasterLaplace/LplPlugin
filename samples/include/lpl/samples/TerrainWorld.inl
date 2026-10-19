#ifndef LPL_SAMPLES_TERRAINWORLD_INL
#define LPL_SAMPLES_TERRAINWORLD_INL

namespace lpl::samples {

inline math::ReliefField makeRealReliefField(const procgen::WorldRecipe &recipe, core::f32 noiseAmplitude) noexcept
{
    math::GeoProjection spec{};
    spec.originLatitudeRaw = (kReliefBlobSouthLatitude + 1) * 65536;
    spec.originLongitudeRaw = kReliefBlobWestLongitude * 65536;
    spec.referenceLatitude = kReliefBlobSouthLatitude;
    spec.metresPerCell = kReliefBlobMetresPerCell;

    const core::f32 span = static_cast<core::f32>(kReliefBlobHighest - kReliefBlobLowest);
    const core::f32 wanted = noiseAmplitude * 2.0f;
    spec.unitsPerMetre = math::Fixed32::fromFloat(span > 1.0f ? wanted / span : 1.0f);
    spec.seaLevelUnits = math::Fixed32::fromFloat(recipe.biomes.seaLevel);

    math::ReliefField field{};
    field.samples = kReliefBlobSamples;
    field.width = kReliefBlobSide;
    field.height = kReliefBlobSide;
    field.originCellX = 0;
    field.originCellZ = 0;
    field.projection = math::makeReliefProjection(spec);
    field.blendCells = kReliefBlobSide / 6u;
    field.exposedEdges = 0xFu;
    field.level = 0u;
    return field;
}

inline core::Expected<void> TerrainWorld::onInit(engine::WorldContext &context)
{
    if (_recipe.width > kMaxSize || _recipe.depth > kMaxSize)
        return core::makeError(core::ErrorCode::InvalidArgument, "TerrainWorld: the grid is wider than kMaxSize");

    _lodRings = context.config.lodRings();
    _viewDistance = context.config.viewDistance() > 0.0f ? context.config.viewDistance() : kPlantFarDistance;
    _terrainShadows = context.config.enableTerrainShadows();
    _shadowChunksPerTick = context.config.shadowChunksPerTick();
    _perPixelSurface = context.config.enablePerPixelSurface();
    _pbrSurface = context.config.enablePbrSurface();
    _reflectionProbeOn = context.config.enableWaterReflection();
    _skyBlock = context.config.skyBlockSize() == 0u ? 1u : context.config.skyBlockSize();

    _terrainSurface.configure(context.config, _view.surface, _seed);
    _terrainSurface.applyLook(_view.sky, _view.water, _view.dayFraction);
    _terrainSurface.attachProbe(_probeColor, _probeDepth, kProbeWidth, kProbeHeight);
    _maxResident = context.config.maxResidentChunks() == 0u ? kMaxResidentCeiling : context.config.maxResidentChunks();
    _caveDrawRadius = context.config.caveDrawRadius();

    _platform = &context.platform;
    _renderer.setClock(&context.platform.clock());
    _hasPointer = context.platform.input().hasPointer();
    core::Log::info(_hasPointer ? "TerrainWorld: pointing device present — mouse look enabled" :
                                  "TerrainWorld: no pointing device — look with I/K and the pointer keys");

    if (!context.platform.display().querySurface(_surface) || _surface.buffer == nullptr ||
        _surface.bitsPerPixel != 32u || _surface.width < 64u || _surface.height < 64u)
    {
        _hasSurface = false;
        core::Log::info("TerrainWorld: no usable 32bpp surface, running headless");
    }
    else
    {
        _hasSurface = true;
        core::Log::info("TerrainWorld: WASD=walk SPACE=jump mouse or J/L=turn I/K=tilt O=map V=detach X=exit");
    }

    _living.bind(registry());

    generate();

    if (context.config.enableStreaming())
        setInfinite(true);
    if (_hasSurface && _infinite)
    {
        _camera.setFirstPerson(true);
        _camera.setPitch(kFirstPersonPitch);
    }

    _creatures.build(registry(), _living, *this);
    if (auto registered = _creatures.registerOn(scheduler()); !registered)
        core::Log::warn("TerrainWorld: a creature system was refused; the living layer will be incomplete");
    return {};
}

inline bool TerrainWorld::standable(math::Fixed32 x, math::Fixed32 z) const
{
    if (_infinite)
    {
        const core::f32 height = procgen::sampleWorldHeight(_streamer.chunkParams(), x.toInt(), z.toInt()).toFloat();
        return height > seaLevel() + 0.2f;
    }
    core::u32 cx = 0u;
    core::u32 cz = 0u;
    if (!worldToCell(x, z, cx, cz))
        return false;
    return _blocked.at(cx, cz) == 0u;
}

inline bool TerrainWorld::recoveryDirection(math::Fixed32 x, math::Fixed32 z, math::Fixed32 &outX,
                                            math::Fixed32 &outZ) const
{
    const math::Fixed32 targetX = _infinite ? math::Fixed32::fromFloat(_camera.focusX()) : math::Fixed32{};
    const math::Fixed32 targetZ = _infinite ? math::Fixed32::fromFloat(_camera.focusZ()) : math::Fixed32{};
    outX = x.raw() > targetX.raw() ? -math::Fixed32::one() : math::Fixed32::one();
    outZ = z.raw() > targetZ.raw() ? -math::Fixed32::one() : math::Fixed32::one();
    return true;
}

inline bool TerrainWorld::consumePlantAt(core::i32 worldX, core::i32 worldZ)
{
    const bool ate = _infinite ? grazeEndless(worldX, worldZ) :
                                 grazeBounded(math::Fixed32::fromInt(worldX), math::Fixed32::fromInt(worldZ));
    if (ate)
        _living.countGrazed();
    return ate;
}

inline void TerrainWorld::onFixedStep(core::f32 dt)
{
    const core::u64 stepBegan = timestamp();
    ++_windowTicks;
    _terrainSurface.advance(kDayFractionPerSecond * dt, kRipplePhasePerSecond * dt);

    if (_infinite)
    {
        stepBody(dt);
        recentreField();
        streamChunks();
        if (_terrainShadows && !_streamer.empty())
            for (core::u32 refreshed = 0u; refreshed < _shadowChunksPerTick; ++refreshed)
            {
                if (ResidentChunk *chunk = _streamer.nextShadowChunk(); chunk != nullptr)
                    refreshShadows(*chunk);
            }
    }

    refreshCreatureView();
    engine::World::onFixedStep(dt);
    ++_ticks;
    tickVegetation();

    if (_ticks % kWebPeriod == 0u)
    {
        _living.web().step(1u);
        reconcile();
    }
    _simCycles += timestamp() - stepBegan;
}

inline void TerrainWorld::onRender(engine::WorldContext &context, core::f64 /*alpha*/)
{
    drainInput(context);
    if (!_hasSurface)
        return;

    render::RenderTarget target{_color, _depth, kRenderWidth, kRenderHeight};

    const core::u64 sceneBegan = timestamp();
    renderScene(target);
    const core::u64 blitBegan = timestamp();
    render::blitScaled(_surface.buffer, _surface.pitch / 4u, _surface.width, _surface.height, _color, kRenderWidth,
                       kRenderHeight);
    const core::u64 hudBegan = timestamp();
    drawHud();
    const core::u64 presentBegan = timestamp();
    context.platform.display().present();
    const core::u64 frameEnded = timestamp();

    _sceneCycles += blitBegan - sceneBegan;
    _blitCycles += hudBegan - blitBegan;
    _hudCycles += presentBegan - hudBegan;
    _presentCycles += frameEnded - presentBegan;
    ++_frames;
    ++_windowFrames;
    refreshProfile();
}

inline void TerrainWorld::refreshProfile() noexcept
{
    if (++_profileWindow < 24u)
        return;
    _profileWindow = 0u;

    const core::u64 total = _simCycles + _sceneCycles + _blitCycles + _hudCycles + _presentCycles;
    if (total != 0u)
    {
        _simShare = static_cast<core::u32>((_simCycles * 100u) / total);
        _sceneShare = static_cast<core::u32>((_sceneCycles * 100u) / total);
        _skyShare = static_cast<core::u32>((_renderer.skyCycles() * 100u) / total);
        _groundShare = static_cast<core::u32>((_renderer.groundCycles() * 100u) / total);
        _waterShare = static_cast<core::u32>((_renderer.waterCycles() * 100u) / total);
        _propShare = static_cast<core::u32>((_renderer.propCycles() * 100u) / total);
        _blitShare = static_cast<core::u32>((_blitCycles * 100u) / total);
        _presentShare = static_cast<core::u32>((_presentCycles * 100u) / total);
    }
    _simCycles = 0u;
    _sceneCycles = 0u;
    _blitCycles = 0u;
    _hudCycles = 0u;
    _presentCycles = 0u;
    _renderer.resetPhaseCounters();
}

inline void TerrainWorld::resetRateWindow() noexcept
{
    _windowFrames = 0u;
    _windowTicks = 0u;
}

inline core::i32 TerrainWorld::spawnSearchCells() const noexcept
{
    const procgen::NoiseParams &noise = _streamer.chunkParams().noise;
    if (noise.frequency <= 0.0f)
        return 64;
    core::f32 wavelength = 1.0f / noise.frequency;
    for (core::u32 octave = 1u; octave < noise.octaves; ++octave)
        wavelength *= noise.lacunarity > 1.0f ? noise.lacunarity : 2.0f;
    const core::i32 cells = static_cast<core::i32>(wavelength);
    return cells < 64 ? 64 : (cells > 512 ? 512 : cells);
}

inline void TerrainWorld::generate()
{
    for (core::usize i = 0u; i < _propIds.size(); ++i)
        (void) registry().destroyEntity(_propIds[i]);
    _propIds.clear();

    {
        procgen::WorldRecipe recipe = _recipe;
        recipe.seed = _seed;
        recipe.terrain.seed = _seed;
        recipe.caves.seed = _seed ^ 0xCA4Eu;
        recipe.materializeGround = false;

        attachRealRelief(recipe.terrain.amplitude);
        procgen::ReliefBlend blend{};
        blend.mosaic = _reliefMosaic.valid() ? &_reliefMosaic : nullptr;
        blend.detail = recipe.terrain;
        blend.detail.amplitude = recipe.terrain.amplitude * 0.12f;
        blend.detail.frequency = recipe.terrain.frequency * 3.0f;
        blend.detail.octaves = 3u;
        blend.seed = _seed;

        const procgen::WorldSnapshot snapshot = procgen::buildSnapshot(
            recipe, &registry(), &_propIds, procgen::WalkabilityRule{recipe.biomes.seaLevel, 2.4f}, &blend);

        _height = snapshot.height;
        _biomes = snapshot.biomes;
        _moisture = snapshot.moisture;
        _rivers = snapshot.rivers;
        _settlement = snapshot.settlement;
        _roads = snapshot.roads;
        _blocked = snapshot.blocked;
        _gridWidth = snapshot.width;
        _gridDepth = snapshot.depth;
        _low = snapshot.lowest;
        _high = snapshot.highest;
        for (core::u32 i = 0u; i < static_cast<core::u32>(procgen::BiomeId::Count); ++i)
            _biomeCounts[i] = snapshot.biomeCounts[i];

        _propCount = snapshot.stats.propEntities;
        _plotCount = snapshot.stats.settlementPlots;
        _roadCells = snapshot.stats.roadCells;
        _riverCells = snapshot.stats.riverCells;
        _caveFloor = snapshot.stats.dungeonFloor;
        _gatePassed = snapshot.gatePassed;
        _boundedHasSea = _low.toFloat() < seaLevel();

        _plants.clear();
        procgen::scatterVegetation(
            snapshot, _seed ^ 0x5EED11u, 2u,
            [](procgen::BiomeId biome) {
                return biome == procgen::BiomeId::Forest || biome == procgen::BiomeId::Taiga ||
                       biome == procgen::BiomeId::Rainforest;
            },
            [this](core::i32 cellX, core::i32 cellZ) {
                Plant plant;
                plant.cellX = cellX;
                plant.cellZ = cellZ;
                _plants.push_back(plant);
            });
    }

    _living.configure(livingParams(), _livingRecipe, _seed);
    _living.seedWeb(static_cast<core::u32>(_plants.size()));
    _living.openScent(_gridWidth < _gridDepth ? _gridWidth : _gridDepth, _living.params().scentLayers);
    _living.scent().centreOn(0, 0);
    _living.scent().field().setObstacles(_blocked);
    seedHerd();

    _ticks = 0u;

    engine::PropLibraryParams propParams;
    propParams.treeSpecies = kTreeSpecies;
    propParams.rockVariants = kRockVariants;
    propParams.viewDistance = _viewDistance;
    propParams.nearDistance = kPlantNearDistance;
    propParams.canopyDistance = kPlantCanopyDistance;
    propParams.propDistance = kPropDistance;
    propParams.fogDensity = kFogDensity;
    propParams.rockAlbedo = kRockAlbedo;
    propParams.plantSalt = kPlantSalt;
    propParams.rockSalt = kRockSalt;
    _props.build(propParams, _seed);

    core::Log::info("TerrainWorld: world generated");
}

inline void TerrainWorld::attachRealRelief(core::f32 amplitude)
{
    _reliefField = makeRealReliefField(_recipe, amplitude);
    _reliefMosaic = math::ReliefMosaic{};
    (void) _reliefMosaic.add(&_reliefField);
}

inline void TerrainWorld::setInfinite(bool infinite)
{
    if (_infinite == infinite)
        return;
    _infinite = infinite;
    _streamer.clear();

    if (!_infinite)
    {
        generate();
        return;
    }

    procgen::WorldRecipe seeded = _recipe;
    seeded.seed = _seed;
    seeded.terrain.seed = _seed;
    const procgen::EndlessPlan plan = procgen::endlessPlanFromRecipe(seeded, kChunkSize);
    _chunkParams = plan.chunk;

    attachRealRelief(_chunkParams.noise.amplitude);
    _chunkParams.relief = _reliefMosaic.valid() ? &_reliefMosaic : nullptr;

    _chunkParams.reliefDetail = _chunkParams.noise;
    _chunkParams.reliefDetail.amplitude = _chunkParams.noise.amplitude * 0.12f;
    _chunkParams.reliefDetail.frequency = _chunkParams.noise.frequency * 3.0f;
    _chunkParams.reliefDetail.octaves = 3u;

    _chunkParams.globe = _reliefField.projection.globe();

    core::Log::info(_chunkParams.relief != nullptr ?
                        "TerrainWorld: standing on measured earth (SRTM, southern Peloponnese)" :
                        "TerrainWorld: no survey attached, ground is invented");

    _streamParams.generateRadius = kStreamRadius;
    _streamParams.releaseRatio16 = 24u;
    _streamParams.directionWeight16 = 12u;
    _streamParams.maxGeneratePerTick = 1u;
    _streamParams.maxReleasePerTick = 4u;

    _camera.setFocus(0.0f, 0.0f);
    _bodySpawned = false;
    _camera.setEyeHeight(_bodyParams.eyeHeight.toFloat());

    _living.configure(livingParams(), _livingRecipe, _seed);
    _living.seedWeb(0u);
    _living.openScent(kFieldSpan, _living.params().scentLayers);
    _living.scent().centreOn(0, 0);

    _riverParams = plan.rivers;
    _riverSurfaceRise = plan.riverSurfaceRise;
    _caveMouthDrop = plan.rule.caveMouthDrop;

    const core::u32 wind = seeded.climate.windDirection & 3u;
    _view.water.setDrift(static_cast<core::f32>(procgen::kNeighbor4X[wind]),
                         static_cast<core::f32>(procgen::kNeighbor4Z[wind]));

    _view.surface.seaLevel = plan.rule.seaLevel;

    _streamer.configure(_chunkParams, _riverParams, _streamParams, _maxResident, plan.rule);

    seedHerd();

    core::Log::info("TerrainWorld: endless mode — WASD walks, the world streams around you");
}

inline void TerrainWorld::recentreField()
{
    (void) _living.scent().follow(static_cast<core::i32>(_camera.focusX()), static_cast<core::i32>(_camera.focusZ()));
}

inline void TerrainWorld::stepBody(core::f32 dt)
{
    if (!_embodied)
        return;
    ensureBodySpawned();

    engine::CharacterIntent intent = _intent;
    intent.turn = _pendingTurn;
    intent.jump = _jumpPressed;
    _pendingTurn = math::Fixed32{};
    _jumpPressed = false;

    _body.step(_bodyParams, intent, math::Fixed32::fromFloat(dt),
               [this](core::i32 x, core::i32 z, math::Fixed32 y) { return _streamer.spanAt(x, z, y); });

    _camera.setFocus(_body.x().toFloat(), _body.z().toFloat());
    _camera.setYaw(_body.yaw().toFloat());
}

inline void TerrainWorld::refreshShadows(ResidentChunk &chunk) const
{
    render::HeightfieldPatchParams patch;
    patch.size = kChunkSize;
    patch.stride = 1u;
    patch.originX = chunk.coord.x * static_cast<core::i32>(kChunkSize);
    patch.originZ = chunk.coord.z * static_cast<core::i32>(kChunkSize);

    const procgen::ChunkParams &params = _streamer.chunkParams();
    _terrainSurface.fillShadowMask(
        patch, [&params](core::i32 x, core::i32 z) { return procgen::sampleWorldHeight(params, x, z).toFloat(); },
        [this, &chunk](auto &&emit) {
            for (core::u32 i = 0u; i < chunk.plants.size(); ++i)
            {
                if (!chunk.plants[i].standing)
                    continue;
                core::f32 height = 0.0f;
                core::f32 spread = 0.0f;
                _props.plantExtent(chunk.plants[i].cellX, chunk.plants[i].cellZ, height, spread);
                emit(chunk.plants[i].cellX, chunk.plants[i].cellZ, height, spread);
            }
        },
        chunk.shade);
}

inline engine::LivingLayerParams TerrainWorld::livingParams() const noexcept
{
    engine::LivingLayerParams params;
    params.maxBodies = kMaxCreatures;
    params.speciesCount = 2u;
    params.scentSpan = kFieldSpan;
    params.webPeriod = kWebPeriod;
    return params;
}

inline void TerrainWorld::seedHerd()
{
    math::Random stock{_seed ^ 0x57EA11u};
    for (core::u32 species = 0u; species + 1u < _living.web().species.size() && species < 2u; ++species)
    {
        const core::u32 wanted = _living.bodiesFor(species);
        for (core::u32 i = 0u; i < wanted; ++i)
            (void) _living.spawn(stock, species,
                                 [this](math::Random &r, core::u32 a, math::Fixed32 &x, math::Fixed32 &z) {
                                     return proposeSpawn(r, a, x, z);
                                 });
    }
}

inline bool TerrainWorld::proposeSpawn(math::Random &random, core::u32 attempt, math::Fixed32 &outX,
                                       math::Fixed32 &outZ) const
{
    (void) attempt;
    if (_infinite)
    {
        const core::f32 offsetX = static_cast<core::f32>(random.range(-20, 20));
        const core::f32 offsetZ = static_cast<core::f32>(random.range(-20, 20));
        outX = math::Fixed32::fromFloat(_camera.focusX() + offsetX);
        outZ = math::Fixed32::fromFloat(_camera.focusZ() + offsetZ);
        return standable(outX, outZ);
    }
    const core::u32 x = random.below(_gridWidth);
    const core::u32 z = random.below(_gridDepth);
    if (_blocked.at(x, z) != 0u)
        return false;
    outX = cellToWorldX(x);
    outZ = cellToWorldZ(z);
    return true;
}

inline void TerrainWorld::streamChunks()
{
    _streamer.update(_camera.focusX(), _camera.focusZ(), -render::OrbitCamera::sinOf(_camera.yaw()),
                     -render::OrbitCamera::cosOf(_camera.yaw()), [this](ResidentChunk &chunk) {
                         if (_terrainShadows)
                             refreshShadows(chunk);
                     });
}

inline bool TerrainWorld::grazeEndless(core::i32 worldX, core::i32 worldZ)
{
    for (core::u32 c = 0u; c < _streamer.size(); ++c)
    {
        ResidentChunk &chunk = _streamer.at(c);
        const core::i32 originX = chunk.coord.x * static_cast<core::i32>(kChunkSize);
        const core::i32 originZ = chunk.coord.z * static_cast<core::i32>(kChunkSize);
        if (worldX < originX || worldZ < originZ || worldX >= originX + static_cast<core::i32>(kChunkSize) ||
            worldZ >= originZ + static_cast<core::i32>(kChunkSize))
            continue;
        if (chunk.plants.empty())
            return false;
        return ecology::grazeAt(&chunk.plants[0], static_cast<core::u32>(chunk.plants.size()), worldX, worldZ, 1,
                                _living.recipe().regrowthTicks);
    }
    return false;
}

inline bool TerrainWorld::grazeBounded(math::Fixed32 x, math::Fixed32 z)
{
    if (_plants.empty())
        return false;
    core::u32 cx = 0u;
    core::u32 cz = 0u;
    if (!worldToCell(x, z, cx, cz))
        return false;
    return ecology::grazeAt(&_plants[0], static_cast<core::u32>(_plants.size()), static_cast<core::i32>(cx),
                            static_cast<core::i32>(cz), 1, _living.recipe().regrowthTicks);
}

inline void TerrainWorld::tickVegetation()
{
    core::u32 standing = 0u;
    if (_infinite)
    {
        for (core::u32 c = 0u; c < _streamer.size(); ++c)
        {
            ResidentChunk &chunk = _streamer.at(c);
            if (!chunk.plants.empty())
                standing += ecology::tickPlants(&chunk.plants[0], static_cast<core::u32>(chunk.plants.size()));
        }
    }
    else if (!_plants.empty())
    {
        standing = ecology::tickPlants(&_plants[0], static_cast<core::u32>(_plants.size()));
    }
    _living.setProducerPopulation(standing);
}

inline void TerrainWorld::reconcile()
{
    _living.reconcile(_ticks, [this](math::Random &r, core::u32 a, math::Fixed32 &x, math::Fixed32 &z) {
        return proposeSpawn(r, a, x, z);
    });
}

inline core::f32 TerrainWorld::bedHeightAt(core::f32 worldX, core::f32 worldZ) const noexcept
{
    if (_infinite)
        return groundAt(static_cast<core::i32>(worldX), static_cast<core::i32>(worldZ));
    core::u32 cx = 0u;
    core::u32 cz = 0u;
    if (_height.empty() || !worldToCell(math::Fixed32::fromFloat(worldX), math::Fixed32::fromFloat(worldZ), cx, cz))
        return kFloor;
    return _height.at(cx, cz).toFloat();
}

inline void TerrainWorld::renderScene(const render::RenderTarget &rt) const noexcept
{
    render::clearTarget(rt, 0x000A0E18u);

    engine::TerrainDrawParams params;
    params.chunkSize = kChunkSize;
    params.lodRings = _lodRings;
    params.seaLevel = _view.surface.seaLevel;
    params.ambient = _view.surface.ambient;
    if (_infinite && _embodied)
    {
        params.useFocusHeight = true;
        params.focusHeight = _body.y().toFloat();
    }
    params.grazerTint = _view.grazerTint;
    params.hunterTint = _view.hunterTint;
    params.bodyScale = _view.bodyScale;
    params.skirtDrop = kSkirtDrop;
    params.chunkCentreY = kChunkCentreY;
    params.chunkHalfHeight = kChunkHalfHeight;
    params.riverSurfaceRise = _riverSurfaceRise;
    params.boundedHasSea = _boundedHasSea;
    params.caveMouthDrop = _caveMouthDrop;
    params.caveDrawRadius = _caveDrawRadius;

    const auto palette = [this](procgen::BiomeId biome) { return biomeColour(biome); };

    if (_infinite)
    {
        _lastTriangles =
            _renderer.drawStreamed(rt, _camera, _streamer, _terrainSurface, _props, registry(), params, _frames,
                                   palette, [this](core::i32 x, core::i32 z) { return groundAt(x, z); });
        return;
    }

    if (_height.empty())
        return;
    _lastTriangles = _renderer.drawBounded(
        rt, _camera, _terrainSurface, _props, registry(), _gridWidth, _gridDepth,
        _plants.empty() ? nullptr : &_plants[0], static_cast<core::u32>(_plants.size()), params, palette,
        [this](core::u32 x, core::u32 z) { return _height.at(x, z).toFloat(); },
        [this](core::u32 x, core::u32 z) { return cellColour(x, z); },
        [this](core::i32 x, core::i32 z) { return bedHeightAt(static_cast<core::f32>(x), static_cast<core::f32>(z)); });
}

inline core::u32 TerrainWorld::builtInBiomeColour(procgen::BiomeId biome) noexcept
{
    switch (biome)
    {
    case procgen::BiomeId::Ocean: return 0x00123A6Au;
    case procgen::BiomeId::Lake: return 0x00295A8Cu;
    case procgen::BiomeId::Beach: return 0x00D4C48Cu;
    case procgen::BiomeId::Snow: return 0x00F0F2F6u;
    case procgen::BiomeId::Tundra: return 0x0099A08Fu;
    case procgen::BiomeId::Taiga: return 0x00335F50u;
    case procgen::BiomeId::Rock: return 0x00706A66u;
    case procgen::BiomeId::Desert: return 0x00D9B86Bu;
    case procgen::BiomeId::Savanna: return 0x00B3AD57u;
    case procgen::BiomeId::Grassland: return 0x006B994Du;
    case procgen::BiomeId::Forest: return 0x00337238u;
    case procgen::BiomeId::Rainforest: return 0x001C5C2Du;
    case procgen::BiomeId::Marsh: return 0x004F6B54u;
    case procgen::BiomeId::Count: break;
    }
    return 0x00FF00FFu;
}

inline core::u32 TerrainWorld::cellColour(core::u32 x, core::u32 z) const noexcept
{
    if (_shading == Shading::Height)
    {
        const core::f32 span = (_high - _low).toFloat();
        const core::f32 t = span > 0.0f ? (_height.at(x, z) - _low).toFloat() / span : 0.5f;
        return render::heatRamp(t);
    }
    if (_shading == Shading::Moisture)
        return _moisture.empty() ? 0x00404040u : render::heatRamp(_moisture.at(x, z).toFloat());

    if (!_rivers.empty() && _rivers.at(x, z) != 0u)
        return 0x002A6BB8u;
    if (!_settlement.empty())
    {
        switch (_settlement.at(x, z))
        {
        case procgen::SettlementCell::Road: return 0x00595049u;
        case procgen::SettlementCell::Plaza: return 0x008C7F70u;
        case procgen::SettlementCell::Plot: return 0x00D18C38u;
        default: break;
        }
    }
    if (!_roads.empty() && _roads.at(x, z) != 0u)
        return 0x003D3833u;
    return biomeColour(_biomes.at(x, z));
}

inline void TerrainWorld::drawHud() const noexcept
{
    const core::u32 pitchPixels = _surface.pitch / 4u;
    HudLine line;

    if (_infinite)
        line.text("LPLKERNEL ENDLESS WORLD  seed ")
            .number(_seed)
            .text("  chunks ")
            .number(static_cast<core::u32>(_streamer.size()))
            .text(" of ")
            .number(_maxResident);
    else
        line.text("LPLKERNEL WORLD VIEWER  seed ")
            .number(_seed)
            .text("  ")
            .number(_gridWidth)
            .text("x")
            .number(_gridDepth);
    drawShadowedText(pitchPixels, 8u, 8u, line, 0x00FFAA22u);

    if (_infinite)
    {
        line.clear()
            .text("at ")
            .integer(static_cast<core::i32>(_camera.focusX()))
            .text(",")
            .integer(static_cast<core::i32>(_camera.focusZ()))
            .text("  built ")
            .number(_streamer.generatedCount())
            .text("  released ")
            .number(_streamer.releasedCount());
        drawShadowedText(pitchPixels, 8u, 26u, line, 0x00C8C8C0u);

        line.clear()
            .text(_camera.isFirstPerson() ? "first person  eye " : "orbit  distance ")
            .decimal(_camera.isFirstPerson() ? _camera.eyeHeight() : _camera.distance())
            .text("  LOD rings ")
            .number(_lodRings)
            .text("  tris ")
            .number(_lastTriangles);
        drawShadowedText(pitchPixels, 8u, 44u, line, 0x00A0B4C8u);

        line.clear()
            .text(_embodied ? "body " : "detached ")
            .text(_body.isSliding() ? "sliding" : (_body.isGrounded() ? "grounded" : "airborne"))
            .text("  y ")
            .decimal(_body.y().toFloat())
            .text("  speed ")
            .decimal(_body.groundSpeed().toFloat())
            .text("  jumps ")
            .number(_body.jumpCount())
            .text("  blocked ")
            .number(_body.blockedCount());
        drawShadowedText(pitchPixels, 8u, 62u, line, _body.isGrounded() ? 0x00A0B4C8u : 0x00E0C070u);

        line.clear()
            .text("herd ")
            .number(countSpecies(0u))
            .text(" grazers  ")
            .number(countSpecies(1u))
            .text(" hunters  grazed ")
            .number(_living.grazedCount());
        drawShadowedText(pitchPixels, 8u, 80u, line, 0x0060FF80u);

        line.clear()
            .text("cost% sim ")
            .number(_simShare)
            .text("  scene ")
            .number(_sceneShare)
            .text("  blit ")
            .number(_blitShare)
            .text("  present ")
            .number(_presentShare)
            .text("  | sky ")
            .number(_skyShare)
            .text(" gnd ")
            .number(_groundShare)
            .text(" wat ")
            .number(_waterShare)
            .text(" prop ")
            .number(_propShare);
        drawShadowedText(pitchPixels, 8u, 98u, line, 0x00FFAA22u);

        line.clear()
            .text("scent window ")
            .number(kFieldSpan)
            .text(" cells, recentred ")
            .number(_living.scent().recentres())
            .text(" times");
        drawShadowedText(pitchPixels, 8u, 116u, line, 0x00A0B4C8u);

        line.clear()
            .text("drawn ")
            .number(_renderer.view().stats().drawn)
            .text(" of ")
            .number(static_cast<core::u32>(_streamer.size()))
            .text(" chunks, culled ")
            .number(_renderer.view().stats().culled)
            .text("  boxes ")
            .number(_renderer.view().stats().considered)
            .text("-")
            .number(_renderer.view().stats().culled)
            .text("  f/100t ")
            .number(_windowTicks == 0u ? 0u : (_windowFrames * 100u) / _windowTicks)
            .text(_pbrSurface ? "  pbr" : (_perPixelSurface ? "  per-pixel" : "  flat"))
            .text("  sky/")
            .number(_skyBlock);
        drawShadowedText(pitchPixels, 8u, 134u, line, 0x00A0B4C8u);

        core::u32 navigable = 0u;
        const core::u32 warrens = _streamer.residentWarrens(&navigable);
        line.clear()
            .text("landmarks: ")
            .number(_streamer.residentCaveMouths())
            .text(" mouths  ")
            .number(warrens)
            .text(" caves (")
            .number(navigable)
            .text(" deep)  ")
            .number(_streamer.residentBuildings())
            .text(" buildings");
        drawShadowedText(pitchPixels, 8u, 152u, line, 0x00C8A050u);

        line.clear()
            .text(_body.isEnclosed() ? "UNDERGROUND  y " : "surface      y ")
            .integer(static_cast<core::i32>(_body.y().toFloat()))
            .text("  ceiling ")
            .integer(_body.isEnclosed() ? static_cast<core::i32>(_body.ceilingHeight().toFloat()) : 0)
            .text("  head ")
            .number(_body.headBumpCount())
            .text("  ducked ")
            .number(_body.duckedCount());
        drawShadowedText(pitchPixels, 8u, 170u, line, _body.isEnclosed() ? 0x00E08040u : 0x00708090u);

        image::drawText8x16(
            _surface.buffer, pitchPixels, 8u, _surface.height - 20u,
            "WASD=walk SPACE=jump C=sprint mouse/JL=turn IK=tilt M=warp to cave V=detach F=view O=bounded X=exit",
            0x00808890u);
        return;
    }

    line.clear()
        .text(shadingName())
        .number(presentBiomes())
        .text(" biomes  ")
        .number(standingPlants())
        .text(" plants of ")
        .number(static_cast<core::u32>(_plants.size()));
    drawShadowedText(pitchPixels, 8u, 26u, line, 0x00C8C8C0u);

    line.clear()
        .text("herd ")
        .number(countSpecies(0u))
        .text(" grazers  ")
        .number(countSpecies(1u))
        .text(" hunters  grazed ")
        .number(_living.grazedCount());
    drawShadowedText(pitchPixels, 8u, 44u, line, 0x0060FF80u);

    line.clear()
        .text("cartridge: ")
        .number(_plotCount)
        .text(" plots  ")
        .number(_roadCells)
        .text(" road cells  ")
        .number(_propCount)
        .text(" props");
    drawShadowedText(pitchPixels, 8u, 62u, line, 0x00A0B4C8u);

    line.clear()
        .text("rivers ")
        .number(_riverCells)
        .text("  cave floor ")
        .number(_caveFloor)
        .text("  gate ")
        .number(_gatePassed ? 1u : 0u);
    drawShadowedText(pitchPixels, 8u, 80u, line, 0x00A0B4C8u);

    image::drawText8x16(_surface.buffer, pitchPixels, 8u, _surface.height - 20u,
                        "WASD=cam Q/E=zoom N=new seed B=shading O=endless X=exit", 0x00808890u);
}

inline const char *TerrainWorld::shadingName() const noexcept
{
    switch (_shading)
    {
    case Shading::Height: return "shading height   ";
    case Shading::Moisture: return "shading moisture ";
    case Shading::Biome:
    case Shading::Count: break;
    }
    return "shading biome    ";
}

inline void TerrainWorld::drainInput(engine::WorldContext &context)
{
    drainPointer(context);
    sampleHeldKeys(context);

    char key;
    while (context.platform.input().tryPopCharacter(key))
    {
        switch (key)
        {
        case 'w':
            if (!_infinite)
                _camera.tilt(0.06f);
            else if (!bodyDrivesMovement(context))
                nudge(1.0f, 0.0f);
            break;
        case 's':
            if (!_infinite)
                _camera.tilt(-0.06f);
            else if (!bodyDrivesMovement(context))
                nudge(-1.0f, 0.0f);
            break;
        case ' ': _jumpPressed = true; break;
        case 'v':
            _embodied = !_embodied;
            _camera.setEyeHeight(_embodied ? _bodyParams.eyeHeight.toFloat() : kDetachedEyeHeight);
            break;
        case 'a':
            if (!_infinite)
                _camera.turn(-0.08f);
            else if (!bodyDrivesMovement(context))
                nudge(0.0f, -1.0f);
            break;
        case 'd':
            if (!_infinite)
                _camera.turn(0.08f);
            else if (!bodyDrivesMovement(context))
                nudge(0.0f, 1.0f);
            break;
        case 'j': look(-0.10f); break;
        case 'l': look(0.10f); break;
        case 'i': _camera.tilt(0.06f); break;
        case 'k': _camera.tilt(-0.06f); break;
        case 'm':
        case ';': teleportToNearestCaveMouth(); break;
        case 'o': setInfinite(!_infinite); break;
        case 'r':
            _reflectionProbeOn = !_reflectionProbeOn;
            resetRateWindow();
            break;
        case 'g':
            _pbrSurface = !_pbrSurface;
            resetRateWindow();
            break;
        case 'y':
            _skyBlock = _skyBlock == 1u ? 3u : 1u;
            resetRateWindow();
            break;
        case 't':
            _perPixelSurface = !_perPixelSurface;
            resetRateWindow();
            break;
        case 'f':
            _camera.setFirstPerson(!_camera.isFirstPerson());
            _camera.setPitch(_camera.isFirstPerson() ? kFirstPersonPitch : kOrbitPitch);
            break;
        case 'q': _camera.zoom(0.88f, 1.6f, 160.0f); break;
        case 'e': _camera.zoom(1.14f, 1.6f, 160.0f); break;
        case 'b':
            _shading =
                static_cast<Shading>((static_cast<core::u32>(_shading) + 1u) % static_cast<core::u32>(Shading::Count));
            break;
        case 'n':
            _seed = _seed * 1664525u + 1013904223u;
            generate();
            break;
        case 'x':
        case 27:
            if (context.engine != nullptr)
                context.engine->requestShutdown();
            break;
        default: {
            const char shown[2] = {static_cast<char>(key), '\0'};
            HudLine line;
            line.text("input: nothing is bound to '").text(shown).text("'");
            core::Log::info(line.c_str());
            break;
        }
        }
    }
}

inline void TerrainWorld::drainPointer(engine::WorldContext &context)
{
    core::i32 deltaX = 0;
    core::i32 deltaY = 0;
    core::u32 buttons = 0u;
    core::i32 turnAccumulator = 0;
    core::i32 tiltAccumulator = 0;

    while (context.platform.input().tryPopPointerMotion(deltaX, deltaY, buttons))
    {
        turnAccumulator += deltaX;
        tiltAccumulator += deltaY;
        _pointerButtons = buttons;
        ++_pointerPackets;
    }

    if (_pointerPackets != 0u && _pointerReported == 0u)
    {
        _pointerReported = _pointerPackets;
        core::Log::info("pointer: motion received — the device and the driver are fine");
    }
    else if (_pointerPackets == 0u && _frames == 240u)
    {
        core::Log::info(context.platform.input().pointerInterruptCount() != 0u ?
                            "pointer: interrupts arrive but no packet is assembled" :
                            "pointer: no interrupt from the device at all");
    }

    if (turnAccumulator != 0)
        look(static_cast<core::f32>(turnAccumulator) * kLookSensitivity);
    if (tiltAccumulator != 0)
        _camera.tilt(static_cast<core::f32>(tiltAccumulator) * kLookSensitivity);
}

inline void TerrainWorld::sampleHeldKeys(engine::WorldContext &context)
{
    const platform::IInputBackend &input = context.platform.input();
    if (!input.hasKeyStates())
        return;

    const bool ahead = input.isKeyHeld('w') || input.isKeyHeld('z');
    const bool back = input.isKeyHeld('s');
    const bool left = input.isKeyHeld('a') || input.isKeyHeld('q');
    const bool right = input.isKeyHeld('d');

    _intent.forward = (ahead == back) ? math::Fixed32{} : (ahead ? math::Fixed32::one() : -math::Fixed32::one());
    _intent.strafe = (left == right) ? math::Fixed32{} : (right ? math::Fixed32::one() : -math::Fixed32::one());
    _intent.sprint = input.isKeyHeld('c');
}

inline void TerrainWorld::look(core::f32 yawRadians) noexcept
{
    if (_embodied && _infinite)
        _pendingTurn = _pendingTurn - math::Fixed32::fromFloat(yawRadians);
    else
        _camera.turn(yawRadians);
}

inline void TerrainWorld::ensureBodySpawned()
{
    if (_bodySpawned || _streamer.size() < 9u)
        return;
    spawnBody();
    _bodySpawned = true;
}

inline void TerrainWorld::spawnBody()
{
    const auto ground = [this](core::i32 x, core::i32 z) { return _streamer.groundHeightAt(x, z); };
    const auto space = [this](core::i32 x, core::i32 z, math::Fixed32 y) { return _streamer.spanAt(x, z, y); };
    const math::Fixed32 shore = math::Fixed32::fromFloat(seaLevel() + 0.5f);

    const core::i32 reach = spawnSearchCells();
    for (core::i32 radius = 0; radius <= reach; ++radius)
    {
        for (core::i32 offsetZ = -radius; offsetZ <= radius; ++offsetZ)
            for (core::i32 offsetX = -radius; offsetX <= radius; ++offsetX)
            {
                const core::i32 ax = offsetX < 0 ? -offsetX : offsetX;
                const core::i32 az = offsetZ < 0 ? -offsetZ : offsetZ;
                if (ax != radius && az != radius)
                    continue;

                const math::Fixed32 here = ground(offsetX, offsetZ);
                if (here < shore)
                    continue;
                const math::Fixed32 slopeX = ground(offsetX + 1, offsetZ) - here;
                const math::Fixed32 slopeZ = ground(offsetX, offsetZ + 1) - here;
                if (slopeX > _bodyParams.maxSlope || -slopeX > _bodyParams.maxSlope || slopeZ > _bodyParams.maxSlope ||
                    -slopeZ > _bodyParams.maxSlope)
                    continue;

                _body.placeAt(math::Fixed32::fromInt(offsetX), math::Fixed32::fromInt(offsetZ), here, space);
                _camera.setFocus(static_cast<core::f32>(offsetX), static_cast<core::f32>(offsetZ));
                return;
            }
    }

    core::Log::warn("TerrainWorld: no dry ground within the spawn search — starting in the water");
    _body.placeAt(math::Fixed32{}, math::Fixed32{}, ground(0, 0), space);
}

inline void TerrainWorld::teleportToNearestCaveMouth()
{
    if (!_infinite)
    {
        core::Log::info("cave warp: only the endless world has cave mouths in it");
        return;
    }

    procgen::CaveWarren nearest;
    const procgen::ChunkTerrainRule &rule = _streamer.rule();
    if (!procgen::findNearestCaveWarren(_streamer.chunkParams(), rule.caveMouths, rule.warren, rule.seaLevel,
                                        rule.caveMouthDrop, static_cast<core::i32>(_camera.focusX()),
                                        static_cast<core::i32>(_camera.focusZ()), kCaveWarpRings, kCaveWarpBuilds,
                                        nearest))
    {
        core::Log::info("cave warp: no cave within the search — this world may have none nearby");
        return;
    }

    const core::i32 standX = nearest.site.cellX;
    const core::i32 standZ = nearest.site.cellZ;

    const auto space = [this](core::i32 x, core::i32 z, math::Fixed32 y) { return _streamer.spanAt(x, z, y); };
    _body.placeAt(math::Fixed32::fromInt(standX), math::Fixed32::fromInt(standZ),
                  math::Fixed32::fromFloat(nearest.adit.floorY), space);
    _body.setYaw(
        math::Cordic::atan2(math::Fixed32::fromInt(-nearest.adit.stepX), math::Fixed32::fromInt(-nearest.adit.stepZ)));
    _pendingTurn = math::Fixed32{};
    _jumpPressed = false;
    _bodySpawned = true;

    _embodied = true;
    _camera.setFirstPerson(true);
    _camera.setPitch(kFirstPersonPitch);
    _camera.setEyeHeight(_bodyParams.eyeHeight.toFloat());
    _camera.setFocus(static_cast<core::f32>(standX), static_cast<core::f32>(standZ));
    _camera.setYaw(_body.yaw().toFloat());

    HudLine line;
    line.text("cave warp: ")
        .integer(standX)
        .text(",")
        .integer(standZ)
        .text("  open ")
        .number(nearest.openCells)
        .text("  deep ")
        .number(nearest.navigable ? 1u : 0u)
        .text(" - walk forward");
    core::Log::info(line.c_str());
}

inline void TerrainWorld::nudge(core::f32 forward, core::f32 strafeAmount) noexcept
{
    if (!_embodied)
    {
        _camera.walk(forward * kWalkStep);
        _camera.strafe(strafeAmount * kWalkStep);
        return;
    }
    _intent.forward = math::Fixed32::fromFloat(forward);
    _intent.strafe = math::Fixed32::fromFloat(strafeAmount);
}

inline math::Fixed32 TerrainWorld::cellToWorldX(core::u32 cell) const noexcept
{
    return math::Fixed32::fromInt(static_cast<core::i32>(cell)) -
           math::Fixed32::fromInt(static_cast<core::i32>(_gridWidth / 2u));
}

inline math::Fixed32 TerrainWorld::cellToWorldZ(core::u32 cell) const noexcept
{
    return math::Fixed32::fromInt(static_cast<core::i32>(cell)) -
           math::Fixed32::fromInt(static_cast<core::i32>(_gridDepth / 2u));
}

inline bool TerrainWorld::worldToCell(math::Fixed32 x, math::Fixed32 z, core::u32 &outX, core::u32 &outZ) const noexcept
{
    const core::i32 gx = (x + math::Fixed32::fromInt(static_cast<core::i32>(_gridWidth / 2u))).toInt();
    const core::i32 gz = (z + math::Fixed32::fromInt(static_cast<core::i32>(_gridDepth / 2u))).toInt();
    if (gx < 0 || gz < 0 || static_cast<core::u32>(gx) >= _gridWidth || static_cast<core::u32>(gz) >= _gridDepth)
        return false;
    outX = static_cast<core::u32>(gx);
    outZ = static_cast<core::u32>(gz);
    return true;
}

inline core::u32 TerrainWorld::presentBiomes() const noexcept
{
    core::u32 present = 0u;
    for (core::u32 i = 0u; i < static_cast<core::u32>(procgen::BiomeId::Count); ++i)
        if (_biomeCounts[i] != 0u)
            ++present;
    return present;
}

} // namespace lpl::samples

#endif // LPL_SAMPLES_TERRAINWORLD_INL
