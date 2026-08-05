/**
 * @file ShapeGrammar.cpp
 * @brief Implementation of the split grammar and its string form.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/procgen/ShapeGrammar.hpp>

#include <lpl/math/Random.hpp>

namespace lpl::procgen {

namespace {

/// Skips spaces. Everything else is significant.
void skipSpace(const char *text, core::u32 &i)
{
    while (text[i] == ' ' || text[i] == '\t')
        ++i;
}

/**
 * @brief Parses `[SYM]` or `[SYM,SYM]` into one module.
 *
 * A module may name several symbols (`[A,P]` is a panel plus its post); the first
 * decides the material and the count decides the height, which is the smallest
 * reading of the notation that does not throw information away.
 */
bool parseModule(const char *text, core::u32 &i, GrammarModule &out)
{
    skipSpace(text, i);
    if (text[i] != '[')
        return false;
    ++i;

    core::u32 symbols = 0u;
    core::u8 material = 0u;
    for (;;)
    {
        skipSpace(text, i);
        const char letter = text[i];
        if (materialForSymbol(letter) == 0u)
            return false; // An empty or non-alphabetic symbol is a typo, not a module.
        if (symbols == 0u)
            material = materialForSymbol(letter);

        // Consume the rest of a multi-letter symbol (BL, BS...).
        while (materialForSymbol(text[i]) != 0u)
            ++i;
        ++symbols;

        skipSpace(text, i);
        if (text[i] == ',')
        {
            ++i;
            continue;
        }
        if (text[i] == ']')
        {
            ++i;
            break;
        }
        return false;
    }

    out.material = material;
    out.height = static_cast<core::u8>(symbols);
    out.weight = 1u;
    return true;
}

} // namespace

bool parseSequenceGrammar(const char *text, SequenceGrammar &out)
{
    out = SequenceGrammar{};
    if (text == nullptr)
        return false;

    // Bounded up front: a grammar arriving from a document is untrusted input,
    // and a parser that walks until NUL is a parser that walks off the end of a
    // truncated file.
    core::u32 length = 0u;
    while (text[length] != '\0')
    {
        ++length;
        if (length > kMaxGrammarLength)
            return false;
    }

    core::u32 i = 0u;
    skipSpace(text, i);
    if (text[i] != '{')
        return false;
    ++i;

    for (;;)
    {
        if (out.alternativeCount >= kMaxGrammarModules)
            return false;

        GrammarModule module{};
        if (!parseModule(text, i, module))
            return false;

        skipSpace(text, i);
        if (text[i] == ':')
        {
            ++i;
            skipSpace(text, i);
            core::u32 weight = 0u;
            core::u32 digits = 0u;
            while (text[i] >= '0' && text[i] <= '9')
            {
                weight = weight * 10u + static_cast<core::u32>(text[i] - '0');
                ++i;
                ++digits;
                if (weight > 0xFFFFu)
                    return false;
            }
            if (digits == 0u || weight == 0u)
                return false; // ":0" and ":" are both meaningless.
            module.weight = static_cast<core::u16>(weight);
        }

        out.alternatives[out.alternativeCount] = module;
        ++out.alternativeCount;
        out.totalWeight += module.weight;

        skipSpace(text, i);
        if (text[i] == ',')
        {
            ++i;
            continue;
        }
        if (text[i] == '}')
        {
            ++i;
            break;
        }
        return false;
    }

    skipSpace(text, i);
    if (text[i] != '*')
        return false;
    ++i;

    skipSpace(text, i);
    if (text[i] == ',')
    {
        ++i;
        if (!parseModule(text, i, out.terminator))
            return false;
        out.hasTerminator = true;
    }

    skipSpace(text, i);
    // The whole string must be consumed. Trailing garbage means the author meant
    // something the parser did not understand, and quietly ignoring it is how a
    // typo becomes a shipped world.
    return text[i] == '\0' && out.alternativeCount > 0u && out.totalWeight > 0u;
}

core::u32 applySequence(const SequenceGrammar &grammar, core::u32 length, core::u32 seed,
                        lpl::pmr::vector<GrammarModule> &out)
{
    out.clear();
    if (length == 0u || grammar.alternativeCount == 0u || grammar.totalWeight == 0u)
        return 0u;

    out.reserve(length);
    math::Random random{seed};

    const core::u32 body = grammar.hasTerminator && length > 0u ? length - 1u : length;
    for (core::u32 slot = 0u; slot < body; ++slot)
    {
        // Roulette selection over the declared weights: `:2` really is twice as
        // likely as `:1`, which is the only reading of the notation that makes
        // the numbers mean anything.
        core::u32 pick = random.below(grammar.totalWeight);
        core::u32 chosen = 0u;
        for (core::u32 a = 0u; a < grammar.alternativeCount; ++a)
        {
            if (pick < grammar.alternatives[a].weight)
            {
                chosen = a;
                break;
            }
            pick -= grammar.alternatives[a].weight;
        }
        out.push_back(grammar.alternatives[chosen]);
    }
    if (grammar.hasTerminator && length > 0u)
        out.push_back(grammar.terminator);

    return static_cast<core::u32>(out.size());
}

VoxelVolume buildingVolume(const BuildingPlot &plot, const BuildingGrammarParams &params, core::u32 seed)
{
    VoxelVolume volume;
    if (plot.width == 0u || plot.depth == 0u)
        return volume;

    math::Random random{seed};

    const core::u32 minFloors = params.minFloors == 0u ? 1u : params.minFloors;
    const core::u32 maxFloors = params.maxFloors < minFloors ? minFloors : params.maxFloors;
    const core::u32 floors = minFloors + random.below(maxFloors - minFloors + 1u);

    const core::u32 baseLevels = params.baseHeight;
    const core::u32 floorLevels = params.floorHeight == 0u ? 1u : params.floorHeight;
    const core::u32 bodyLevels = floors * floorLevels;
    const core::u32 roofLevels = params.roofHeight;

    volume.width = plot.width;
    volume.depth = plot.depth;
    volume.levels = baseLevels + bodyLevels + roofLevels;
    if (volume.levels == 0u)
        return VoxelVolume{};
    volume.cells.resize(static_cast<core::usize>(volume.width) * volume.depth * volume.levels, core::u8{0});

    // Inset is clamped so a narrow plot does not produce a building of negative
    // width — which, unsigned, is a building the size of the world.
    const core::u32 maxInset = (plot.width < plot.depth ? plot.width : plot.depth) / 2u;
    const core::u32 inset = params.inset > maxInset ? maxInset : params.inset;

    const auto fillLevel = [&](core::u32 level, core::u32 shrink, core::u8 material, bool hollow) {
        if (level >= volume.levels || material == 0u)
            return;
        const core::u32 pad = inset + shrink;
        if (pad * 2u >= volume.width || pad * 2u >= volume.depth)
            return;
        for (core::u32 z = pad; z + pad < volume.depth; ++z)
            for (core::u32 x = pad; x + pad < volume.width; ++x)
            {
                const bool onEdge =
                    x == pad || z == pad || x + pad + 1u == volume.width || z + pad + 1u == volume.depth;
                if (hollow && !onEdge)
                    continue;
                volume.at(x, level, z) = material;
            }
    };

    // ── Split Y: base | floors* | roof ──────────────────────────────────────
    core::u32 level = 0u;
    for (core::u32 b = 0u; b < baseLevels; ++b, ++level)
        fillLevel(level, 0u, params.baseMaterial, false); // The base is solid: it is the floor.

    for (core::u32 f = 0u; f < bodyLevels; ++f, ++level)
        fillLevel(level, 0u, params.wallMaterial, params.hollow);

    // The roof steps inward as it rises, by the requested share. A flat cap is
    // what `roofTaper = 0` asks for and it is a legitimate answer, so it is not
    // treated as a degenerate case.
    for (core::u32 r = 0u; r < roofLevels; ++r, ++level)
    {
        const core::u32 shrink = roofLevels <= 1u ?
                                     0u :
                                     static_cast<core::u32>((math::Fixed32::fromFloat(params.roofTaper) *
                                                             math::Fixed32::fromInt(static_cast<core::i32>(r)))
                                                                .toInt());
        fillLevel(level, shrink, params.roofMaterial, false);
    }

    return volume;
}

VoxelVolume buildTown(const SettlementMap &settlement, const lpl::pmr::vector<BuildingPlot> &plots,
                      const BuildingGrammarParams &params, core::u32 worldSeed, core::u32 levels)
{
    VoxelVolume town;
    if (settlement.empty() || levels == 0u)
        return town;

    town.width = settlement.width();
    town.depth = settlement.depth();
    town.levels = levels;
    town.cells.resize(static_cast<core::usize>(town.width) * town.depth * town.levels, core::u8{0});

    for (core::u32 p = 0u; p < plots.size(); ++p)
    {
        const BuildingPlot &plot = plots[p];

        // Keyed by the plot's POSITION, not by its index in the list. A plot
        // inserted earlier in the vector must not re-roll every building after
        // it, or adding one house redraws the town.
        const core::u32 seed = params.seed != 0u ? params.seed ^ (plot.x * 73856093u) ^ (plot.z * 19349663u) :
                                                   worldSeed ^ (plot.x * 73856093u) ^ (plot.z * 19349663u);

        const VoxelVolume building = buildingVolume(plot, params, seed);
        if (building.empty())
            continue;

        for (core::u32 z = 0u; z < building.depth; ++z)
        {
            for (core::u32 x = 0u; x < building.width; ++x)
            {
                const core::u32 wx = plot.x + x;
                const core::u32 wz = plot.z + z;
                if (wx >= town.width || wz >= town.depth)
                    continue;

                // The rule that stops buildings standing in their own street. A
                // plot is a rectangle proposed before the roads were cut; by now
                // part of it is road, and only what the map still calls Plot may
                // be raised.
                if (settlement.at(wx, wz) != SettlementCell::Plot)
                    continue;

                for (core::u32 y = 0u; y < building.levels && y < town.levels; ++y)
                {
                    const core::u8 material = building.at(x, y, z);
                    if (material != 0u)
                        town.at(wx, y, wz) = material;
                }
            }
        }
    }
    return town;
}

VoxelVolume decoratePath(const Grid<core::u8> &path, const SequenceGrammar &grammar, core::u32 seed, core::u32 levels,
                         core::u32 &outCount)
{
    outCount = 0u;
    VoxelVolume volume;
    if (path.empty() || levels == 0u || grammar.alternativeCount == 0u)
        return volume;

    volume.width = path.width();
    volume.depth = path.depth();
    volume.levels = levels;
    volume.cells.resize(static_cast<core::usize>(volume.width) * volume.depth * volume.levels, core::u8{0});

    // Scan order, so the sequence does not depend on how the path was traced —
    // two callers that produce the same marked cells get the same fence.
    lpl::pmr::vector<core::u32> cells;
    for (core::u32 i = 0u; i < path.cellCount(); ++i)
        if (path[i] != 0u)
            cells.push_back(i);
    if (cells.empty())
        return volume;

    lpl::pmr::vector<GrammarModule> modules;
    const core::u32 placed = applySequence(grammar, static_cast<core::u32>(cells.size()), seed, modules);

    for (core::u32 i = 0u; i < placed && i < cells.size(); ++i)
    {
        const GrammarModule &module = modules[i];
        if (module.material == 0u)
            continue;
        const core::u32 x = cells[i] % volume.width;
        const core::u32 z = cells[i] / volume.width;
        const core::u32 top = module.height < levels ? module.height : levels;
        for (core::u32 y = 0u; y < top; ++y)
            volume.at(x, y, z) = module.material;
        ++outCount;
    }
    return volume;
}

} // namespace lpl::procgen
