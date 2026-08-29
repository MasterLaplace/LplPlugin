/**
 * @file ReliefParity.cpp
 * @brief Implementation of the relief determinism gate.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/engine/ReliefParity.hpp>

#include <lpl/math/Geo.hpp>
#include <lpl/procgen/Chunking.hpp>

namespace lpl::engine {

namespace {

constexpr core::u32 kFnv1aOffsetBasis = 0x811C9DC5u;
constexpr core::u32 kFnv1aPrime = 0x01000193u;

/**
 * @brief Folds one word into a running signature, byte by byte.
 *
 * @warning Byte by byte and not word at a time: a fold over machine words would agree with itself on
 * one endianness and quietly disagree across targets, which is the exact class of disagreement a
 * parity gate exists to catch.
 *
 * @param signature Running value.
 * @param value     What to fold.
 */
void fold(core::u32 &signature, core::u32 value)
{
    for (core::u32 shift = 0u; shift < 32u; shift += 8u)
    {
        signature ^= (value >> shift) & 0xFFu;
        signature *= kFnv1aPrime;
    }
}

/// Cells on a side of the canonical survey. Small enough to live in BSS on both targets.
constexpr core::u32 kSurveySide = 96u;
/// Cells of border band. Wide enough that the walk crosses it and meets the fade.
constexpr core::u32 kBlendCells = 12u;
/// Steps the body takes downhill.
constexpr core::u32 kWalkSteps = 120u;

/**
 * The survey's samples.
 *
 * @warning **Static, and that is a kernel requirement rather than a style.** Nine thousand samples is
 * eighteen kilobytes; on a kernel stack that is an overflow, and the same trap has already been
 * paid once here by a scene of a thousand entities. BSS on both targets, filled once.
 */
core::i16 gSamples[kSurveySide * kSurveySide];
bool gSamplesReady = false;

/**
 * @brief Fills the canonical survey.
 *
 * A coast: land in the west, falling east past sea level into bathymetry, with a ridge across it so
 * a body has somewhere to walk down from. A rectangular hole is punched in the middle, because a
 * survey with no gaps never exercises the branch that falls back to invented ground -- and that
 * branch is the one whose failure looks like ground thirty-two kilometres below the sea.
 */
void buildSurvey()
{
    if (gSamplesReady)
        return;
    for (core::u32 row = 0u; row < kSurveySide; ++row)
    {
        for (core::u32 col = 0u; col < kSurveySide; ++col)
        {
            // Falls from +900 m in the west to -300 m in the east, so the coastline is inside the
            // survey and both signs of elevation are exercised.
            const core::i32 slope = 900 - static_cast<core::i32>(col) * 15;
            // A ridge running north-south, so the surface is not a plane and a downhill walk has a
            // direction to choose rather than a tie to break.
            const core::i32 distance = static_cast<core::i32>(col) - 30;
            const core::i32 ridge = distance * distance < 400 ? (400 - distance * distance) / 4 : 0;
            // A gentle north-south variation, so rows are distinguishable: a field constant along a
            // row would be read identically by a transposed index.
            const core::i32 northSouth = static_cast<core::i32>(row) * 3;

            core::i16 metres = static_cast<core::i16>(slope + ridge + northSouth);
            // The hole in the survey.
            if (row >= 40u && row < 56u && col >= 60u && col < 76u)
                metres = math::kReliefNoSample;
            gSamples[row * kSurveySide + col] = metres;
        }
    }
    gSamplesReady = true;
}

/**
 * @brief The projection the canonical world stands on.
 *
 * @return It, resolved.
 */
[[nodiscard]] math::ReliefProjection canonicalProjection()
{
    math::GeoProjection spec{};
    // The southern Peloponnese, which is the ground the reader was measured against. The corner is
    // the NORTH-west one; see lpl/math/Geo.hpp for why that is worth an unfamiliar corner.
    spec.originLatitudeRaw = 37 * 65536;
    spec.originLongitudeRaw = 22 * 65536;
    spec.referenceLatitude = 37;
    spec.metresPerCell = 30u;
    // A quarter unit per metre: enough that a nine-hundred-metre hill is a real climb in world
    // units, and far enough from saturation that the whole range of the earth still fits.
    spec.unitsPerMetre = math::Fixed32::fromFloat(0.25f);
    // THE reconciliation: elevation zero lands here and nowhere else.
    spec.seaLevelUnits = math::Fixed32::fromFloat(-1.0f);
    return math::makeReliefProjection(spec);
}

/**
 * @brief The world parameters, with or without the survey behind them.
 *
 * @param field Survey to stand on, or nullptr for the control.
 * @return The parameters.
 */
[[nodiscard]] procgen::ChunkParams canonicalParams(const math::ReliefMosaic *field)
{
    procgen::ChunkParams params{};
    params.size = 32u;
    params.worldSeed = 20260816u;
    params.noise.amplitude = 40.0f;
    params.noise.frequency = 0.02f;
    params.noise.octaves = 4u;
    params.noise.baseHeight = 0.0f;
    params.relief = field;
    // The high frequencies thirty-metre samples cannot carry. Silent in the control, because the
    // control must be the world as it was before any of this existed.
    if (field != nullptr)
    {
        params.reliefDetail.amplitude = 1.5f;
        params.reliefDetail.frequency = 0.35f;
        params.reliefDetail.octaves = 2u;
    }
    return params;
}

/**
 * @brief Runs the canonical world and folds it.
 *
 * @param field Survey, or nullptr.
 * @return The signatures.
 */
[[nodiscard]] ReliefFoldResult run(const math::ReliefMosaic *field)
{
    ReliefFoldResult out{};
    out.sampleSignature = kFnv1aOffsetBasis;
    out.heightSignature = kFnv1aOffsetBasis;
    out.walkSignature = kFnv1aOffsetBasis;
    out.coastSignature = kFnv1aOffsetBasis;

    const math::ReliefProjection projection = canonicalProjection();
    const procgen::ChunkParams params = canonicalParams(field);
    const math::Fixed32 sea = projection.projection.seaLevelUnits;

    // The survey itself, folded before anything reads it: a signature over the ground separates
    // "the sampler changed" from "the survey changed", which one signature over the result cannot.
    for (core::u32 i = 0u; i < kSurveySide * kSurveySide; ++i)
        fold(out.sampleSignature, static_cast<core::u32>(static_cast<core::u16>(gSamples[i])));

    // A window deliberately WIDER than the survey, so it spans measured ground, the border band and
    // invented ground beyond. A window that stopped at the edge would never fold the blend, which
    // is the part with arithmetic in it.
    constexpr core::i32 kMargin = 20;
    const core::i32 from = -kMargin;
    const core::i32 to = static_cast<core::i32>(kSurveySide) + kMargin;
    for (core::i32 z = from; z < to; z += 2)
    {
        for (core::i32 x = from; x < to; x += 2)
        {
            const math::Fixed32 height = procgen::sampleWorldHeight(params, x, z);
            fold(out.heightSignature, static_cast<core::u32>(height.raw()));

            const bool drowned = height <= sea;
            if (drowned)
                ++out.seaCells;
            fold(out.coastSignature, drowned ? 1u : 0u);

            if (field == nullptr)
            {
                ++out.inventedCells;
                continue;
            }
            math::Fixed32 measured{};
            const bool hasGround = field->heightAt(x, z, measured);
            const math::Fixed32 weight = field->weightAt(x, z);
            if (!hasGround || weight.raw() <= 0)
                ++out.inventedCells;
            else if (weight.raw() >= math::Fixed32::one().raw())
                ++out.measuredCells;
            else
                ++out.blendedCells;
        }
    }

    // A body walking downhill. Not a physics tick: the claim under test is that the GROUND agrees
    // on two targets, and a walk driven by the ground alone fails when the ground disagrees while
    // adding nothing else that could.
    core::i32 x = 30;
    core::i32 z = 48;
    const math::Fixed32 start = procgen::sampleWorldHeight(params, x, z);
    for (core::u32 step = 0u; step < kWalkSteps; ++step)
    {
        math::Fixed32 here = procgen::sampleWorldHeight(params, x, z);
        core::i32 bestX = x;
        core::i32 bestZ = z;
        // Ties break on the lowest neighbour index, so a flat shelf is walked identically
        // everywhere rather than according to whatever the compiler ordered.
        for (core::i32 n = 0; n < 8; ++n)
        {
            const core::i32 dx = (n % 3) - 1;
            const core::i32 dz = (n / 3) - 1;
            if (dx == 0 && dz == 0)
                continue;
            const math::Fixed32 there = procgen::sampleWorldHeight(params, x + dx, z + dz);
            if (there < here)
            {
                here = there;
                bestX = x + dx;
                bestZ = z + dz;
            }
        }
        if (bestX == x && bestZ == z)
            break;
        x = bestX;
        z = bestZ;
        ++out.walkSteps;
        fold(out.walkSignature, static_cast<core::u32>(x));
        fold(out.walkSignature, static_cast<core::u32>(z));
        fold(out.walkSignature, static_cast<core::u32>(here.raw()));
    }
    out.descended = start.raw() - procgen::sampleWorldHeight(params, x, z).raw();
    return out;
}

} // namespace

ReliefFoldResult foldReliefParity()
{
    buildSurvey();

    math::ReliefField field{};
    field.samples = gSamples;
    field.width = kSurveySide;
    field.height = kSurveySide;
    field.originCellX = 0;
    field.originCellZ = 0;
    field.projection = canonicalProjection();
    field.blendCells = kBlendCells;
    // A mosaic of ONE, with every edge exposed: exactly what a lone field was before tiling
    // existed. That this gate's signatures do not move is the control on the whole change.
    math::ReliefMosaic mosaic{};
    (void) mosaic.add(&field);
    return run(&mosaic);
}

ReliefFoldResult foldInventedParity()
{
    buildSurvey();
    return run(nullptr);
}

} // namespace lpl::engine
