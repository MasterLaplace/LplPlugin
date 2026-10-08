#include <lpl/math/Geo.hpp>
#include <lpl/procgen/Chunking.hpp>
#include <lpl/testing/Test.hpp>

LPL_TEST_SUITE(relief);

namespace {

constexpr lpl::core::u32 kFnv1aOffsetBasis = 0x811C9DC5u;
constexpr lpl::core::u32 kFnv1aPrime = 0x01000193u;

/**
 * @brief Cells on a side of the canonical survey: small enough to live in BSS on both targets.
 */
constexpr lpl::core::u32 kSurveySide = 96u;

/**
 * @brief Cells of border band: wide enough that the walk crosses it and meets the fade.
 */
constexpr lpl::core::u32 kBlendCells = 12u;

/**
 * @brief Steps the body takes downhill.
 */
constexpr lpl::core::u32 kWalkSteps = 120u;

/**
 * @brief Cells the folded window reaches beyond each edge of the survey.
 */
constexpr lpl::core::i32 kWindowMargin = 20;

/**
 * @struct ReliefFoldResult
 * @brief What gate P21 relief records: a world standing on measured ground, on two targets.
 *
 * @details The gate claims that a world whose lowest frequency is a survey rather than a generator
 *          comes out bit-identical on the Linux oracle and in ring 0: the projection, the cell
 *          lookup, the border blend, the detail layer added back, and a body walking across all
 *          of it. It does not claim the samples are SRTM: they are built from an integer formula,
 *          so both sides hold the same ones without a file; the reader of real tiles is measured
 *          against the real Peloponnese in LplKnowledge's `test-relief`. Reading a `.lplknow` section in ring 0
 *          is what gate P18 already proves; what is new on this path is the arithmetic.
 */
struct ReliefFoldResult {
    lpl::core::u32 sampleSignature{0u}; /**< The survey itself, before anything reads it. */
    lpl::core::u32 heightSignature{0u}; /**< Ground over a window spanning inside, border and outside. */
    lpl::core::u32 walkSignature{0u};   /**< Where a body went, step by step. */
    lpl::core::u32 coastSignature{0u};  /**< Which cells came out sea and which land. */
    lpl::core::u32 measuredCells{0u};   /**< Cells the survey answered for. */
    lpl::core::u32 inventedCells{0u};   /**< Cells outside the survey, or inside its hole. */
    lpl::core::u32 blendedCells{0u};    /**< Cells in the border band, part measured and part invented. */
    lpl::core::u32 seaCells{0u};        /**< Cells at or below the world's sea level. */
    lpl::core::u32 walkSteps{0u};       /**< Steps the body actually took. */
    lpl::core::i32 descended{0};        /**< Raw Q16.16 height the walk lost, start to finish. */
};

/**
 * @brief The survey's samples.
 *
 * @warning Static, which is a kernel requirement rather than a style: nine thousand samples are
 *          eighteen kilobytes, an overflow on a kernel stack. BSS on both targets, filled once.
 */
lpl::core::i16 gSamples[kSurveySide * kSurveySide];
bool gSamplesReady = false;

/**
 * @brief Folds one word into a running signature, byte by byte.
 *
 * @details Byte by byte, not word at a time: a fold over machine words would agree with itself on
 *          one endianness and quietly disagree across targets, the very disagreement a parity gate
 *          exists to catch.
 */
void fold(lpl::core::u32 &signature, lpl::core::u32 value)
{
    for (lpl::core::u32 shift = 0u; shift < 32u; shift += 8u)
    {
        signature ^= (value >> shift) & 0xFFu;
        signature *= kFnv1aPrime;
    }
}

/**
 * @brief The elevation of the canonical survey at one cell, in metres.
 *
 * @details A coast: a slope from +900 m in the west to -300 m in the east, so the coastline is
 *          inside the survey and both signs of elevation are exercised; a ridge running north to
 *          south, so the surface is not a plane and a downhill walk has a direction to choose
 *          rather than a tie to break; and a gentle rise from north to south, so that rows are
 *          distinguishable and a transposed index does not read the same field. A rectangle in
 *          the middle holds no sample: a survey without gaps never exercises the fallback to
 *          invented ground, the branch whose failure looks like ground thirty-two kilometres
 *          below the sea.
 */
[[nodiscard]] lpl::core::i16 surveyMetres(lpl::core::u32 row, lpl::core::u32 column)
{
    if (row >= 40u && row < 56u && column >= 60u && column < 76u)
        return lpl::math::kReliefNoSample;

    const lpl::core::i32 slope = 900 - static_cast<lpl::core::i32>(column) * 15;
    const lpl::core::i32 distance = static_cast<lpl::core::i32>(column) - 30;
    const lpl::core::i32 ridge = distance * distance < 400 ? (400 - distance * distance) / 4 : 0;
    const lpl::core::i32 northSouth = static_cast<lpl::core::i32>(row) * 3;

    return static_cast<lpl::core::i16>(slope + ridge + northSouth);
}

void buildSurvey()
{
    if (gSamplesReady)
        return;
    for (lpl::core::u32 row = 0u; row < kSurveySide; ++row)
    {
        for (lpl::core::u32 column = 0u; column < kSurveySide; ++column)
            gSamples[row * kSurveySide + column] = surveyMetres(row, column);
    }
    gSamplesReady = true;
}

/**
 * @brief The projection the canonical world stands on.
 *
 * @details The southern Peloponnese, the ground the reader was measured against, from its
 *          north-west corner (lpl/math/Geo.hpp says why that corner). A quarter unit per metre
 *          makes a nine-hundred-metre hill a real climb in world units, and stays far enough from
 *          saturation that the whole range of the earth still fits. Elevation zero lands at the
 *          sea level of -1 unit, and nowhere else.
 */
[[nodiscard]] lpl::math::ReliefProjection canonicalProjection()
{
    lpl::math::GeoProjection specification{};

    specification.originLatitudeRaw = 37 * 65536;
    specification.originLongitudeRaw = 22 * 65536;
    specification.referenceLatitude = 37;
    specification.metresPerCell = 30u;
    specification.unitsPerMetre = lpl::math::Fixed32::fromFloat(0.25f);
    specification.seaLevelUnits = lpl::math::Fixed32::fromFloat(-1.0f);
    return lpl::math::makeReliefProjection(specification);
}

/**
 * @brief The world parameters, with or without the survey behind them.
 *
 * @details With a survey, a detail layer adds back the high frequencies thirty-metre samples
 *          cannot carry; it stays silent in the control, which must be the world as it was before
 *          relief existed.
 *
 * @param field Survey to stand on, or nullptr for the control.
 */
[[nodiscard]] lpl::procgen::ChunkParams canonicalParams(const lpl::math::ReliefMosaic *field)
{
    lpl::procgen::ChunkParams params{};

    params.size = 32u;
    params.worldSeed = 20260816u;
    params.noise.amplitude = 40.0f;
    params.noise.frequency = 0.02f;
    params.noise.octaves = 4u;
    params.noise.baseHeight = 0.0f;
    params.relief = field;
    if (field == nullptr)
        return params;
    params.reliefDetail.amplitude = 1.5f;
    params.reliefDetail.frequency = 0.35f;
    params.reliefDetail.octaves = 2u;
    return params;
}

/**
 * @brief Folds the survey itself, before anything reads it: this separates a change of the
 *        sampler from a change of the survey, which one signature over the result cannot.
 */
void foldSurvey(ReliefFoldResult &out)
{
    for (lpl::core::u32 index = 0u; index < kSurveySide * kSurveySide; ++index)
        fold(out.sampleSignature, static_cast<lpl::core::u32>(static_cast<lpl::core::u16>(gSamples[index])));
}

/**
 * @brief Counts whether one cell of the window is measured, blended or invented ground.
 */
void countGround(const lpl::math::ReliefMosaic *field, lpl::core::i32 x, lpl::core::i32 z, ReliefFoldResult &out)
{
    if (field == nullptr)
    {
        ++out.inventedCells;
        return;
    }

    lpl::math::Fixed32 measured{};
    const bool hasGround = field->heightAt(x, z, measured);
    const lpl::math::Fixed32 weight = field->weightAt(x, z);

    if (!hasGround || weight.raw() <= 0)
        ++out.inventedCells;
    else if (weight.raw() >= lpl::math::Fixed32::one().raw())
        ++out.measuredCells;
    else
        ++out.blendedCells;
}

/**
 * @brief Folds the ground and the coastline over a window wider than the survey.
 *
 * @details Wider on purpose, so the window spans measured ground, the border band and invented
 *          ground beyond: a window that stopped at the edge would never fold the blend, the part
 *          with arithmetic in it.
 */
void foldWindow(const lpl::procgen::ChunkParams &params, const lpl::math::ReliefMosaic *field, lpl::math::Fixed32 sea,
                ReliefFoldResult &out)
{
    const lpl::core::i32 from = -kWindowMargin;
    const lpl::core::i32 to = static_cast<lpl::core::i32>(kSurveySide) + kWindowMargin;

    for (lpl::core::i32 z = from; z < to; z += 2)
    {
        for (lpl::core::i32 x = from; x < to; x += 2)
        {
            const lpl::math::Fixed32 height = lpl::procgen::sampleWorldHeight(params, x, z);
            const bool drowned = height <= sea;

            fold(out.heightSignature, static_cast<lpl::core::u32>(height.raw()));
            if (drowned)
                ++out.seaCells;
            fold(out.coastSignature, drowned ? 1u : 0u);
            countGround(field, x, z, out);
        }
    }
}

/**
 * @brief Walks a body downhill from a fixed cell, to the lowest of its eight neighbours each step.
 *
 * @details Not a physics tick: the claim is that the ground agrees on two targets, and a walk
 *          driven by the ground alone fails when the ground disagrees while adding nothing else
 *          that could. Ties break on the lowest neighbour index, so a flat shelf is walked the same
 *          everywhere rather than in whatever order the compiler chose.
 */
void walkDownhill(const lpl::procgen::ChunkParams &params, ReliefFoldResult &out)
{
    lpl::core::i32 x = 30;
    lpl::core::i32 z = 48;
    const lpl::math::Fixed32 start = lpl::procgen::sampleWorldHeight(params, x, z);

    for (lpl::core::u32 step = 0u; step < kWalkSteps; ++step)
    {
        lpl::math::Fixed32 here = lpl::procgen::sampleWorldHeight(params, x, z);
        lpl::core::i32 bestX = x;
        lpl::core::i32 bestZ = z;

        for (lpl::core::i32 neighbour = 0; neighbour < 8; ++neighbour)
        {
            const lpl::core::i32 dx = (neighbour % 3) - 1;
            const lpl::core::i32 dz = (neighbour / 3) - 1;

            if (dx == 0 && dz == 0)
                continue;

            const lpl::math::Fixed32 there = lpl::procgen::sampleWorldHeight(params, x + dx, z + dz);

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
        fold(out.walkSignature, static_cast<lpl::core::u32>(x));
        fold(out.walkSignature, static_cast<lpl::core::u32>(z));
        fold(out.walkSignature, static_cast<lpl::core::u32>(here.raw()));
    }
    out.descended = start.raw() - lpl::procgen::sampleWorldHeight(params, x, z).raw();
}

/**
 * @brief Runs the canonical world, standing on @p field, and folds it.
 *
 * @param field Survey, or nullptr for the world that ignores it.
 */
[[nodiscard]] ReliefFoldResult foldReliefWorld(const lpl::math::ReliefMosaic *field)
{
    ReliefFoldResult out{};
    const lpl::math::ReliefProjection projection = canonicalProjection();
    const lpl::procgen::ChunkParams params = canonicalParams(field);

    out.sampleSignature = kFnv1aOffsetBasis;
    out.heightSignature = kFnv1aOffsetBasis;
    out.walkSignature = kFnv1aOffsetBasis;
    out.coastSignature = kFnv1aOffsetBasis;
    foldSurvey(out);
    foldWindow(params, field, projection.projection.seaLevelUnits, out);
    walkDownhill(params, out);
    return out;
}

/**
 * @brief The canonical world, standing on the canonical survey.
 *
 * @details The survey is a mosaic of one field with every edge exposed: exactly what a lone field
 *          was before tiling existed, so these signatures not moving is the control on tiling.
 */
[[nodiscard]] ReliefFoldResult foldMeasuredWorld()
{
    lpl::math::ReliefField field{};
    lpl::math::ReliefMosaic mosaic{};

    buildSurvey();
    field.samples = gSamples;
    field.width = kSurveySide;
    field.height = kSurveySide;
    field.originCellX = 0;
    field.originCellZ = 0;
    field.projection = canonicalProjection();
    field.blendCells = kBlendCells;
    (void) mosaic.add(&field);
    return foldReliefWorld(&mosaic);
}

/**
 * @brief The same world with no survey behind it: the control.
 *
 * @details Every signature is stable on both targets whether or not the relief is ever read; only
 *          the comparison with the world that ignores it shows that it was.
 */
[[nodiscard]] ReliefFoldResult foldInventedWorld()
{
    buildSurvey();
    return foldReliefWorld(nullptr);
}

} // namespace

/**
 * @brief Gate P21 relief: a world standing on measured ground comes out the same on both targets.
 *
 * @details The signatures alone would agree just as well if the survey were never read, so the
 *          counts show that it was, and the world that ignores it must differ.
 */
LPL_TEST(world_stands_on_measured_ground)
{
    const ReliefFoldResult measured = foldMeasuredWorld();
    const ReliefFoldResult invented = foldInventedWorld();

    test.check(measured.measuredCells > 0u, "measured ground was stood on");
    test.check(measured.inventedCells > 0u, "invented ground was reached beyond it");
    test.check(measured.blendedCells > 0u, "and the border band was crossed");
    test.check(measured.seaCells > 0u &&
                   measured.seaCells < measured.measuredCells + measured.inventedCells + measured.blendedCells,
               "the coast is inside the world");

    test.check(measured.heightSignature != invented.heightSignature, "the ground differs from the invented world's");
    test.check(measured.coastSignature != invented.coastSignature, "the coastline differs");
    test.check(measured.walkSignature != invented.walkSignature, "and the body goes somewhere else");
    test.check(measured.sampleSignature == invented.sampleSignature, "but the survey itself is one survey");

    test.check(measured.walkSteps >= 8u, "the body walked more than a handful of steps");
    test.check(measured.descended > 0, "and ended lower than it started");

    test.measureHexadecimal("sample_signature", measured.sampleSignature);
    test.measureHexadecimal("height_signature", measured.heightSignature);
    test.measureHexadecimal("walk_signature", measured.walkSignature);
    test.measureHexadecimal("coast_signature", measured.coastSignature);
    test.measure("measured_cells", measured.measuredCells);
    test.measure("invented_cells", measured.inventedCells);
    test.measure("blended_cells", measured.blendedCells);
    test.measure("sea_cells", measured.seaCells);
    test.measure("walk_steps", measured.walkSteps);
    test.measure("descended", measured.descended);
    test.measureHexadecimal("invented_height_signature", invented.heightSignature);
    test.measureHexadecimal("invented_walk_signature", invented.walkSignature);
}
