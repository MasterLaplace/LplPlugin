#include <lpl/voxel/Raymarch.hpp>
#include <lpl/voxel/Residency.hpp>

#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

namespace {

int gChecks = 0;
int gFailures = 0;

void check(bool ok, const char *what)
{
    ++gChecks;
    if (!ok)
    {
        ++gFailures;
        std::printf("  FAIL  %s\n", what);
    }
}

void checkEq(long long got, long long want, const char *what)
{
    ++gChecks;
    if (got != want)
    {
        ++gFailures;
        std::printf("  FAIL  %s: got %lld, want %lld\n", what, got, want);
    }
}

constexpr lpl::core::u8 kMedium = 130u;
constexpr lpl::core::u8 kSheet = 200u;

/**
 * @brief Sheets perpendicular to volume axis X, one slab of `thick` every `pitch` samples.
 */
std::vector<lpl::core::u8> makeSheetBrick(lpl::core::u32 level, lpl::core::i64 brickX, lpl::core::u32 pitch,
                                          lpl::core::u32 thick)
{
    std::vector<lpl::core::u8> bytes(lpl::voxel::kBrickVoxels, kMedium);
    const lpl::core::i64 span = lpl::voxel::brickSpanInBaseSamples(level);
    const lpl::core::i64 originX = brickX * span;
    const lpl::core::i64 step = static_cast<lpl::core::i64>(1) << level;
    for (lpl::core::u32 lz = 0u; lz < lpl::voxel::kBrickEdge; ++lz)
    {
        for (lpl::core::u32 ly = 0u; ly < lpl::voxel::kBrickEdge; ++ly)
        {
            for (lpl::core::u32 lx = 0u; lx < lpl::voxel::kBrickEdge; ++lx)
            {
                const lpl::core::i64 baseX = originX + static_cast<lpl::core::i64>(lx) * step;
                const lpl::core::i64 phase =
                    ((baseX % static_cast<lpl::core::i64>(pitch)) + pitch) % static_cast<lpl::core::i64>(pitch);
                if (phase < static_cast<lpl::core::i64>(thick))
                    bytes[(static_cast<std::size_t>(lz) * lpl::voxel::kBrickEdge + ly) * lpl::voxel::kBrickEdge + lx] =
                        kSheet;
            }
        }
    }
    return bytes;
}

std::vector<lpl::core::u8> makeFlatBrick(lpl::core::u8 value)
{
    return std::vector<lpl::core::u8>(lpl::voxel::kBrickVoxels, value);
}

/**
 * @brief A brick that is medium everywhere except a slab of `thick` samples at its low Z face.
 */
std::vector<lpl::core::u8> makeEntrySlabBrick(lpl::core::u32 thick)
{
    std::vector<lpl::core::u8> bytes(lpl::voxel::kBrickVoxels, kMedium);
    for (lpl::core::u32 lz = 0u; lz < thick && lz < lpl::voxel::kBrickEdge; ++lz)
    {
        for (lpl::core::u32 ly = 0u; ly < lpl::voxel::kBrickEdge; ++ly)
        {
            for (lpl::core::u32 lx = 0u; lx < lpl::voxel::kBrickEdge; ++lx)
                bytes[(static_cast<std::size_t>(lz) * lpl::voxel::kBrickEdge + ly) * lpl::voxel::kBrickEdge + lx] =
                    kSheet;
        }
    }
    return bytes;
}

/**
 * @brief A dense ball in a medium: the only fixture here whose normals actually vary.
 */
std::vector<lpl::core::u8> makeBallBrick(lpl::core::u32 radius)
{
    std::vector<lpl::core::u8> bytes(lpl::voxel::kBrickVoxels, kMedium);
    const long long c = lpl::voxel::kBrickEdge / 2;
    const long long r2 = static_cast<long long>(radius) * radius;
    for (lpl::core::u32 lz = 0u; lz < lpl::voxel::kBrickEdge; ++lz)
    {
        for (lpl::core::u32 ly = 0u; ly < lpl::voxel::kBrickEdge; ++ly)
        {
            for (lpl::core::u32 lx = 0u; lx < lpl::voxel::kBrickEdge; ++lx)
            {
                const long long dz = static_cast<long long>(lz) - c;
                const long long dy = static_cast<long long>(ly) - c;
                const long long dx = static_cast<long long>(lx) - c;
                if (dz * dz + dy * dy + dx * dx <= r2)
                    bytes[(static_cast<std::size_t>(lz) * lpl::voxel::kBrickEdge + ly) * lpl::voxel::kBrickEdge + lx] =
                        kSheet;
            }
        }
    }
    return bytes;
}

lpl::voxel::BrickView viewOf(const std::vector<lpl::core::u8> &bytes, lpl::voxel::BrickKey key)
{
    lpl::voxel::BrickView v{};
    v.voxels = bytes.data();
    v.key = key;
    lpl::voxel::summarise(v);
    return v;
}

/** Field of view of the one-pixel frames below. Their single ray goes straight ahead whatever it is. */
constexpr lpl::core::f32 kPencil = 0.01f;

/**
 * @brief A camera looking down volume Z from (x, y, z), given in level-0 samples.
 */
lpl::voxel::Eye eyeAlongZ(lpl::core::f32 metresPerSample, lpl::core::f32 x, lpl::core::f32 y, lpl::core::f32 z,
                          lpl::core::f32 fieldOfView)
{
    lpl::voxel::Eye eye{};
    eye.position = {x * metresPerSample, y * metresPerSample, z * metresPerSample};
    eye.forward = {0.0f, 0.0f, 1.0f};
    eye.right = {1.0f, 0.0f, 0.0f};
    eye.up = {0.0f, 1.0f, 0.0f};
    eye.horizontalFieldOfView = fieldOfView;
    return eye;
}

/**
 * @brief Mean luminance of the non-background part of a frame, in [0,1].
 */
lpl::core::f32 meanLuminance(const std::vector<lpl::core::u32> &pixels, lpl::core::u32 background)
{
    double acc = 0.0;
    std::size_t n = 0;
    for (lpl::core::u32 p : pixels)
    {
        if (p == background)
            continue;
        acc += static_cast<double>(((p >> 16) & 0xFFu) + ((p >> 8) & 0xFFu) + (p & 0xFFu)) / (3.0 * 255.0);
        ++n;
    }
    return n == 0 ? 0.0f : static_cast<lpl::core::f32>(acc / static_cast<double>(n));
}

/**
 * @brief Red channel of a 0xAARRGGBB pixel, in [0,1].
 *
 * The ramp's red rises fastest with density.
 */
lpl::core::f32 redLevel(lpl::core::u32 pixel) { return static_cast<lpl::core::f32>((pixel >> 16) & 0xFFu) / 255.0f; }

/**
 * @brief Largest change between two consecutive values of @p readingAt(0 .. count-1).
 */
template <typename Reading> lpl::core::f32 worstAdjacentChange(int count, Reading readingAt)
{
    lpl::core::f32 worst = 0.0f;
    lpl::core::f32 previous = readingAt(0);
    for (int i = 1; i < count; ++i)
    {
        const lpl::core::f32 current = readingAt(i);
        const lpl::core::f32 change = current > previous ? current - previous : previous - current;
        if (change > worst)
            worst = change;
        previous = current;
    }
    return worst;
}

/**
 * @brief Whether @p key is the brick, at its own level, that holds the level-0 sample @p eye.
 */
bool isInEyeColumn(const lpl::voxel::BrickKey &key, const lpl::core::i64 eye[3])
{
    return lpl::voxel::brickIndexOfBase(eye[0], key.level) == key.z &&
           lpl::voxel::brickIndexOfBase(eye[1], key.level) == key.y &&
           lpl::voxel::brickIndexOfBase(eye[2], key.level) == key.x;
}

/**
 * @brief Bricks and indexing.
 *
 * @details A brick index is a floor division, not a truncation: the brick below the origin is -1, never 0.
 */
void checkBricksAndIndexing()
{
    const auto flat = makeFlatBrick(77u);
    lpl::voxel::BrickView v = viewOf(flat, lpl::voxel::BrickKey{0u, 0, 0, 0});
    checkEq(v.lowest, 77, "a flat brick reports its value as both bounds (low)");
    checkEq(v.highest, 77, "a flat brick reports its value as both bounds (high)");

    const auto sheets = makeSheetBrick(0u, 0, 32u, 4u);
    v = viewOf(sheets, lpl::voxel::BrickKey{0u, 0, 0, 0});
    checkEq(v.lowest, kMedium, "a sheet brick's floor is the medium");
    checkEq(v.highest, kSheet, "a sheet brick's ceiling is the sheet");

    checkEq(lpl::voxel::brickIndexOfBase(0, 0u), 0, "sample 0 is in brick 0");
    checkEq(lpl::voxel::brickIndexOfBase(127, 0u), 0, "the last sample of brick 0 is in brick 0");
    checkEq(lpl::voxel::brickIndexOfBase(128, 0u), 1, "sample 128 opens brick 1");
    checkEq(lpl::voxel::brickIndexOfBase(-1, 0u), -1, "the sample below the origin is in brick -1");
    checkEq(lpl::voxel::brickIndexOfBase(-128, 0u), -1, "brick -1 starts at -128");
    checkEq(lpl::voxel::brickIndexOfBase(-129, 0u), -2, "and brick -2 below it");
    checkEq(lpl::voxel::brickIndexOfBase(255, 1u), 0, "a level-1 brick spans 256 base samples");
    checkEq(lpl::voxel::brickIndexOfBase(256, 1u), 1, "and the next one starts at 256");
}

/**
 * @brief The mosaic: finest wins.
 *
 * @details The coarse brick is inserted FIRST on purpose: a mosaic that returned the first match would
 * answer with it, and the picture would depend on which download landed first.
 *
 * Outside the fine brick but inside the coarse one, the coarse one answers: that overlap
 * is what lets a fine brick be evicted without leaving a hole.
 */
void checkMosaicAnswersWithTheFinestBrick()
{
    const auto fine = makeFlatBrick(10u);
    const auto coarse = makeFlatBrick(20u);
    lpl::voxel::BrickMosaic m;
    check(m.insert(viewOf(coarse, lpl::voxel::BrickKey{3u, 0, 0, 0})), "the coarse brick is listed");
    check(m.insert(viewOf(fine, lpl::voxel::BrickKey{0u, 0, 0, 0})), "the fine brick is listed");
    checkEq(m.count(), 2, "both are resident");
    checkEq(m.coarsestLevel(), 3, "the mosaic knows its coarsest level");
    checkEq(m.finestLevel(), 0, "and its finest, which bounds every hop a ray takes");

    const lpl::voxel::BrickView *hit = m.find(4, 4, 4);
    check(hit != nullptr, "a covered point finds a brick");
    checkEq(hit == nullptr ? -1 : static_cast<long long>(hit->key.level), 0, "the FINEST brick answers, not the first");

    hit = m.find(500, 4, 4);
    check(hit != nullptr && hit->key.level == 3u, "outside the fine brick, the coarse one covers");

    check(m.remove(lpl::voxel::BrickKey{0u, 0, 0, 0}), "a listed brick can be unlisted");
    check(!m.contains(lpl::voxel::BrickKey{0u, 0, 0, 0}), "and is then absent");
    checkEq(m.finestLevel(), 3, "evicting the finest brick moves the finest level up");
    hit = m.find(4, 4, 4);
    check(hit != nullptr && hit->key.level == 3u, "eviction of the fine brick leaves the coarse one, not a hole");
    checkEq(m.count(), 1, "the count follows the removal");
}

/**
 * @brief The nearest sample, read from the brick find returns.
 *
 * @details 2^40 + 256 lands in brick index 2^32 + 1, which an i32 index holds as 1: the resident brick.
 */
void checkNearestSampleIsReadFromTheBrickFindReturns()
{
    std::vector<lpl::core::u8> risingAlongX(lpl::voxel::kBrickVoxels, 0u);
    for (lpl::core::u32 lx = 0u; lx < lpl::voxel::kBrickEdge; ++lx)
        risingAlongX[lx] = static_cast<lpl::core::u8>(100u + lx);
    lpl::voxel::BrickMosaic m;
    check(m.insert(viewOf(risingAlongX, lpl::voxel::BrickKey{1u, 0, 0, 1})),
          "a level-1 brick at x 256 to 511 is listed");

    checkEq(m.sampleAt(0, 0, 256).value_or(0u), 100, "its first cell answers at its origin");
    checkEq(m.sampleAt(1, 1, 261).value_or(0u), 102, "a level-1 cell spans two level-0 samples on every axis");
    check(!m.sampleAt(0, 0, 255).has_value(), "nothing answers where nothing is resident");
    check(!m.sampleAt(0, 0, (static_cast<lpl::core::i64>(1) << 40) + 256).has_value(),
          "nor where a coordinate is past what a brick index holds, rather than the brick it wraps onto");
}

/**
 * @brief Residency: nearest first.
 *
 * @details The first four entries are the eye's own column at each level: radius zero, every level.
 * That is what makes a truncated plan still cover everywhere. A budget smaller than the plan must
 * lose the far shell, never the near one.
 *
 * Nothing outside the subject is ever requested, and no key is asked for twice: a duplicate would
 * push a real brick past the budget.
 */
void checkResidencyPlansNearestFirst()
{
    lpl::voxel::VolumeGeometry geometry{};
    geometry.samples[0] = 2048;
    geometry.samples[1] = 2048;
    geometry.samples[2] = 2048;
    geometry.levels = 4u;
    geometry.voxelMicrometres = 7.91f;
    geometry.metresPerMicrometre = 11538.0f;
    check(geometry.valid(), "the fixture geometry is valid");

    lpl::voxel::ResidencyParams params{};
    params.finestLevel = 0u;
    params.coarsestLevel = 3u;
    params.ringRadius = 2;
    params.budget = 512u;

    const lpl::core::i64 eye[3]{1024, 1024, 1024};
    std::vector<lpl::voxel::BrickKey> plan(512);
    lpl::core::u32 n =
        lpl::voxel::planResidency(geometry, params, eye, plan.data(), static_cast<lpl::core::u32>(plan.size()));
    check(n > 0u, "a plan is produced");

    bool firstShellIsRadiusZero = true;
    for (lpl::core::u32 i = 0u; i < 4u && i < n; ++i)
        firstShellIsRadiusZero = firstShellIsRadiusZero && isInEyeColumn(plan[i], eye);
    check(firstShellIsRadiusZero, "the plan opens with the eye's own brick at every level");

    params.budget = 4u;
    const lpl::core::u32 tight =
        lpl::voxel::planResidency(geometry, params, eye, plan.data(), static_cast<lpl::core::u32>(plan.size()));
    checkEq(tight, 4, "a tight budget is respected exactly");
    bool tightIsAllNear = true;
    for (lpl::core::u32 i = 0u; i < tight; ++i)
        tightIsAllNear = tightIsAllNear && isInEyeColumn(plan[i], eye);
    check(tightIsAllNear, "a truncated plan keeps the ground under the eye and drops the horizon");

    const lpl::core::i64 corner[3]{0, 0, 0};
    params.budget = 512u;
    n = lpl::voxel::planResidency(geometry, params, corner, plan.data(), static_cast<lpl::core::u32>(plan.size()));
    bool allInside = true;
    for (lpl::core::u32 i = 0u; i < n; ++i)
    {
        const lpl::voxel::BrickKey &k = plan[i];
        if (k.z < 0 || k.y < 0 || k.x < 0 || k.z >= geometry.bricksAtLevel(0, k.level) ||
            k.y >= geometry.bricksAtLevel(1, k.level) || k.x >= geometry.bricksAtLevel(2, k.level))
            allInside = false;
    }
    check(allInside, "no request ever leaves the subject");

    bool unique = true;
    for (lpl::core::u32 i = 0u; i < n && unique; ++i)
    {
        for (lpl::core::u32 j = i + 1u; j < n; ++j)
        {
            if (plan[i] == plan[j])
            {
                unique = false;
                break;
            }
        }
    }
    check(unique, "the plan never asks for the same brick twice");
}

/**
 * @brief The transfer curve.
 *
 * @details A coarser level's window is squeezed toward the mean: the samples moved there, so the window
 * has to follow. Without this the level-4 curve would paint the entire histogram.
 *
 * The level-0 window is [140, 200]. Squeezed toward 147 by 0.485 at level 4 it opens at
 * 143.6, and by 0.383 at level 5 it closes at 167.3. Recovered from the table rather than
 * stored, it had shrunk to [141, 199] first: it opened at 144.1 and closed at 166.9, one
 * density off at each end.
 */
void checkTransferCurveFollowsTheCollapsingSpread(const lpl::voxel::DensityProfile &profile,
                                                  const lpl::voxel::TransferFunction &base)
{
    check(base.alpha[100] == 0.0f, "below the floor, nothing is painted");
    check(base.alpha[255] > base.alpha[170], "the ramp rises toward the sheet");
    check(base.alpha[170] > base.alpha[150], "and rises through the middle of the band");
    check(base.firstVisible > profile.floorSample && base.firstVisible < profile.sheetSample,
          "the first visible density sits inside the band");
    check(!base.reachesVisible(100u), "a brick that never reaches the band is skippable");
    check(base.reachesVisible(255u), "a brick that reaches it is not");

    const lpl::voxel::TransferFunction coarse = base.forLevel(profile, 4u);
    check(coarse.firstVisible > base.firstVisible,
          "a coarser level's window starts higher, following the collapsing spread");
    check(coarse.firstVisible < static_cast<lpl::core::u8>(profile.mean + 30.0f),
          "and stays near the mean rather than running away");
    check(coarse.alpha[143] == 0.0f && coarse.alpha[144] > 0.0f, "the rescaled window opens where it should");
    const lpl::voxel::TransferFunction coarser = base.forLevel(profile, 5u);
    check(coarser.alpha[167] < coarser.alpha[168], "and closes where it should");
}

/**
 * @brief The march: every pixel is traced, and a stack of sheets comes out bright.
 *
 * @details Not saturation: whether a four-sample sheet goes fully opaque is a fact about the peak
 * opacity somebody chose, and boundary modulation deliberately lets a ray through the flat
 * middle of one. What the fixture can actually claim is that a stack of sheets is bright.
 */
void checkMarchTracesEveryPixelAndPaintsTheSheets(
    const lpl::voxel::BrickMosaic &mosaic, const lpl::voxel::VolumeGeometry &geometry,
    const lpl::voxel::DensityProfile &profile, const lpl::voxel::TransferFunction &base, const lpl::voxel::Eye &eye,
    const lpl::voxel::MarchParams &params, std::vector<lpl::core::u32> &frame, lpl::core::u32 kW, lpl::core::u32 kH)
{
    const lpl::voxel::MarchReport r =
        lpl::voxel::march(mosaic, geometry, profile, base, eye, params, frame.data(), kW, kH, 0u, kH);
    checkEq(static_cast<long long>(r.rays), static_cast<long long>(kW) * kH, "every pixel is traced");
    check(r.steps > 0, "the ray actually entered the volume");
    check(meanLuminance(frame, params.background) > 0.4f, "a stack of sheets comes out bright");

    std::size_t painted = 0;
    for (lpl::core::u32 p : frame)
    {
        if (p != params.background)
            ++painted;
    }
    check(painted > frame.size() / 4, "most of the frame is the subject, not the background");
}

/**
 * @brief Looking away must produce nothing.
 *
 * @details Without this, every check above is also satisfied by a renderer that paints the same
 * picture regardless of where the camera points.
 */
void checkLookingAwayProducesNothing(const lpl::voxel::BrickMosaic &mosaic, const lpl::voxel::VolumeGeometry &geometry,
                                     const lpl::voxel::DensityProfile &profile,
                                     const lpl::voxel::TransferFunction &base, const lpl::voxel::Eye &eye,
                                     const lpl::voxel::MarchParams &params, const std::vector<lpl::core::u32> &frame,
                                     lpl::core::u32 kW, lpl::core::u32 kH)
{
    lpl::voxel::Eye away = eye;
    away.forward = {-1.0f, 0.0f, 0.0f};
    away.right = {0.0f, 0.0f, -1.0f};
    std::vector<lpl::core::u32> blank(frame.size(), 0u);
    const lpl::voxel::MarchReport r =
        lpl::voxel::march(mosaic, geometry, profile, base, away, params, blank.data(), kW, kH, 0u, kH);
    checkEq(static_cast<long long>(r.escaped), static_cast<long long>(r.rays), "a camera pointed away hits nothing");
    checkEq(static_cast<long long>(r.steps), 0, "and takes no samples at all");
    bool allBackground = true;
    for (lpl::core::u32 p : blank)
    {
        if (p != params.background)
            allBackground = false;
    }
    check(allBackground, "the frame is entirely background");
}

/**
 * @brief Empty-space skipping.
 *
 * @details A brick whose whole range is below the band must cost one comparison, not two million
 * reads. The step count is the only thing that can tell the difference.
 */
void checkEmptyBrickIsSkippedWhole(const lpl::voxel::VolumeGeometry &geometry,
                                   const lpl::voxel::DensityProfile &profile, const lpl::voxel::TransferFunction &base,
                                   const lpl::voxel::Eye &eye, const lpl::voxel::MarchParams &params,
                                   const std::vector<lpl::core::u32> &frame, lpl::core::u32 kW, lpl::core::u32 kH)
{
    const auto dark = makeFlatBrick(90u);
    lpl::voxel::BrickMosaic dim;
    dim.insert(viewOf(dark, lpl::voxel::BrickKey{0u, 0, 0, 0}));
    std::vector<lpl::core::u32> blank(frame.size(), 0u);
    const lpl::voxel::MarchReport r =
        lpl::voxel::march(dim, geometry, profile, base, eye, params, blank.data(), kW, kH, 0u, kH);
    check(r.skippedBricks > 0, "an all-medium brick is skipped whole");
    check(r.steps < r.rays * 4, "and skipping means a handful of samples per ray, not hundreds");
}

/**
 * @brief The occupancy grid: leaving the medium in one comparison.
 *
 * @details Ninety-eight per cent of the samples a real frame takes paint nothing. The whole-brick
 * summary cannot skip them because the brick DOES contain sheets somewhere; a cell can. The
 * fixture is medium everywhere except a slab at the far end: the near half of every ray is nothing.
 *
 * With nearest samples the skip changes nothing at all. Trilinear points lean across a cell's face
 * (MarchParams::skipEmptyCells), so what is checked there is the size of the difference.
 *
 * Ten of 255 is what was measured, and twelve leaves room for a change of rounding alone. It
 * buys a 3.4x cut in samples; whether that trade is acceptable is the caller's call, which
 * is why the skip can be switched off and why the count is printed beside every frame.
 */
void checkOccupancyGridLeavesTheMediumInOneComparison(const lpl::voxel::VolumeGeometry &geometry,
                                                      const lpl::voxel::DensityProfile &profile,
                                                      const lpl::voxel::TransferFunction &base,
                                                      const lpl::voxel::MarchParams &params,
                                                      const std::vector<lpl::core::u32> &frame, lpl::core::u32 kW,
                                                      lpl::core::u32 kH)
{
    std::vector<lpl::core::u8> half(lpl::voxel::kBrickVoxels, kMedium);
    for (lpl::core::u32 lz = 96u; lz < lpl::voxel::kBrickEdge; ++lz)
        for (lpl::core::u32 ly = 0u; ly < lpl::voxel::kBrickEdge; ++ly)
            for (lpl::core::u32 lx = 0u; lx < lpl::voxel::kBrickEdge; ++lx)
                half[(static_cast<std::size_t>(lz) * lpl::voxel::kBrickEdge + ly) * lpl::voxel::kBrickEdge + lx] =
                    kSheet;

    lpl::voxel::BrickView plain{};
    plain.voxels = half.data();
    plain.key = lpl::voxel::BrickKey{0u, 0, 0, 0};
    lpl::voxel::summarise(plain);

    std::vector<lpl::core::u8> cells(lpl::voxel::kOccupancyBytes);
    lpl::voxel::BrickView celled{};
    celled.voxels = half.data();
    celled.key = lpl::voxel::BrickKey{0u, 0, 0, 0};
    lpl::voxel::summariseCells(celled, cells.data());
    check(celled.occupancy != nullptr, "the occupancy grid is filled");
    checkEq(celled.cellHighest(0u, 0u, 0u), kMedium, "a cell in the medium reports the medium");
    checkEq(celled.cellHighest(120u, 0u, 0u), kSheet, "and a cell in the slab reports the slab");

    lpl::voxel::VolumeGeometry one = geometry;
    one.samples[0] = 128;
    const lpl::voxel::Eye wide =
        eyeAlongZ(one.metresPerSample(), 64.0f, 64.0f, -4.0f, lpl::voxel::Eye{}.horizontalFieldOfView);

    std::vector<lpl::core::u32> a(frame.size(), 0u);
    lpl::voxel::BrickMosaic without;
    without.insert(plain);
    const lpl::voxel::MarchReport slow =
        lpl::voxel::march(without, one, profile, base, wide, params, a.data(), kW, kH, 0u, kH);

    std::vector<lpl::core::u32> b(frame.size(), 0u);
    lpl::voxel::BrickMosaic with;
    with.insert(celled);
    const lpl::voxel::MarchReport fast =
        lpl::voxel::march(with, one, profile, base, wide, params, b.data(), kW, kH, 0u, kH);

    std::printf("  occupancy: %llu samples without, %llu with, %llu cells skipped\n",
                static_cast<unsigned long long>(slow.steps), static_cast<unsigned long long>(fast.steps),
                static_cast<unsigned long long>(fast.skippedCells));
    check(fast.skippedCells > 0, "cells in the medium are skipped");
    check(fast.steps < slow.steps * 3u / 4u, "and skipping them costs materially fewer samples");
    lpl::voxel::MarchParams nearest = params;
    nearest.trilinear = false;
    std::vector<lpl::core::u32> nearestWithout(frame.size(), 0u);
    std::vector<lpl::core::u32> nearestWith(frame.size(), 0u);
    lpl::voxel::march(without, one, profile, base, wide, nearest, nearestWithout.data(), kW, kH, 0u, kH);
    lpl::voxel::march(with, one, profile, base, wide, nearest, nearestWith.data(), kW, kH, 0u, kH);
    checkEq(lpl::voxel::foldFrame(nearestWith.data(), static_cast<lpl::core::u32>(nearestWith.size())),
            lpl::voxel::foldFrame(nearestWithout.data(), static_cast<lpl::core::u32>(nearestWithout.size())),
            "with nearest samples, skipping cells leaves the frame identical");

    lpl::core::u32 worst = 0u;
    std::size_t differing = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        for (lpl::core::u32 shift = 0u; shift < 24u; shift += 8u)
        {
            const int x = static_cast<int>((a[i] >> shift) & 0xFFu);
            const int y = static_cast<int>((b[i] >> shift) & 0xFFu);
            const lpl::core::u32 d = static_cast<lpl::core::u32>(x > y ? x - y : y - x);
            if (d > worst)
                worst = d;
            if (d != 0u)
                ++differing;
        }
    }
    std::printf("  occupancy: worst channel difference %u of 255, %zu channels of %zu differ\n", worst, differing,
                a.size() * 3u);
    check(worst <= 12u, "and the picture differs by at most a few per cent of a channel");
    check(fast.steps * 3u < slow.steps, "for a threefold cut in samples");
}

/**
 * @brief Missing bricks are counted rather than silently painted.
 */
void checkMissingBricksAreCounted(const lpl::voxel::VolumeGeometry &geometry, const lpl::voxel::DensityProfile &profile,
                                  const lpl::voxel::TransferFunction &base, const lpl::voxel::Eye &eye,
                                  const lpl::voxel::MarchParams &params, const std::vector<lpl::core::u32> &frame,
                                  lpl::core::u32 kW, lpl::core::u32 kH)
{
    lpl::voxel::BrickMosaic empty;
    std::vector<lpl::core::u32> blank(frame.size(), 0u);
    const lpl::voxel::MarchReport r =
        lpl::voxel::march(empty, geometry, profile, base, eye, params, blank.data(), kW, kH, 0u, kH);
    check(r.missingBricks > 0, "a mosaic with nothing in it reports the holes");
    check(r.saturated == 0, "and paints nothing");
}

/**
 * @brief The step-length correction.
 *
 * @details Opacity is declared per level-0 sample of path, so a longer step has to absorb what it
 * stepped over; without that, the same matter looks thinner purely because the camera moved back,
 * which reads as the object changing at every level boundary.
 *
 * @warning The fixture matters more than the assertion here. A first version marched the sheet
 * field, where four-sample sheets at high opacity saturate either way -- so removing the
 * correction entirely left the drift at 0.2 %, and the check could not fail. It takes a FAINT,
 * UNIFORM medium, far from saturation, for the difference to be visible at all: there,
 * accumulated opacity is nearly proportional to the number of samples taken, and quartering
 * the samples quarters the picture.
 *
 * Shading and boundary opacity are off HERE: they exist to make a uniform medium nearly invisible,
 * which is exactly the property this fixture needs to keep in order to measure the step-length
 * correction at all.
 */
void checkStepLengthCorrectionKeepsAFaintMediumSteady(const lpl::voxel::VolumeGeometry &geometry,
                                                      const lpl::voxel::DensityProfile &profile,
                                                      const lpl::voxel::Eye &eye, const lpl::voxel::MarchParams &params,
                                                      const std::vector<lpl::core::u32> &frame, lpl::core::u32 kW,
                                                      lpl::core::u32 kH)
{
    const auto haze = makeFlatBrick(static_cast<lpl::core::u8>((profile.floorSample + profile.sheetSample) / 2u));
    lpl::voxel::BrickMosaic fog;
    fog.insert(viewOf(haze, lpl::voxel::BrickKey{0u, 0, 0, 0}));
    const lpl::voxel::TransferFunction faint = lpl::voxel::rampTransfer(profile.floorSample, profile.sheetSample,
                                                                        lpl::voxel::alphaForOpaqueAfter(2000.0f, 0.9f));

    std::vector<lpl::core::u32> fine(frame.size(), 0u);
    std::vector<lpl::core::u32> coarse(frame.size(), 0u);
    lpl::voxel::MarchParams p1 = params;
    p1.stepSamples = 1.0f;
    p1.shading = 0.0f;
    p1.boundaryOpacity = 0.0f;
    lpl::voxel::MarchParams p4 = p1;
    p4.stepSamples = 4.0f;
    lpl::voxel::march(fog, geometry, profile, faint, eye, p1, fine.data(), kW, kH, 0u, kH);
    lpl::voxel::march(fog, geometry, profile, faint, eye, p4, coarse.data(), kW, kH, 0u, kH);
    const lpl::core::f32 a = meanLuminance(fine, params.background);
    const lpl::core::f32 b = meanLuminance(coarse, params.background);
    const lpl::core::f32 drift = (a > b ? a - b : b - a) / (a > 0.0f ? a : 1.0f);
    std::printf("  faint haze, step 1 -> %.4f, step 4 -> %.4f, drift %.1f %%\n", static_cast<double>(a),
                static_cast<double>(b), static_cast<double>(drift * 100.0f));
    check(a > 0.05f, "the faint fixture actually accumulates something to compare");
    check(drift < 0.05f, "quadrupling the step barely moves a faint medium");
}

/**
 * @brief The same comparison on the sheet field, which is where a viewer actually lives.
 */
void checkDoublingTheStepBarelyMovesTheSheets(const lpl::voxel::BrickMosaic &mosaic,
                                              const lpl::voxel::VolumeGeometry &geometry,
                                              const lpl::voxel::DensityProfile &profile,
                                              const lpl::voxel::TransferFunction &base, const lpl::voxel::Eye &eye,
                                              const lpl::voxel::MarchParams &params,
                                              const std::vector<lpl::core::u32> &frame, lpl::core::u32 kW,
                                              lpl::core::u32 kH)
{
    std::vector<lpl::core::u32> fine(frame.size(), 0u);
    std::vector<lpl::core::u32> coarse(frame.size(), 0u);
    lpl::voxel::MarchParams p1 = params;
    p1.stepSamples = 1.0f;
    lpl::voxel::MarchParams p2 = params;
    p2.stepSamples = 2.0f;
    lpl::voxel::march(mosaic, geometry, profile, base, eye, p1, fine.data(), kW, kH, 0u, kH);
    lpl::voxel::march(mosaic, geometry, profile, base, eye, p2, coarse.data(), kW, kH, 0u, kH);
    const lpl::core::f32 a = meanLuminance(fine, params.background);
    const lpl::core::f32 b = meanLuminance(coarse, params.background);
    const lpl::core::f32 drift = (a > b ? a - b : b - a) / (a > 0.0f ? a : 1.0f);
    std::printf("  sheets, step 1 -> %.4f, step 2 -> %.4f, drift %.1f %%\n", static_cast<double>(a),
                static_cast<double>(b), static_cast<double>(drift * 100.0f));
    check(drift < 0.08f, "doubling the step barely moves the picture");
}

/**
 * @brief A rendered frame is stable across two identical calls.
 *
 * @details Not a parity gate -- it is float, and this project reserves that word for folds that
 * survive a change of target.
 */
void checkSamePoseRendersTheSameFrame(const lpl::voxel::BrickMosaic &mosaic, const lpl::voxel::VolumeGeometry &geometry,
                                      const lpl::voxel::DensityProfile &profile,
                                      const lpl::voxel::TransferFunction &base, const lpl::voxel::Eye &eye,
                                      const lpl::voxel::MarchParams &params, const std::vector<lpl::core::u32> &frame,
                                      lpl::core::u32 kW, lpl::core::u32 kH)
{
    std::vector<lpl::core::u32> a(frame.size(), 0u);
    std::vector<lpl::core::u32> b(frame.size(), 0u);
    lpl::voxel::march(mosaic, geometry, profile, base, eye, params, a.data(), kW, kH, 0u, kH);
    lpl::voxel::march(mosaic, geometry, profile, base, eye, params, b.data(), kW, kH, 0u, kH);
    const lpl::core::u32 fa = lpl::voxel::foldFrame(a.data(), static_cast<lpl::core::u32>(a.size()));
    const lpl::core::u32 fb = lpl::voxel::foldFrame(b.data(), static_cast<lpl::core::u32>(b.size()));
    checkEq(fa, fb, "the same pose renders the same frame");
    std::printf("  frame signature 0x%08X\n", fa);
}

/**
 * @brief Rendering by bands must equal rendering in one pass, or a thread pool changes the picture.
 */
void checkBandedRenderingEqualsOnePass(const lpl::voxel::BrickMosaic &mosaic,
                                       const lpl::voxel::VolumeGeometry &geometry,
                                       const lpl::voxel::DensityProfile &profile,
                                       const lpl::voxel::TransferFunction &base, const lpl::voxel::Eye &eye,
                                       const lpl::voxel::MarchParams &params, const std::vector<lpl::core::u32> &frame,
                                       lpl::core::u32 kW, lpl::core::u32 kH)
{
    std::vector<lpl::core::u32> whole(frame.size(), 0u);
    std::vector<lpl::core::u32> banded(frame.size(), 0u);
    lpl::voxel::march(mosaic, geometry, profile, base, eye, params, whole.data(), kW, kH, 0u, kH);
    for (lpl::core::u32 row = 0u; row < kH; row += 7u)
        lpl::voxel::march(mosaic, geometry, profile, base, eye, params, banded.data(), kW, kH, row, 7u);
    checkEq(lpl::voxel::foldFrame(banded.data(), static_cast<lpl::core::u32>(banded.size())),
            lpl::voxel::foldFrame(whole.data(), static_cast<lpl::core::u32>(whole.size())),
            "splitting a frame into bands renders the same frame");
}

/**
 * @brief Shading: a lit surface, not a bright silhouette.
 *
 * @details What shading does is put a gradient ACROSS an object: the part facing the eye is bright and
 * the limb falls away. So the measurement is centre-versus-limb inside the silhouette, not
 * contrast over the frame. Only shading varies between the two frames; boundary opacity stays on
 * in both.
 *
 * @warning Two earlier versions of this check measured the wrong thing and both reported the
 * wrong SIGN. Mean luminance points down, because shading darkens oblique surfaces. Variance
 * points down too, because an unshaded object is a flat disc against a dark ground -- maximum
 * contrast, and no structure whatsoever. A metric that moves in the wrong direction is worse
 * than no metric: it argues confidently for removing the feature.
 */
void checkShadingLightsASurfaceNotASilhouette(const lpl::voxel::VolumeGeometry &geometry,
                                              const lpl::voxel::DensityProfile &profile,
                                              const lpl::voxel::TransferFunction &base, const lpl::voxel::Eye &eye,
                                              const lpl::voxel::MarchParams &params,
                                              const std::vector<lpl::core::u32> &frame, lpl::core::u32 kW,
                                              lpl::core::u32 kH)
{
    const auto ball = makeBallBrick(40u);
    lpl::voxel::BrickMosaic round;
    round.insert(viewOf(ball, lpl::voxel::BrickKey{0u, 0, 0, 0}));

    lpl::voxel::MarchParams lit = params;
    lpl::voxel::MarchParams plain = params;
    plain.shading = 0.0f;

    auto centreOverLimb = [&](const lpl::voxel::MarchParams &p) {
        std::vector<lpl::core::u32> px(frame.size(), 0u);
        lpl::voxel::march(round, geometry, profile, base, eye, p, px.data(), kW, kH, 0u, kH);
        double centre = 0.0;
        double limb = 0.0;
        std::size_t nc = 0;
        std::size_t nl = 0;
        for (lpl::core::u32 y = 0u; y < kH; ++y)
        {
            for (lpl::core::u32 x = 0u; x < kW; ++x)
            {
                const lpl::core::u32 v = px[static_cast<std::size_t>(y) * kW + x];
                if (v == p.background)
                    continue;
                const double l =
                    static_cast<double>(((v >> 16) & 0xFFu) + ((v >> 8) & 0xFFu) + (v & 0xFFu)) / (3.0 * 255.0);
                const double dx = (static_cast<double>(x) - kW * 0.5) / (kW * 0.5);
                const double dy = (static_cast<double>(y) - kH * 0.5) / (kH * 0.5);
                const double r = dx * dx + dy * dy;
                if (r < 0.09)
                {
                    centre += l;
                    ++nc;
                }
                else if (r > 0.36)
                {
                    limb += l;
                    ++nl;
                }
            }
        }
        if (nc == 0 || nl == 0)
            return 0.0;
        return (centre / static_cast<double>(nc)) / ((limb / static_cast<double>(nl)) + 1e-9);
    };

    const double litRatio = centreOverLimb(lit);
    const double plainRatio = centreOverLimb(plain);
    std::printf("  ball, centre over limb: shaded %.3f, unshaded %.3f\n", litRatio, plainRatio);
    check(litRatio > 1.15, "a shaded ball is brighter where it faces the eye");
    check(litRatio > plainRatio * 1.10, "and flatter without shading");
}

/**
 * @brief Boundary opacity buys DEPTH, and depth is measured in steps rather than in brightness.
 *
 * @details A ray crossing homogeneous matter has to reach the first real surface instead of drowning
 * in the medium on the way there.
 */
void checkBoundaryOpacityBuysDepth(const lpl::voxel::VolumeGeometry &geometry,
                                   const lpl::voxel::DensityProfile &profile, const lpl::voxel::TransferFunction &base,
                                   const lpl::voxel::Eye &eye, const lpl::voxel::MarchParams &params,
                                   const std::vector<lpl::core::u32> &frame, lpl::core::u32 kW, lpl::core::u32 kH)
{
    const auto uniform = makeFlatBrick(kSheet);
    lpl::voxel::BrickMosaic solid;
    solid.insert(viewOf(uniform, lpl::voxel::BrickKey{0u, 0, 0, 0}));

    std::vector<lpl::core::u32> a(frame.size(), 0u);
    const lpl::voxel::MarchReport withIt =
        lpl::voxel::march(solid, geometry, profile, base, eye, params, a.data(), kW, kH, 0u, kH);

    lpl::voxel::MarchParams none = params;
    none.boundaryOpacity = 0.0f;
    const lpl::voxel::MarchReport without =
        lpl::voxel::march(solid, geometry, profile, base, eye, none, a.data(), kW, kH, 0u, kH);

    std::printf("  uniform matter: %llu steps with boundary opacity, %llu without\n",
                static_cast<unsigned long long>(withIt.steps), static_cast<unsigned long long>(without.steps));
    check(withIt.steps > without.steps * 2, "homogeneous matter lets a ray through instead of stopping it");
}

/**
 * @brief Skipping a brick must leave the ray AT ITS EXIT, not a span further on.
 *
 * @details samplesToFinestBoxExit in Raymarch.cpp says what a blind span jump looked like. The
 * corridor is an empty brick, skippable since nothing in it can paint, then a brick visible only in
 * its first 8 samples: two bricks deep along Z. The ray enters the empty brick well past its face:
 * a blind span jump from there overshoots the slab entirely.
 */
void checkSkippedBrickLeavesTheRayAtItsExit(const lpl::voxel::VolumeGeometry &geometry,
                                            const lpl::voxel::DensityProfile &profile,
                                            const lpl::voxel::TransferFunction &base,
                                            const lpl::voxel::MarchParams &params)
{
    const auto empty = makeFlatBrick(kMedium);
    const auto slab = makeEntrySlabBrick(8u);
    lpl::voxel::BrickMosaic corridor;
    corridor.insert(viewOf(empty, lpl::voxel::BrickKey{0u, 0, 0, 0}));
    corridor.insert(viewOf(slab, lpl::voxel::BrickKey{0u, 1, 0, 0}));

    lpl::voxel::VolumeGeometry two = geometry;
    two.samples[0] = 256;

    const lpl::voxel::Eye pencil = eyeAlongZ(two.metresPerSample(), 64.0f, 64.0f, 50.0f, kPencil);
    lpl::core::u32 pixel = 0u;
    const lpl::voxel::MarchReport r =
        lpl::voxel::march(corridor, two, profile, base, pencil, params, &pixel, 1u, 1u, 0u, 1u);
    std::printf("  skip-then-hit: %llu bricks skipped, red %.3f\n", static_cast<unsigned long long>(r.skippedBricks),
                static_cast<double>(redLevel(pixel)));
    check(r.skippedBricks >= 1u, "the empty brick really is skipped");
    check(redLevel(pixel) > 0.4f, "and the slab just past it is still hit");
}

/**
 * @brief No skip ever leaps over a finer resident brick.
 *
 * @details Levels overlap, so the brick answering at a point can be coarser than one a few samples
 * further on. A hop sized by the answering brick, or by the coarsest resident level, lands past
 * that finer brick and it vanishes from the picture: here, a bright brick that should stop the
 * ray outright.
 *
 * The same hops are then taken from a ray that sits exactly on the low face of its box and leaves
 * through it: the exit is right there, so the hop is one step, never a whole box.
 */
void checkNoSkipLeapsOverAFinerResidentBrick(const lpl::voxel::VolumeGeometry &geometry,
                                             const lpl::voxel::DensityProfile &profile,
                                             const lpl::voxel::TransferFunction &base,
                                             const lpl::voxel::MarchParams &params)
{
    lpl::voxel::VolumeGeometry deep = geometry;
    deep.samples[0] = 2048;
    deep.samples[1] = 2048;
    deep.samples[2] = 2048;
    deep.levels = 5u;
    const lpl::voxel::Eye pencil = eyeAlongZ(deep.metresPerSample(), 64.0f, 64.0f, -10.0f, kPencil);
    const auto saturatedRays = [&](const lpl::voxel::BrickMosaic &resident) {
        lpl::core::u32 pixel = 0u;
        return static_cast<long long>(
            lpl::voxel::march(resident, deep, profile, base, pencil, params, &pixel, 1u, 1u, 0u, 1u).saturated);
    };

    const auto bright = makeFlatBrick(kSheet);
    const auto dark = makeFlatBrick(100u);

    lpl::voxel::BrickMosaic insideDarkBrick;
    insideDarkBrick.insert(viewOf(dark, lpl::voxel::BrickKey{2u, 0, 0, 0}));
    insideDarkBrick.insert(viewOf(bright, lpl::voxel::BrickKey{0u, 2, 0, 0}));
    checkEq(saturatedRays(insideDarkBrick), 1, "skipping a dark coarse brick stops at the fine brick inside it");

    lpl::voxel::BrickMosaic pastAHole;
    pastAHole.insert(viewOf(bright, lpl::voxel::BrickKey{0u, 1, 0, 0}));
    pastAHole.insert(viewOf(bright, lpl::voxel::BrickKey{3u, 1, 1, 1}));
    checkEq(saturatedRays(pastAHole), 1, "crossing a hole stops at the fine brick beyond it");

    std::vector<lpl::core::u8> darkButVisibleElsewhere = dark;
    for (lpl::core::u32 lz = 0u; lz < lpl::voxel::kBrickEdge; ++lz)
        for (lpl::core::u32 ly = 0u; ly < lpl::voxel::kBrickEdge; ++ly)
            for (lpl::core::u32 lx = 120u; lx < lpl::voxel::kBrickEdge; ++lx)
                darkButVisibleElsewhere[(static_cast<std::size_t>(lz) * lpl::voxel::kBrickEdge + ly) *
                                            lpl::voxel::kBrickEdge +
                                        lx] = kSheet;
    std::vector<lpl::core::u8> cells(lpl::voxel::kOccupancyBytes);
    lpl::voxel::BrickView celled = viewOf(darkButVisibleElsewhere, lpl::voxel::BrickKey{4u, 0, 0, 0});
    lpl::voxel::summariseCells(celled, cells.data());
    lpl::voxel::BrickMosaic insideDarkCell;
    insideDarkCell.insert(celled);
    insideDarkCell.insert(viewOf(bright, lpl::voxel::BrickKey{0u, 1, 0, 0}));
    checkEq(saturatedRays(insideDarkCell), 1, "skipping a dark coarse cell stops at the fine brick inside it");

    lpl::voxel::VolumeGeometry exact = geometry;
    exact.samples[0] = 512;
    exact.samples[1] = 512;
    exact.samples[2] = 512;
    exact.levels = 3u;
    exact.voxelMicrometres = 1.0f;
    exact.metresPerMicrometre = 1.0e6f;
    check(exact.metresPerSample() == 1.0f, "one sample is exactly one metre, so a brick face is an exact position");
    lpl::voxel::Eye down = eyeAlongZ(1.0f, 64.0f, 64.0f, 256.0f, kPencil);
    down.forward = {0.0f, 0.0f, -1.0f};
    down.right = {-1.0f, 0.0f, 0.0f};
    const auto saturatedGoingDown = [&](const lpl::voxel::BrickMosaic &resident) {
        lpl::core::u32 pixel = 0u;
        return static_cast<long long>(
            lpl::voxel::march(resident, exact, profile, base, down, params, &pixel, 1u, 1u, 0u, 1u).saturated);
    };

    lpl::voxel::BrickMosaic belowAHole;
    belowAHole.insert(viewOf(bright, lpl::voxel::BrickKey{0u, 1, 0, 0}));
    checkEq(saturatedGoingDown(belowAHole), 1, "leaving a hole through the face it sits on stops right below it");

    lpl::voxel::BrickMosaic belowADarkBrick;
    belowADarkBrick.insert(viewOf(dark, lpl::voxel::BrickKey{2u, 0, 0, 0}));
    belowADarkBrick.insert(viewOf(bright, lpl::voxel::BrickKey{0u, 1, 0, 0}));
    checkEq(saturatedGoingDown(belowADarkBrick), 1, "and so does leaving a dark coarse brick through that face");
}

/**
 * @brief The brick face must not be visible.
 *
 * @details crossBrickSample in Raymarch.cpp says what the one-sample seam of a clamped cell looked
 * like. Two bricks side by side along Z with a smooth ramp running through both, so a correct
 * sample crossing the face sees a straight line and a clamped one sees a step. The ramp is
 * monotone across BOTH bricks. A first version used a sawtooth, which put a real discontinuity at
 * exactly the face being tested -- the fixture would have failed a perfect sampler.
 *
 * Each march takes the FIRST sample only: this is a probe of the field, not a walk. The field is
 * sampled either side of the face by putting the eye at successive Z offsets, from 120 to 136
 * across the face at 128, and reading the very first thing each ray paints.
 */
void checkBrickFaceIsNotVisible(const lpl::voxel::VolumeGeometry &geometry, const lpl::voxel::DensityProfile &profile,
                                const lpl::voxel::TransferFunction &base, const lpl::voxel::MarchParams &params)
{
    auto ramp = [](lpl::core::i64 brickZ) {
        std::vector<lpl::core::u8> bytes(lpl::voxel::kBrickVoxels, kMedium);
        for (lpl::core::u32 lz = 0u; lz < lpl::voxel::kBrickEdge; ++lz)
        {
            const lpl::core::i64 z =
                brickZ * static_cast<lpl::core::i64>(lpl::voxel::kBrickEdge) + static_cast<lpl::core::i64>(lz);
            const lpl::core::u8 v = static_cast<lpl::core::u8>(140 + z / 4);
            for (lpl::core::u32 ly = 0u; ly < lpl::voxel::kBrickEdge; ++ly)
                for (lpl::core::u32 lx = 0u; lx < lpl::voxel::kBrickEdge; ++lx)
                    bytes[(static_cast<std::size_t>(lz) * lpl::voxel::kBrickEdge + ly) * lpl::voxel::kBrickEdge + lx] =
                        v;
        }
        return bytes;
    };
    const auto low = ramp(0);
    const auto high = ramp(1);
    lpl::voxel::BrickMosaic pair;
    pair.insert(viewOf(low, lpl::voxel::BrickKey{0u, 0, 0, 0}));
    pair.insert(viewOf(high, lpl::voxel::BrickKey{0u, 1, 0, 0}));

    lpl::voxel::VolumeGeometry two = geometry;
    two.samples[0] = 256;

    lpl::voxel::MarchParams once = params;
    once.shading = 0.0f;
    once.boundaryOpacity = 0.0f;
    once.stepSamples = 0.5f;
    once.maxSteps = 1u;

    const lpl::core::f32 worst = worstAdjacentChange(40, [&](int i) {
        const lpl::core::f32 z = 120.0f + static_cast<lpl::core::f32>(i) * 0.4f;
        lpl::core::u32 pixel = 0u;
        lpl::voxel::march(pair, two, profile, base, eyeAlongZ(two.metresPerSample(), 64.0f, 64.0f, z, kPencil), once,
                          &pixel, 1u, 1u, 0u, 1u);
        return redLevel(pixel);
    });
    std::printf("  brick face: worst single-step change across it %.4f\n", static_cast<double>(worst));
    check(worst < 0.05f, "crossing a brick face leaves no step in a field that has none");
}

/**
 * @brief The level-of-detail seam.
 *
 * @details MarchParams::levelBlendSamples says what a hard switch between levels looked like. A fine
 * brick over the first 128 samples and a coarse one covering four times that, at a different
 * density: crossing out of the fine one is exactly the seam. Pencil rays are fired at successive
 * points from x = 100 to 160, across the fine brick's outer face at 128, and the biggest jump
 * between neighbours is kept.
 *
 * Measured, not hoped for: 0.62 down to 0.35, a 44 % cut. It is not zero and cannot be,
 * because a ray INTEGRATES -- a density that ramps smoothly still produces a pixel that
 * does not, since opacity accumulates nonlinearly along the path. The band softens the
 * step; only a much wider band would flatten it, and a wide band throws away the detail it
 * was supposed to be showing. This fixture is also deliberately extreme: 200 against 150,
 * with the coarse level's window at 141 to 191, so the two levels are opaque against
 * nearly transparent.
 */
void checkLevelOfDetailSeamIsBlended(const lpl::voxel::VolumeGeometry &geometry, lpl::core::f32 metresPerSample,
                                     const lpl::voxel::DensityProfile &profile,
                                     const lpl::voxel::TransferFunction &base, const lpl::voxel::MarchParams &params)
{
    const auto near = makeFlatBrick(200u);
    const auto far = makeFlatBrick(150u);
    lpl::voxel::BrickMosaic layered;
    layered.insert(viewOf(far, lpl::voxel::BrickKey{2u, 0, 0, 0}));
    layered.insert(viewOf(near, lpl::voxel::BrickKey{0u, 0, 0, 0}));

    auto biggestStep = [&](lpl::core::f32 band) {
        lpl::voxel::MarchParams p = params;
        p.levelBlendSamples = band;
        p.shading = 0.0f;
        p.boundaryOpacity = 0.0f;
        return worstAdjacentChange(60, [&](int i) {
            const lpl::core::f32 x = 100.0f + static_cast<lpl::core::f32>(i);
            lpl::core::u32 pixel = 0u;
            lpl::voxel::march(layered, geometry, profile, base, eyeAlongZ(metresPerSample, x, 64.0f, -20.0f, kPencil),
                              p, &pixel, 1u, 1u, 0u, 1u);
            return redLevel(pixel);
        });
    };

    const lpl::core::f32 hard = biggestStep(0.0f);
    const lpl::core::f32 soft = biggestStep(24.0f);
    std::printf("  level seam: hard switch jumps %.4f, blended jumps %.4f\n", static_cast<double>(hard),
                static_cast<double>(soft));
    check(hard > 0.02f, "the fixture really does have a seam to hide");
    check(soft < hard * 0.7f, "blending across the band cuts the worst jump by a third or more");
}

/**
 * @brief The corridor: looking ALONG the sheets versus ACROSS them.
 *
 * @details This is the whole point of walking a scroll, and it is also the only check here that can
 * catch a transposed axis mapping. The sheets are perpendicular to volume X, so a ray running
 * along volume Z stays in the gap and comes out the far side; the same ray turned to cross
 * them piles up three sheets and goes opaque. Swap the world-to-volume mapping and the two
 * assertions trade places -- while every other check in this file still passes, because a
 * transposed subject is still a plausible subject. The ray starts at x = 16.5, inside a gap.
 *
 * The ratio is the claim, not either number: from inside a corridor you see ALONG it and
 * not THROUGH its walls. A threshold on saturation would instead be a claim about the
 * peak opacity somebody picked, which is a setting rather than a property of the field.
 */
void checkCorridorIsSeenAlongAndNotThrough(const lpl::voxel::BrickMosaic &mosaic,
                                           const lpl::voxel::VolumeGeometry &geometry, lpl::core::f32 metresPerSample,
                                           const lpl::voxel::DensityProfile &profile,
                                           const lpl::voxel::TransferFunction &base,
                                           const lpl::voxel::MarchParams &params)
{
    const lpl::voxel::Eye down = eyeAlongZ(metresPerSample, 16.5f, 64.0f, 2.0f, kPencil);
    lpl::voxel::Eye across = down;
    across.forward = {1.0f, 0.0f, 0.0f};
    across.right = {0.0f, 0.0f, -1.0f};

    lpl::core::u32 onePixel = 0u;
    const lpl::voxel::MarchReport alongRay =
        lpl::voxel::march(mosaic, geometry, profile, base, down, params, &onePixel, 1u, 1u, 0u, 1u);
    const lpl::core::f32 alongRed = redLevel(onePixel);

    const lpl::voxel::MarchReport acrossRay =
        lpl::voxel::march(mosaic, geometry, profile, base, across, params, &onePixel, 1u, 1u, 0u, 1u);
    const lpl::core::f32 acrossRed = redLevel(onePixel);

    std::printf("  along the corridor: %llu steps, red %.3f | across the sheets: %llu steps, red %.3f\n",
                static_cast<unsigned long long>(alongRay.steps), static_cast<double>(alongRed),
                static_cast<unsigned long long>(acrossRay.steps), static_cast<double>(acrossRed));

    checkEq(static_cast<long long>(alongRay.escaped), 1, "a ray down the corridor comes out the far side");
    check(alongRed < 0.15f, "and comes out having picked up almost nothing: the gap is a gap");
    check(acrossRed > 0.6f, "a ray across the sheets is stopped by them");
    check(acrossRed > alongRed * 10.0f, "the wall is an order of magnitude brighter than the corridor");
}

/**
 * @brief The profile, measured rather than assumed.
 *
 * @details Sheets are 4 samples of every 32, so an eighth of the matter is sheet.
 */
void checkProfileIsMeasuredRatherThanAssumed()
{
    const auto sheetBrick = makeSheetBrick(0u, 0, 32u, 4u);
    lpl::voxel::BrickView v = viewOf(sheetBrick, lpl::voxel::BrickKey{0u, 0, 0, 0});
    const lpl::voxel::DensityProfile measuredProfile = lpl::voxel::measureProfile(&v, 1u, 0.125f);
    check(measuredProfile.valid(), "a profile comes out usable");
    check(measuredProfile.sheetSample > kMedium, "the sheet end of the window is above the medium");
    check(measuredProfile.sheetSample <= kSheet, "and does not exceed the brightest matter present");
    check(measuredProfile.mean > kMedium && measuredProfile.mean < kSheet, "the mean sits between the two");
}

/**
 * @brief The free camera, and the bug it exists to not repeat.
 *
 * @details FreeCamera's header names the bug: a sine that does not reduce its argument freezes the
 * camera, and the first sixty degrees of every turn still work perfectly. The reference is
 * the library sine: the same reduction on both sides would only prove the code agrees with
 * itself -- a mistake this repository has already recorded twice. The sine is compared over four
 * full turns both ways, then out to ten thousand radians.
 *
 * The camera then turns a half turn, sixteen turns more -- the case that froze -- and six hundred
 * and fifty more. Pitch is clamped rather than wrapped: rolling over the vertical flips the horizon
 * and no reading of the controls recovers from it.
 *
 * Moving with no input must not move: a normalize of a zero vector is how a camera drifts.
 */
void checkFreeCameraTurnsWithoutFreezing()
{
    const auto worstSineError = [](int first, int last, lpl::core::f32 radiansPerStep) {
        lpl::core::f32 worst = 0.0f;
        for (int i = first; i <= last; ++i)
        {
            const lpl::core::f32 a = static_cast<lpl::core::f32>(i) * radiansPerStep;
            const lpl::core::f32 got = lpl::voxel::wrappedSine(a);
            const lpl::core::f32 want = static_cast<lpl::core::f32>(std::sin(static_cast<double>(a)));
            const lpl::core::f32 e = got > want ? got - want : want - got;
            if (e > worst)
                worst = e;
        }
        return worst;
    };
    const lpl::core::f32 nearTurns = worstSineError(-1440, 1440, 0.0174532925f);
    const lpl::core::f32 farTurns = worstSineError(-7300, 7300, 1.37f);
    std::printf("  wrapped sine: worst error %.2e over four turns, %.2e out to 1e4 radians\n",
                static_cast<double>(nearTurns), static_cast<double>(farTurns));
    check(nearTurns < 2e-4f, "the sine holds over the whole circle, not just near zero");
    check(farTurns < 2e-4f, "and holds out to ten thousand radians, well past 64 turns");
    check(lpl::voxel::wrapAngle(std::numeric_limits<lpl::core::f32>::infinity()) == 0.0f &&
              lpl::voxel::wrapAngle(std::numeric_limits<lpl::core::f32>::quiet_NaN()) == 0.0f,
          "an angle that is not a number wraps to zero instead of poisoning a direction");
    bool hugeStayOnTheCircle = true;
    for (lpl::core::f32 huge = 1.0e8f; huge < 3.0e38f; huge *= 3.0f)
    {
        for (const lpl::core::f32 a : {huge, -huge, 1.0e17f})
        {
            const lpl::core::f32 wrapped = lpl::voxel::wrapAngle(a);
            const lpl::core::f32 sine = lpl::voxel::wrappedSine(a);
            if (!(wrapped >= -3.1415927f && wrapped <= 3.1415927f) || !(sine >= -1.0f && sine <= 1.0f))
                hugeStayOnTheCircle = false;
        }
    }
    check(hugeStayOnTheCircle, "an angle too large to hold a fraction of a turn still lands on the circle");

    const auto facesYaw = [](const lpl::voxel::Eye &e, double yaw) {
        return std::fabs(static_cast<double>(e.forward.x) - std::sin(yaw)) < 1e-3 &&
               std::fabs(static_cast<double>(e.forward.z) - std::cos(yaw)) < 1e-3;
    };
    lpl::voxel::FreeCamera camera;
    camera.turn(3.14159265f, 0.0f);
    check(camera.eye().forward.z < -0.99f, "a half turn actually faces the other way");
    camera.turn(100.0f, 0.0f);
    check(facesYaw(camera.eye(), 3.14159265 + 100.0), "sixteen more turns face exactly where they should");
    camera.turn(4096.0f, 0.0f);
    check(facesYaw(camera.eye(), 3.14159265 + 100.0 + 4096.0), "and so do six hundred and fifty more");

    camera.turn(0.0f, 100.0f);
    check(camera.pitch() < 1.5708f, "pitch stops just short of straight up");
    camera.turn(0.0f, -200.0f);
    check(camera.pitch() > -1.5708f, "and just short of straight down");

    lpl::voxel::FreeCamera still;
    const lpl::math::Vec3<lpl::core::f32> before = still.position;
    still.move(0.0f, 0.0f, 0.0f, 10.0f);
    check(still.position.x == before.x && still.position.y == before.y && still.position.z == before.z,
          "no input moves nothing");
    still.move(1.0f, 0.0f, 0.0f, 5.0f);
    check(still.position.z > 4.9f && still.position.z < 5.1f, "forward moves forward by the distance asked");
}

/**
 * @brief The overlay, and the three rules that keep it honest.
 *
 * @details MarchParams::overlay says why a prediction must never look like the scan. The overlay
 * here is "ink everywhere", the most demanding overlay there is.
 *
 * Rule one: off by default. A caller that does not ask gets the scan.
 *
 * Rule two: confidence multiplies the tint. A map worth half must not look as solid as one
 * worth nearly all -- putting the number in a caption instead is how uncertainty gets lost.
 *
 * Rule three: the overlay only tints matter the scan already put there. A prediction
 * floating in a void would be the model drawing rather than reading.
 */
void checkOverlayKeepsItsThreeRules(const lpl::voxel::BrickMosaic &mosaic, const lpl::voxel::VolumeGeometry &geometry,
                                    const lpl::voxel::DensityProfile &profile, const lpl::voxel::TransferFunction &base,
                                    const lpl::voxel::Eye &eye, const lpl::voxel::MarchParams &params,
                                    const std::vector<lpl::core::u32> &frame, lpl::core::u32 kW, lpl::core::u32 kH)
{
    const auto ink = makeFlatBrick(255u);
    lpl::voxel::BrickMosaic predicted;
    predicted.insert(viewOf(ink, lpl::voxel::BrickKey{0u, 0, 0, 0}));

    std::vector<lpl::core::u32> plainFrame(frame.size(), 0u);
    std::vector<lpl::core::u32> tinted(frame.size(), 0u);

    lpl::voxel::MarchParams bare = params;
    const lpl::voxel::MarchReport noOverlay =
        lpl::voxel::march(mosaic, geometry, profile, base, eye, bare, plainFrame.data(), kW, kH, 0u, kH);
    checkEq(static_cast<long long>(noOverlay.overlaid), 0, "no overlay is asked for, so none is painted");

    lpl::voxel::MarchParams shown = params;
    shown.overlay = &predicted;
    shown.overlayConfidence = 1.0f;
    const lpl::voxel::MarchReport withOverlay =
        lpl::voxel::march(mosaic, geometry, profile, base, eye, shown, tinted.data(), kW, kH, 0u, kH);
    check(withOverlay.overlaid > 0, "asked for, it paints");

    auto redness = [](const std::vector<lpl::core::u32> &px, lpl::core::u32 background) {
        double r = 0.0;
        double other = 0.0;
        for (lpl::core::u32 p : px)
        {
            if (p == background)
                continue;
            r += static_cast<double>((p >> 16) & 0xFFu);
            other += static_cast<double>(((p >> 8) & 0xFFu) + (p & 0xFFu)) * 0.5;
        }
        return other > 0.0 ? r / other : 0.0;
    };
    const double plainRed = redness(plainFrame, params.background);
    const double fullRed = redness(tinted, params.background);
    check(fullRed > plainRed * 1.2, "and at full confidence it is clearly visible");

    std::vector<lpl::core::u32> half(frame.size(), 0u);
    lpl::voxel::MarchParams unsure = shown;
    unsure.overlayConfidence = 0.3f;
    lpl::voxel::march(mosaic, geometry, profile, base, eye, unsure, half.data(), kW, kH, 0u, kH);
    const double halfRed = redness(half, params.background);
    std::printf("  overlay redness: none %.3f, confidence 0.3 -> %.3f, confidence 1.0 -> %.3f\n", plainRed, halfRed,
                fullRed);
    check(halfRed > plainRed && halfRed < fullRed, "a less confident map is painted less strongly");

    lpl::voxel::BrickMosaic emptyScan;
    std::vector<lpl::core::u32> nothing(frame.size(), 0u);
    const lpl::voxel::MarchReport onNothing =
        lpl::voxel::march(emptyScan, geometry, profile, base, eye, shown, nothing.data(), kW, kH, 0u, kH);
    checkEq(static_cast<long long>(onNothing.overlaid), 0, "with no scan under it, an overlay paints nothing");
}

} // namespace

/**
 * @brief What the volume renderer must do, checked against a field whose answer is known.
 *
 * @details The density profile the march checks share carries the spread ratios measured on one
 * paired cubic millimetre of PHerc0172. A sheet is four samples thick in this fixture, and a sheet
 * you can see through is not a sheet: the transfer curve's peak opacity is derived from that intent
 * rather than picked.
 *
 * @warning **The fixture is synthetic, and that is the point.** A real brick is a download, and a
 * check that needs the network goes red when the network does. The sheets here come from an
 * integer formula, so the expected answer is derivable by hand -- a slab of high density every
 * `pitch` samples inside a medium that is dark but never empty, which is the shape real
 * carbonised papyrus turned out to have when it was actually measured.
 */
int main()
{
    std::printf("voxel raymarch\n");

    checkBricksAndIndexing();
    checkMosaicAnswersWithTheFinestBrick();
    checkNearestSampleIsReadFromTheBrickFindReturns();
    checkResidencyPlansNearestFirst();

    lpl::voxel::DensityProfile profile{};
    profile.floorSample = 140u;
    profile.sheetSample = 200u;
    profile.mean = 147.0f;
    profile.deviation = 23.5f;
    const lpl::core::f32 measured[6]{1.0f, 0.940f, 0.821f, 0.651f, 0.485f, 0.383f};
    for (lpl::core::u32 lv = 0u; lv < 6u; ++lv)
        profile.spreadRatio[lv] = measured[lv];

    const lpl::core::f32 peakAlpha = lpl::voxel::alphaForOpaqueAfter(4.0f, 0.85f);
    check(peakAlpha > 0.3f && peakAlpha < 0.6f, "the derived peak opacity is in a sane range");
    const lpl::voxel::TransferFunction base =
        lpl::voxel::rampTransfer(profile.floorSample, profile.sheetSample, peakAlpha);
    checkTransferCurveFollowsTheCollapsingSpread(profile, base);

    lpl::voxel::VolumeGeometry geometry{};
    geometry.samples[0] = 128;
    geometry.samples[1] = 128;
    geometry.samples[2] = 128;
    geometry.levels = 1u;
    geometry.voxelMicrometres = 7.91f;
    geometry.metresPerMicrometre = 11538.0f;
    const lpl::core::f32 metresPerSample = geometry.metresPerSample();

    const auto sheets = makeSheetBrick(0u, 0, 32u, 4u);
    lpl::voxel::BrickMosaic mosaic;
    mosaic.insert(viewOf(sheets, lpl::voxel::BrickKey{0u, 0, 0, 0}));

    constexpr lpl::core::u32 kW = 96u;
    constexpr lpl::core::u32 kH = 64u;
    std::vector<lpl::core::u32> frame(static_cast<std::size_t>(kW) * kH, 0u);

    lpl::voxel::MarchParams params{};
    params.stepSamples = 1.0f;
    params.maxDistanceMetres = 1.0e6f;

    lpl::voxel::Eye eye{};
    eye.position = {-40.0f * metresPerSample, 64.0f * metresPerSample, 64.0f * metresPerSample};
    eye.forward = {1.0f, 0.0f, 0.0f};
    eye.right = {0.0f, 0.0f, 1.0f};
    eye.up = {0.0f, 1.0f, 0.0f};

    checkMarchTracesEveryPixelAndPaintsTheSheets(mosaic, geometry, profile, base, eye, params, frame, kW, kH);
    checkLookingAwayProducesNothing(mosaic, geometry, profile, base, eye, params, frame, kW, kH);
    checkEmptyBrickIsSkippedWhole(geometry, profile, base, eye, params, frame, kW, kH);
    checkOccupancyGridLeavesTheMediumInOneComparison(geometry, profile, base, params, frame, kW, kH);
    checkMissingBricksAreCounted(geometry, profile, base, eye, params, frame, kW, kH);
    checkStepLengthCorrectionKeepsAFaintMediumSteady(geometry, profile, eye, params, frame, kW, kH);
    checkDoublingTheStepBarelyMovesTheSheets(mosaic, geometry, profile, base, eye, params, frame, kW, kH);
    checkSamePoseRendersTheSameFrame(mosaic, geometry, profile, base, eye, params, frame, kW, kH);
    checkBandedRenderingEqualsOnePass(mosaic, geometry, profile, base, eye, params, frame, kW, kH);
    checkShadingLightsASurfaceNotASilhouette(geometry, profile, base, eye, params, frame, kW, kH);
    checkBoundaryOpacityBuysDepth(geometry, profile, base, eye, params, frame, kW, kH);
    checkSkippedBrickLeavesTheRayAtItsExit(geometry, profile, base, params);
    checkNoSkipLeapsOverAFinerResidentBrick(geometry, profile, base, params);
    checkBrickFaceIsNotVisible(geometry, profile, base, params);
    checkLevelOfDetailSeamIsBlended(geometry, metresPerSample, profile, base, params);
    checkCorridorIsSeenAlongAndNotThrough(mosaic, geometry, metresPerSample, profile, base, params);
    checkProfileIsMeasuredRatherThanAssumed();
    checkFreeCameraTurnsWithoutFreezing();
    checkOverlayKeepsItsThreeRules(mosaic, geometry, profile, base, eye, params, frame, kW, kH);

    std::printf("%s (%d failures, %d checks)\n", gFailures == 0 ? "ALL PASS" : "FAILURES", gFailures, gChecks);
    return gFailures == 0 ? 0 : 1;
}
