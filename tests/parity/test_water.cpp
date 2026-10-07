/**
 * @file test_water.cpp
 * @brief Water: the swell runs where it is pointed, its gradient is its own derivative, and foam
 *        whitens the shore more than the open sea.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-06-28
 * @copyright MIT License
 */

#include <cstdio>
#include <lpl/render/Water.hpp>

#include <utility>

static int failures = 0;

static void check(bool ok, const char *what)
{
    std::printf("  %s: %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok)
        ++failures;
}

int main()
{
    // ── The swell's direction ───────────────────────────────────────────────
    //
    // The two crest directions used to be constants in the ripple function; they are now a
    // drift a caller points. The property worth asserting is not that the waves move — it is
    // the one the original comment named and that setting four fields by hand would lose:
    // the two crests must NOT be perpendicular, or they interfere into a checkerboard that
    // reads as a tiled floor.
    {
        std::printf("== water: the swell runs where it is pointed ==\n");
        for (const auto &dir : {
                 std::pair{1.0f,  0.0f },
                 std::pair{0.0f,  1.0f },
                 std::pair{-1.0f, 0.0f },
                 std::pair{0.6f,  -0.8f}
        })
        {
            lpl::render::WaterParams water;
            water.setDrift(dir.first, dir.second);
            const float dot = water.driftX * water.crossX + water.driftZ * water.crossZ;
            const float driftLen = water.driftX * water.driftX + water.driftZ * water.driftZ;
            const float crossLen = water.crossX * water.crossX + water.crossZ * water.crossZ;
            // Cosine of the angle, squared, against the 103 degrees the pair is rotated by:
            // cos(103) is about -0.225, so the squared cosine is about 0.0506. A pair that had
            // become perpendicular would give zero, and a parallel one would give one.
            const float cosSq = (dot * dot) / (driftLen * crossLen);
            check(cosSq > 0.02f && cosSq < 0.10f, "the crossing crest stays off perpendicular");
            check(crossLen > 0.5f * driftLen, "and keeps the drift's magnitude");
        }
    }

    // ── The swell ───────────────────────────────────────────────────────────
    //
    // Every field of it defaults to OFF, which is the first thing to assert: a lake in a
    // courtyard must not start heaving because a sea was added to the engine.
    {
        std::printf("== water: the swell ==\n");
        const lpl::render::WaterParams still;
        check(lpl::render::waterHeight(3.0f, 7.0f, still) == 0.0f, "a surface with no swell is flat");

        lpl::render::WaterParams sea;
        sea.swellHeight = 0.8f;
        sea.crestSharpness = 0.6f;
        sea.foamCrest = 0.52f;
        float highest = -1.0e9f;
        float lowest = 1.0e9f;
        for (int i = 0; i < 512; ++i)
        {
            const float h = lpl::render::waterHeight(static_cast<float>(i) * 0.37f, static_cast<float>(i) * 0.11f, sea);
            highest = h > highest ? h : highest;
            lowest = h < lowest ? h : lowest;
        }
        std::printf("    swell spans %.3f .. %.3f for a stated height of %.2f\n", static_cast<double>(lowest),
                    static_cast<double>(highest), static_cast<double>(sea.swellHeight));
        check(highest <= sea.swellHeight + 1.0e-4f && lowest >= -sea.swellHeight - 1.0e-4f,
              "and one with a swell stays inside the height it declares");

        // The property the previous version did NOT have, and the reason foam could not be
        // trusted to land on a crest: the gradient must be the derivative of the SAME sum
        // the height comes from. The old normal used the wave's value, which is a quarter
        // period out — a highlight on the flank instead of the face.
        //
        // Only the SIGN is compared: the gradient is normalised against its own worst case
        // rather than the height's, so its magnitude is not d(height)/dx. Points where the
        // difference is tiny are skipped — that is a crest or a trough, where the sign of a
        // finite difference is noise.
        int compared = 0;
        int disagreements = 0;
        for (int i = 0; i < 400; ++i)
        {
            const float x = static_cast<float>(i) * 0.23f;
            const float z = 5.0f;
            const float step = 0.004f;
            const float before = lpl::render::waterHeight(x - step, z, sea);
            const float after = lpl::render::waterHeight(x + step, z, sea);
            const float difference = after - before;
            if (difference > -0.0008f && difference < 0.0008f)
                continue;
            // A triangle wave has TURNING POINTS, and it is not differentiable at them. If
            // one lies inside the sampling interval the finite difference is a chord across
            // a peak, whose sign says nothing about either side — so those points are
            // excluded rather than counted as disagreements. Detected by the slope changing
            // sign across the interval, which is what a turning point is.
            const float slopeBefore = lpl::render::sampleWater(x - step, z, sea).slopeX;
            const float slopeAfter = lpl::render::sampleWater(x + step, z, sea).slopeX;
            if ((slopeBefore > 0.0f) != (slopeAfter > 0.0f))
                continue;
            const float slope = lpl::render::sampleWater(x, z, sea).slopeX;
            ++compared;
            if ((difference > 0.0f) != (slope > 0.0f))
                ++disagreements;
        }
        std::printf("    %d sampled slopes, %d disagreeing in sign with the height they came from\n", compared,
                    disagreements);
        check(compared > 100, "enough points were steep enough to judge");
        // Not exactly zero, and the allowance is reasoned rather than tuned: three octaves
        // of a folded triangle have turning points everywhere, and a sampling interval can
        // straddle TWO of them — the slope then has the same sign at both ends while the
        // chord between them does not. What this rules out is the defect it exists for: a
        // normal taken from the wave's VALUE is a quarter period out of phase and disagrees
        // on about half of all points, so two per cent is a fifty-fold margin, not a
        // threshold chosen to make today's numbers pass.
        check(disagreements * 50 <= compared, "the gradient is the derivative of the height, not a second guess at it");

        // A crest is the TOP of the wave and nothing else. A threshold that let the whole
        // surface count as a crest would make the sea milk and the scatter uniform.
        int cresting = 0;
        int crestAboveThreshold = 0;
        for (int i = 0; i < 2000; ++i)
        {
            const float x = static_cast<float>(i) * 0.17f;
            const lpl::render::WaveSample sample = lpl::render::sampleWater(x, 2.0f, sea);
            if (sample.crest <= 0.0f)
                continue;
            ++cresting;
            if (sample.height > sea.foamCrest - 1.0e-5f)
                ++crestAboveThreshold;
        }
        std::printf("    %d of 2000 samples are cresting, %d of them above the threshold\n", cresting,
                    crestAboveThreshold);
        // A BAND, not a floor. Eleven samples in two thousand was the first measurement and
        // it is a flat calm — a whitecap you meet once a minute is one nobody sees. Most of
        // the sea breaking is a different failure and just as wrong.
        check(cresting > 40, "enough of the sea is cresting to see whitecaps");
        check(cresting < 1000, "but not most of it");
        check(cresting == crestAboveThreshold, "and a crest is exactly the part above the stated threshold");

        // Sharpening broadens the trough and narrows the peak, so the MEAN drops. Asserted
        // as monotonicity across three settings rather than against a number: a fixed
        // number here would be one chosen so that today's tuning passes.
        float previous = 1.0e9f;
        bool monotone = true;
        for (float sharpness : {0.0f, 0.45f, 0.9f})
        {
            lpl::render::WaterParams shaped = sea;
            shaped.crestSharpness = sharpness;
            float total = 0.0f;
            for (int i = 0; i < 1024; ++i)
                total += lpl::render::sampleWater(static_cast<float>(i) * 0.031f, 1.0f, shaped).height;
            const float mean = total / 1024.0f;
            std::printf("    sharpness %.2f -> mean height %+.4f\n", static_cast<double>(sharpness),
                        static_cast<double>(mean));
            if (mean >= previous)
                monotone = false;
            previous = mean;
        }
        check(monotone, "a sharper crest is a broader trough");
    }

    // ── Foam ────────────────────────────────────────────────────────────────
    //
    // Foam is the only part of a sea whose colour comes from neither what is behind it nor
    // what is above it, so it is the one thing that must be applied LAST — after the
    // reflection, the Fresnel and the glint. Two claims: it is off unless asked for, and
    // when asked for it whitens the shallows and not the deeps.
    {
        std::printf("== water: foam ==\n");
        const lpl::render::SunState sun = lpl::render::sunAt(0.32f);
        const lpl::render::SkyParams sky;
        const lpl::math::Vec3<float> eye{0.0f, 6.0f, -12.0f};

        lpl::render::WaterParams dry;
        dry.swellHeight = 0.8f;
        dry.foamGain = 0.0f;
        lpl::render::WaterParams surf = dry;
        surf.foamGain = 0.9f;
        surf.foamWidth = 1.8f;

        const unsigned shallowDry = lpl::render::waterColour(2.0f, 0.0f, 4.0f, eye, sun, sky, dry, 0.1f);
        const unsigned shallowSurf = lpl::render::waterColour(2.0f, 0.0f, 4.0f, eye, sun, sky, surf, 0.1f);
        const unsigned deepDry = lpl::render::waterColour(2.0f, 0.0f, 4.0f, eye, sun, sky, dry, 40.0f);
        const unsigned deepSurf = lpl::render::waterColour(2.0f, 0.0f, 4.0f, eye, sun, sky, surf, 40.0f);

        const auto luminance = [](unsigned c) {
            return static_cast<int>((c >> 16) & 0xFFu) + static_cast<int>((c >> 8) & 0xFFu) +
                   static_cast<int>(c & 0xFFu);
        };
        std::printf("    shallow %06X -> %06X, deep %06X -> %06X\n", shallowDry, shallowSurf, deepDry, deepSurf);
        check(shallowSurf != shallowDry, "asking for foam changes the shallows");
        check(luminance(shallowSurf) > luminance(shallowDry), "and makes them brighter, which is what surf is");
        // Deep water may still whiten AT a crest — that is a whitecap, and it is wanted.
        // What must not happen is the shore band reaching out to sea, so the deep water must
        // whiten LESS than the shallows do.
        check(luminance(shallowSurf) - luminance(shallowDry) > luminance(deepSurf) - luminance(deepDry),
              "and whitens a shore more than open water");
    }

    std::printf("%s (%d failures)\n", failures == 0 ? "ALL PASS" : "FAILURES", failures);
    return failures == 0 ? 0 : 1;
}
