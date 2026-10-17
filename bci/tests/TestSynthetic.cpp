#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "lpl/bci/source/SyntheticSource.hpp"
#include "lpl/bci/source/sim/SyntheticGenerator.hpp"

namespace lpl::bci {

TEST_CASE("SyntheticGenerator produces correct sample count", "[source][synthetic]")
{
    const std::uint64_t seed = 42;
    const std::size_t channelCount = 4;

    lpl::bci::source::SyntheticGenerator gen(seed, channelCount);
    auto samples = gen.generate(100);

    REQUIRE(samples.size() == 100);
    for (const auto &s : samples)
    {
        REQUIRE(s.channels.size() == 4);
    }
}

TEST_CASE("SyntheticGenerator is deterministic with same seed", "[source][synthetic]")
{
    const std::uint64_t seed = 123;
    const std::size_t channelCount = 2;

    lpl::bci::source::SyntheticGenerator gen1(seed, channelCount);
    lpl::bci::source::SyntheticGenerator gen2(seed, channelCount);

    auto samples1 = gen1.generate(50);
    auto samples2 = gen2.generate(50);

    for (std::size_t i = 0; i < 50; ++i)
    {
        for (std::size_t ch = 0; ch < 2; ++ch)
        {
            REQUIRE_THAT(static_cast<double>(samples1[i].channels[ch]),
                         Catch::Matchers::WithinAbs(static_cast<double>(samples2[i].channels[ch]), 1e-6));
        }
    }
}

TEST_CASE("SyntheticGenerator produces non-zero signal with oscillators", "[source][synthetic]")
{
    lpl::bci::source::SyntheticGenerator gen(42, 1);

    lpl::bci::source::SyntheticProfile profile;
    profile.oscillators = {
        {.freqHz = 10.0f, .amplitudeUv = 50.0f, .phaseOffset = 0.0f}
    };
    profile.noiseAmplitudeUv = 0.0f;
    profile.blinkProbability = 0.0f;

    gen.setProfile(profile);
    auto samples = gen.generate(250);

    float maxVal = 0.0f;
    for (const auto &s : samples)
        maxVal = std::max(maxVal, std::abs(s.channels[0]));

    REQUIRE(maxVal > 10.0f);
}

/**
 * @brief In burst mode (realtime false), read() yields kFftUpdateInterval (32) samples when the buffer
 *        allows.
 */
TEST_CASE("SyntheticSource implements ISource interface", "[source][synthetic]")
{
    const std::uint64_t seed = 42;
    const bool realtime = false;
    const std::size_t channelCount = 2;
    const float sampleRate = 250.0f;

    lpl::bci::source::SyntheticSource source(seed, realtime, channelCount, sampleRate);
    auto startResult = source.start();
    REQUIRE(startResult.has_value());

    std::vector<bci::Sample> buffer(32);
    auto readResult = source.read(buffer);
    REQUIRE(readResult.has_value());
    REQUIRE(*readResult == 32);

    auto info = source.info();
    REQUIRE(info.channelCount == channelCount);
    REQUIRE(info.sampleRate == sampleRate);

    source.stop();
}

} // namespace lpl::bci
