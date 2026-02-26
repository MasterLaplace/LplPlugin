// /////////////////////////////////////////////////////////////////////////////
/// @file BciAdapter.cpp
/// @brief BciAdapter stub implementation.
// /////////////////////////////////////////////////////////////////////////////

#include <lpl/bci/BciAdapter.hpp>
#include <lpl/core/Assert.hpp>
#include <lpl/core/Log.hpp>

namespace lpl::bci {

struct BciAdapter::Impl
{
    std::unique_ptr<IBciDriver> driver;
    BciAdapterConfig config{};
};

BciAdapter::BciAdapter(std::unique_ptr<IBciDriver> driver,
                       const BciAdapterConfig& config)
    : impl_{std::make_unique<Impl>()}
{
    LPL_ASSERT(driver != nullptr);
    impl_->driver = std::move(driver);
    impl_->config = config;
}

BciAdapter::~BciAdapter() { stop(); }

core::Expected<void> BciAdapter::start()
{
    auto res = impl_->driver->connect();
    if (!res) { return res; }
    return impl_->driver->startStream();
}

void BciAdapter::stop()
{
    if (impl_ && impl_->driver)
    {
        impl_->driver->stopStream();
        impl_->driver->disconnect();
    }
}

core::Expected<input::NeuralInputState> BciAdapter::update()
{
    auto sampleResult = impl_->driver->poll();
    if (!sampleResult)
    {
        return core::Unexpected(sampleResult.error());
    }

    const auto& raw = sampleResult.value();

    input::NeuralInputState state{};
    state.sequence = raw.sequence;
    state.confidence = math::Fixed32{0};
    state.validated = false;

    const core::usize count = (raw.channelCount < input::NeuralInputState::kChannels)
                                  ? raw.channelCount
                                  : input::NeuralInputState::kChannels;

    for (core::usize i = 0; i < count; ++i)
    {
        state.channels[i] = math::Fixed32::fromFloat(raw.channels[i]);
    }

    return state;
}

const IBciDriver& BciAdapter::driver() const noexcept
{
    return *impl_->driver;
}

} // namespace lpl::bci
