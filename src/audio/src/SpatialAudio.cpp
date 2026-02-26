// /////////////////////////////////////////////////////////////////////////////
/// @file SpatialAudio.cpp
/// @brief SpatialAudio stub implementation.
// /////////////////////////////////////////////////////////////////////////////

#include <lpl/audio/SpatialAudio.hpp>
#include <lpl/core/Assert.hpp>
#include <lpl/core/Log.hpp>

namespace lpl::audio {

struct SpatialAudio::Impl {};

SpatialAudio::SpatialAudio() : impl_{std::make_unique<Impl>()} {}
SpatialAudio::~SpatialAudio() = default;

core::Expected<void> SpatialAudio::init()
{
    core::Log::info("SpatialAudio::init (stub)");
    return {};
}

void SpatialAudio::shutdown()
{
    core::Log::info("SpatialAudio::shutdown (stub)");
}

void SpatialAudio::setListenerPosition(
    const math::Vec3<core::f32>& /*position*/,
    const math::Vec3<core::f32>& /*forward*/,
    const math::Vec3<core::f32>& /*up*/)
{
    LPL_ASSERT(false && "SpatialAudio::setListenerPosition not yet implemented");
}

void SpatialAudio::playAt(core::u32 /*soundId*/,
                           const math::Vec3<core::f32>& /*position*/,
                           core::f32 /*volume*/)
{
    LPL_ASSERT(false && "SpatialAudio::playAt not yet implemented");
}

void SpatialAudio::stopAll()
{
    LPL_ASSERT(false && "SpatialAudio::stopAll not yet implemented");
}

const char* SpatialAudio::name() const noexcept { return "SpatialAudio"; }

} // namespace lpl::audio
