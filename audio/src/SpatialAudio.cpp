#include <lpl/audio/SpatialAudio.hpp>
#include <lpl/core/Assert.hpp>
#include <lpl/core/Log.hpp>

namespace lpl::audio {

struct SpatialAudio::Impl {};

SpatialAudio::SpatialAudio() : _impl{std::make_unique<Impl>()} {}
SpatialAudio::~SpatialAudio() = default;

core::Expected<void> SpatialAudio::init()
{
    core::Log::info("SpatialAudio::init (stub)");
    return {};
}

void SpatialAudio::shutdown() { core::Log::info("SpatialAudio::shutdown (stub)"); }

void SpatialAudio::setListenerPosition(const math::Vec3<core::f32> & /*position*/,
                                       const math::Vec3<core::f32> & /*forward*/, const math::Vec3<core::f32> & /*up*/)
{
    // TODO(#316): implement HRTF-based listener positioning for 3D spatial audio.
    LPL_ASSERT(false && "unimplemented");
}

void SpatialAudio::playAt(core::u32 /*soundId*/, const math::Vec3<core::f32> & /*position*/, core::f32 /*volume*/)
{
    // TODO(#316): implement positional audio playback with distance attenuation.
    LPL_ASSERT(false && "unimplemented");
}

void SpatialAudio::stopAll()
{
    // TODO(#316): implement stopping all active spatial audio sources.
    LPL_ASSERT(false && "unimplemented");
}

const char *SpatialAudio::name() const noexcept { return "SpatialAudio"; }

} // namespace lpl::audio
