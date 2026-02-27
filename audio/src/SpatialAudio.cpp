/**
 * @file SpatialAudio.cpp
 * @brief SpatialAudio stub implementation.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#include <lpl/audio/SpatialAudio.hpp>
#include <stdexcept>
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

void SpatialAudio::shutdown()
{
    core::Log::info("SpatialAudio::shutdown (stub)");
}

void SpatialAudio::setListenerPosition(
    const math::Vec3<core::f32>& /*position*/,
    const math::Vec3<core::f32>& /*forward*/,
    const math::Vec3<core::f32>& /*up*/)
{
    LPL_ASSERT(false && "unimplemented");
}

void SpatialAudio::playAt(core::u32 /*soundId*/,
                           const math::Vec3<core::f32>& /*position*/,
                           core::f32 /*volume*/)
{
    LPL_ASSERT(false && "unimplemented");
}

void SpatialAudio::stopAll()
{
    LPL_ASSERT(false && "unimplemented");
}

const char* SpatialAudio::name() const noexcept { return "SpatialAudio"; }

} // namespace lpl::audio
