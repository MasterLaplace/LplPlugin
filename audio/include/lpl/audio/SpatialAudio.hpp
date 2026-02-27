/**
 * @file SpatialAudio.hpp
 * @brief HRTF-based spatial audio engine.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_AUDIO_SPATIALAUDIO_HPP
    #define LPL_AUDIO_SPATIALAUDIO_HPP

#include <lpl/audio/IAudioEngine.hpp>
#include <lpl/core/NonCopyable.hpp>

#include <memory>

namespace lpl::audio {

/**
 * @class SpatialAudio
 * @brief Spatial audio using head-related transfer functions (HRTF).
 *
 * Provides 3D positional sound rendering for VR immersion.
 */
class SpatialAudio final : public IAudioEngine,
                           public core::NonCopyable<SpatialAudio>
{
public:
    SpatialAudio();
    ~SpatialAudio() override;

    [[nodiscard]] core::Expected<void> init() override;
    void shutdown() override;

    void setListenerPosition(const math::Vec3<core::f32>& position,
                             const math::Vec3<core::f32>& forward,
                             const math::Vec3<core::f32>& up) override;

    void playAt(core::u32 soundId,
                const math::Vec3<core::f32>& position,
                core::f32 volume) override;

    void stopAll() override;

    [[nodiscard]] const char* name() const noexcept override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace lpl::audio

#endif // LPL_AUDIO_SPATIALAUDIO_HPP
