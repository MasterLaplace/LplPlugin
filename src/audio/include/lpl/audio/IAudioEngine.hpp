// /////////////////////////////////////////////////////////////////////////////
/// @file IAudioEngine.hpp
/// @brief Abstract audio engine interface for 3D spatial audio.
// /////////////////////////////////////////////////////////////////////////////

#pragma once

#include <lpl/math/Vec3.hpp>
#include <lpl/core/Types.hpp>
#include <lpl/core/Expected.hpp>

namespace lpl::audio {

// /////////////////////////////////////////////////////////////////////////////
/// @class IAudioEngine
/// @brief Strategy interface for the audio subsystem.
// /////////////////////////////////////////////////////////////////////////////
class IAudioEngine
{
public:
    virtual ~IAudioEngine() = default;

    /// @brief Initializes the audio context.
    [[nodiscard]] virtual core::Expected<void> init() = 0;

    /// @brief Shuts down the audio context.
    virtual void shutdown() = 0;

    /// @brief Updates the listener position and orientation.
    virtual void setListenerPosition(const math::Vec3<core::f32>& position,
                                     const math::Vec3<core::f32>& forward,
                                     const math::Vec3<core::f32>& up) = 0;

    /// @brief Plays a sound at a 3D position.
    /// @param soundId Identifier for the loaded sound asset.
    /// @param position World position.
    /// @param volume Volume multiplier [0, 1].
    virtual void playAt(core::u32 soundId,
                        const math::Vec3<core::f32>& position,
                        core::f32 volume = 1.0f) = 0;

    /// @brief Stops all playing sounds.
    virtual void stopAll() = 0;

    /// @brief Returns a human-readable name.
    [[nodiscard]] virtual const char* name() const noexcept = 0;
};

} // namespace lpl::audio
