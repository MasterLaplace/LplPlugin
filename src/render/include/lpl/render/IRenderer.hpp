// /////////////////////////////////////////////////////////////////////////////
/// @file IRenderer.hpp
/// @brief Abstract renderer interface.
// /////////////////////////////////////////////////////////////////////////////

#pragma once

#include <lpl/core/Types.hpp>
#include <lpl/core/Expected.hpp>

namespace lpl::render {

// /////////////////////////////////////////////////////////////////////////////
/// @class IRenderer
/// @brief Strategy interface for graphics rendering backends.
///
/// Concrete implementations: OpenGL ES, Vulkan, etc.
/// The renderer operates exclusively in float space (no Fixed32).
// /////////////////////////////////////////////////////////////////////////////
class IRenderer
{
public:
    virtual ~IRenderer() = default;

    /// @brief Initializes the rendering context.
    [[nodiscard]] virtual core::Expected<void> init(core::u32 width,
                                                     core::u32 height) = 0;

    /// @brief Begins a new frame.
    virtual void beginFrame() = 0;

    /// @brief Ends the frame and presents.
    virtual void endFrame() = 0;

    /// @brief Resizes the viewport.
    virtual void resize(core::u32 width, core::u32 height) = 0;

    /// @brief Shuts down the renderer.
    virtual void shutdown() = 0;

    /// @brief Returns a human-readable name.
    [[nodiscard]] virtual const char* name() const noexcept = 0;
};

} // namespace lpl::render
