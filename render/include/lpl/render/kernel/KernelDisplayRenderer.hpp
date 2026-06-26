/**
 * @file KernelDisplayRenderer.hpp
 * @brief Software renderer over the kernel IDisplayBackend (P3 exit gate).
 *
 * Implements IRenderer using a pure-C++ software rasterizer that writes
 * directly into the linear framebuffer exposed by IDisplayBackend. The
 * triangle geometry authority is Fixed32 (deterministic simulation state);
 * the rasterization (projection → screen coords → pixel fill) is float-based
 * and runs on SSE via the standard -mfpmath=sse compile flags.
 *
 * Only compiled when LPL_TARGET_KERNEL=1 (freestanding build).
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-06-26
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_RENDER_KERNEL_KERNELDISPLAYRENDERER_HPP
#    define LPL_RENDER_KERNEL_KERNELDISPLAYRENDERER_HPP

#    if LPL_TARGET_KERNEL

#        include <lpl/math/FixedPoint.hpp>
#        include <lpl/platform/IDisplayBackend.hpp>
#        include <lpl/render/IRenderer.hpp>

namespace lpl::render::kernel {

/**
 * @class KernelDisplayRenderer
 * @brief Software-rasterized renderer over the kernel linear framebuffer.
 *
 * Draws a rotating equilateral triangle whose rotation state is driven by
 * a Fixed32 angle (deterministic authority); screen-space projection and
 * per-pixel fill use float (non-authoritative, SSE-computed). Satisfies
 * the P3 exit gate: triangle visible in QEMU via the HAL software path.
 */
class KernelDisplayRenderer final : public IRenderer {
public:
    explicit KernelDisplayRenderer(platform::IDisplayBackend &display) noexcept;

    /** @brief Advance the rotation angle by one fixed tick (call from fixedUpdate). */
    void tick() noexcept;

    // IRenderer
    [[nodiscard]] core::Expected<void> init(core::u32 width, core::u32 height) override;
    void beginFrame() override;
    void endFrame() override;
    void resize(core::u32 width, core::u32 height) override;
    void shutdown() override;
    [[nodiscard]] const char *name() const noexcept override;

private:
    void drawTriangle() noexcept;

    platform::IDisplayBackend  &_display;
    platform::SurfaceDescriptor _surface{};
    math::Fixed32               _angle{0};
    bool                        _initialized{false};
};

} // namespace lpl::render::kernel

#    endif // LPL_TARGET_KERNEL
#endif     // LPL_RENDER_KERNEL_KERNELDISPLAYRENDERER_HPP
