/**
 * @file VulkanRenderer.hpp
 * @brief Concrete IRenderer implementation wrapping the Vulkan API.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-27
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_RENDER_VK_VULKANRENDERER_HPP
#    define LPL_RENDER_VK_VULKANRENDERER_HPP

#    include <lpl/render/IRenderer.hpp>
#    include <memory>
#    include <string>

struct GLFWwindow;

namespace lpl::render::vk {

class Wrapper;

/**
 * @class VulkanRenderer
 * @brief Integrates the legacy VkWrapper into LplPlugin's rendering interface.
 */
class VulkanRenderer final : public IRenderer {
public:
    VulkanRenderer();
    ~VulkanRenderer() override;

    /**
     * @brief Creates the wrapper only.
     *
     * The instance and the pipeline need the window, which the engine hands to
     * initVulkanContext once it has created it; they are configured there.
     */
    [[nodiscard]] core::Expected<void> init(core::u32 width, core::u32 height) override;

    void beginFrame() override;
    void endFrame() override;

    void resize(core::u32 width, core::u32 height) override;
    /**
     * @brief Destroys the wrapper; idempotent.
     *
     * Guards against double-destroy: Engine::shutdown() calls shutdown() then
     * reset() on the unique_ptr, which invokes the destructor (and thus
     * shutdown() again). Nullifying the wrapper makes the call idempotent.
     */
    void shutdown() override;

    /**
     * @brief Initializes the Vulkan instance and pipeline bounding it to a GLFW window.
     *
     * The shaders and the texture are read from LPL_SHADER_DIR and LPL_ASSET_DIR,
     * which render/xmake.lua defines as absolute paths so the binary works
     * regardless of the current working directory; without them, from shaders/
     * and assets/ relative to it.
     */
    void initVulkanContext(GLFWwindow *window);

    [[nodiscard]] const char *name() const noexcept override { return "VulkanRenderer (VkWrapper)"; }

    /**
     * @brief Exposes the under-the-hood wrapper for BCI / GUI injections.
     */
    [[nodiscard]] class Wrapper &getWrapper() noexcept;

private:
    std::unique_ptr<class Wrapper> _wrapper;
};

} // namespace lpl::render::vk

#endif // LPL_RENDER_VK_VULKANRENDERER_HPP
