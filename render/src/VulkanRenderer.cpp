#include <lpl/render/VulkanRenderer.hpp>
#include <lpl/core/Log.hpp>
#include <stdexcept>

#include "wrapper/Wrapper.hpp"
#include "vk/wrapper/Wrapper.hpp"

namespace lpl::render::vk {

VulkanRenderer::VulkanRenderer() = default;

VulkanRenderer::~VulkanRenderer()
{
    shutdown();
}

core::Expected<void> VulkanRenderer::init(core::u32 /*width*/, core::u32 /*height*/)
{
    _wrapper = std::make_unique<Wrapper>();
    // Note: The original engine created the instance immediately here.
    // In reality, apps/client/main.cpp must provide the window.
    // We will configure the pipeline lazily when textures and models are ready.
    core::Log::info("VulkanRenderer initialized.");
    return core::Expected<void>{};
}

void VulkanRenderer::beginFrame()
{
    // Usually clears or waits for semaphores
}

void VulkanRenderer::endFrame()
{
    auto result = _wrapper->DrawFrame();
    if (result == Wrapper::Result::NeedResize)
    {
        // Window pointer is held internally by Wrapper's instance
    }
}

void VulkanRenderer::resize(core::u32 /*width*/, core::u32 /*height*/)
{
    // The Wrapper already intercepts resize via GLFW callback, 
    // but we might need explicit bounds setting here.
}

void VulkanRenderer::initVulkanContext(GLFWwindow* window)
{
    _wrapper->CreateInstance(window, "LplPlugin Client", 800, 600);
    _wrapper->CreatePipeline();
}

void VulkanRenderer::shutdown()
{
    _wrapper->Destroy();
}

Wrapper& VulkanRenderer::getWrapper() noexcept
{
    return *_wrapper;
}

} // namespace lpl::render::vk
