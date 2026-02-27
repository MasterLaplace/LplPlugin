#include "surface/Surface.hpp"
#include <lpl/core/Log.hpp>
#include <cstdlib>

namespace lpl::render::vk {

void Surface::Create(GLFWwindow *window, const VkInstance &instance, VkAllocationCallbacks *allocator)
{
    if (glfwCreateWindowSurface(instance, window, allocator, &_surface) != VK_SUCCESS)
        { ::lpl::core::Log::fatal("Failed to create window surface!"); std::abort(); }
}

void Surface::Destroy(const VkInstance &instance) { vkDestroySurfaceKHR(instance, _surface, nullptr); }

} // namespace lpl::render::vk
