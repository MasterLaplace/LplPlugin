#include "device/PhysicalDevice.hpp"
#include <lpl/core/Log.hpp>
#include <cstdlib>

namespace lpl::render::vk {

void PhysicalDevice::PickPhysicalDevice(const VkInstance &instance, const VkSurfaceKHR &surface)
{
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

    if (deviceCount == 0)
        { ::lpl::core::Log::fatal("failed to find GPUs with Vulkan support!"); std::abort(); }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    for (const auto &device : devices)
    {
        if (IsDeviceSuitable(device, surface))
        {
            _physicalDevice = device;
            DetermineMaxUsableSampleCount();
            break;
        }
    }

    if (_physicalDevice == VK_NULL_HANDLE)
        { ::lpl::core::Log::fatal("failed to find a suitable GPU!"); std::abort(); }
}

bool PhysicalDevice::IsDeviceSuitable(const VkPhysicalDevice &device, const VkSurfaceKHR &surface)
{
    _queueFamilies.FindQueueFamilies(device, surface);

    bool extensionsSupported = CheckDeviceExtensionSupport(device);

    bool swapChainAdequate = false;
    if (extensionsSupported)
    {
        SwapChain::SupportDetails swapChainSupport = SwapChain::QuerySupport(device, surface);
        swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
    }

    VkPhysicalDeviceFeatures supportedFeatures{};
    vkGetPhysicalDeviceFeatures(device, &supportedFeatures);

    return _queueFamilies.IsComplete() && extensionsSupported && swapChainAdequate &&
           supportedFeatures.samplerAnisotropy;
}

bool PhysicalDevice::CheckDeviceExtensionSupport(const VkPhysicalDevice &device) const
{
    uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

    std::set<std::string, std::less<>> requiredExtensions(DEVICE_EXTENSIONS.begin(), DEVICE_EXTENSIONS.end());

    for (const auto &extension : availableExtensions)
        requiredExtensions.erase(extension.extensionName);

    return requiredExtensions.empty();
}

uint32_t PhysicalDevice::RateDeviceSuitability(const VkPhysicalDevice &device) const
{
    VkPhysicalDeviceProperties deviceProperties{};
    VkPhysicalDeviceFeatures deviceFeatures{};
    vkGetPhysicalDeviceProperties(device, &deviceProperties);
    vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

    uint32_t score = 0;

    if (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
        score += 1000;

    score += deviceProperties.limits.maxImageDimension2D;
    score += deviceFeatures.samplerAnisotropy ? 1000 : 0;

    if (!deviceFeatures.geometryShader)
        return 0;

    return score;
}

uint32_t PhysicalDevice::GetQueueFamilyIndex() const { return _queueFamilies.GetIndices().graphicsFamily.value(); }

void PhysicalDevice::DetermineMaxUsableSampleCount()
{
    VkPhysicalDeviceProperties physicalDeviceProperties;
    vkGetPhysicalDeviceProperties(_physicalDevice, &physicalDeviceProperties);

    VkSampleCountFlags counts = physicalDeviceProperties.limits.framebufferColorSampleCounts &
                                physicalDeviceProperties.limits.framebufferDepthSampleCounts;

    if (counts & VK_SAMPLE_COUNT_64_BIT) { _msaaSamples = VK_SAMPLE_COUNT_64_BIT; return; }
    if (counts & VK_SAMPLE_COUNT_32_BIT) { _msaaSamples = VK_SAMPLE_COUNT_32_BIT; return; }
    if (counts & VK_SAMPLE_COUNT_16_BIT) { _msaaSamples = VK_SAMPLE_COUNT_16_BIT; return; }
    if (counts & VK_SAMPLE_COUNT_8_BIT)  { _msaaSamples =  VK_SAMPLE_COUNT_8_BIT; return; }
    if (counts & VK_SAMPLE_COUNT_4_BIT)  { _msaaSamples =  VK_SAMPLE_COUNT_4_BIT; return; }
    if (counts & VK_SAMPLE_COUNT_2_BIT)  { _msaaSamples =  VK_SAMPLE_COUNT_2_BIT; return; }

    _msaaSamples = VK_SAMPLE_COUNT_1_BIT;
}

} // namespace lpl::render::vk
