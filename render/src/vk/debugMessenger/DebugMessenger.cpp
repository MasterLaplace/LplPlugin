#include "debugMessenger/DebugMessenger.hpp"
#include <lpl/core/Log.hpp>
#include <lpl/core/Log.hpp>
#include <cstdlib>

namespace lpl::render::vk {

DebugMessenger::~DebugMessenger()
{
    if (enableValidationLayers)
        DestroyDebugUtilsMessengerEXT(_instance, nullptr);
}

void DebugMessenger::SetupDebugMessenger(const VkInstance &instance)
{
    if (!enableValidationLayers)
        return;

    VkDebugUtilsMessengerCreateInfoEXT createInfo;
    PopulateDebugMessengerCreateInfo(createInfo);

    if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr) != VK_SUCCESS)
        { ::lpl::core::Log::fatal("failed to set up debug messenger!"); std::abort(); }
}

VKAPI_ATTR VkBool32 VKAPI_CALL DebugMessenger::Callback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                                        [[maybe_unused]] VkDebugUtilsMessageTypeFlagsEXT messageType,
                                                        const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
                                                        [[maybe_unused]] void *pUserData)
{
    std::string msg = "validation layer: " + std::string(pCallbackData->pMessage);
    if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
        ::lpl::core::Log::error(msg);
    else if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
        ::lpl::core::Log::warn(msg);
    else if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT)
        ::lpl::core::Log::info(msg);
    else
        ::lpl::core::Log::debug(msg);
        
    return VK_FALSE;
}

void DebugMessenger::PopulateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT &createInfo) const
{
    if (!enableValidationLayers)
        return;

    createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                                 VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                 VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = DebugMessenger::Callback;
}

VkResult DebugMessenger::CreateDebugUtilsMessengerEXT(const VkInstance &instance,
                                                      const VkDebugUtilsMessengerCreateInfoEXT *pCreateInfo,
                                                      const VkAllocationCallbacks *pAllocator)
{
    if (auto func =
            (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
        func != nullptr)
        return func(instance, pCreateInfo, pAllocator, &_debugMessenger);

    return VK_ERROR_EXTENSION_NOT_PRESENT;
}

void DebugMessenger::DestroyDebugUtilsMessengerEXT(const VkInstance &instance, const VkAllocationCallbacks *pAllocator)
{
    if (!enableValidationLayers)
        return;

    auto func =
        (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr)
        func(instance, _debugMessenger, pAllocator);
}

} // namespace lpl::render::vk
