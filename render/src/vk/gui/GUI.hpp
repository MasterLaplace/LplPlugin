/**************************************************************************
 * VkWrapper v0.0.4
 *
 * VkWrapper is a software package, part of the Engine².
 *
 * This file is part of the VkWrapper project that is under GPL-3.0 License.
 * Copyright © 2024 by @EngineSquared, All rights reserved.
 *
 * VkWrapper is a free software: you can redistribute it and/or modify
 * it under the terms of the GPL-3.0 License as published by the
 * Free Software Foundation. See the GPL-3.0 License for more details.
 *
 * @file GUI.hpp
 * @brief GUI class declaration.
 *
 * This class is used to create a GUI instance for the Vulkan API.
 * It provides a graphical user interface for the application to interact with
 * the user and display information. It is used for debugging and development
 * purposes to visualize data and control the application. The GUI class is
 * based on the ImGui library and provides a simple and intuitive interface
 * for creating user interfaces.
 *
 * @ref https://frguthmann.github.io/posts/vulkan_imgui/
 *
 * @author @MasterLaplace
 * @version 0.0.4
 * @date 2024-10-24
 **************************************************************************/

#ifndef LPL_RENDER_VK_GUI_HPP_
    #define LPL_RENDER_VK_GUI_HPP_

#include "buffers/Buffer.hpp"
#include "queueFamilies/QueueFamilies.hpp"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"
#include <memory>

namespace lpl::render::vk {

/**
 * @brief GUI class.
 *
 * @example "Usage of the GUI class:"
 * @code
 * lpl::render::vk::GUI::CreateInfo info{};
 * info.window = window;
 * info.instance = instance;
 * info.physicalDevice = physicalDevice;
 * info.device = device;
 * info.queueFamily = queueFamily;
 * info.queue = queue;
 * info.minImageCount = minImageCount;
 * info.descriptorPool = descriptorPool;
 * info.renderPass = renderPass;
 * info.allocator = allocator;
 * lpl::render::vk::GUI::CreateInstance(info);
 * lpl::render::vk::GUI::GetInstance().Render(clear_color, commandBuffer);
 * lpl::render::vk::GUI::DestroyInstance();
 */
class GUI {
  public:
    struct CreateInfo {
        GLFWwindow *window;
        VkInstance instance;
        VkPhysicalDevice physicalDevice;
        VkDevice device;
        uint32_t queueFamily;
        VkQueue queue;
        uint32_t minImageCount;
        VkDescriptorPool descriptorPool;
        VkRenderPass renderPass;
        const VkAllocationCallbacks *allocator;
    };

    static inline void CreateInstance(const CreateInfo &info)
    {
        if (_instance == nullptr)
            _instance = std::make_unique<GUI>(info);
    }

    static inline void DestroyInstance()
    {
        if (_instance != nullptr)
            _instance.reset();
    }

    static inline GUI &GetInstance()
    {
        if (!_instance) {
            ::lpl::core::Log::fatal("GUI instance not created!");
            std::abort();
        }
        return *_instance;
    }

    /**
     * @brief Protected GUI constructor.
     *
     * This ensures that the GUI class can only be instantiated through
     * the CreateInstance method.
     */
    GUI(const CreateInfo &info);

    /**
     * @brief Protected GUI destructor.
     *
     * This ensures that the GUI class can be properly destroyed.
     */
    ~GUI();

    /**
     * @brief Deleted copy constructor.
     *
     * This prevents copying of the GUI instance.
     *
     * @param copy The instance to copy.
     */
    GUI(const GUI &copy) = delete;

    /**
     * @brief Deleted copy assignment operator.
     *
     * This prevents assignment of the GUI instance.
     *
     * @param copy The instance to assign.
     * @return T& A reference to the assigned instance.
     */
    GUI &operator=(const GUI &copy) = delete;

    static inline void Resize() { ImGui_ImplVulkan_SetMinImageCount(2); }

    void Render(const glm::vec4 &clear_color, const VkCommandBuffer &commandBuffer);

    static void check_vk_result(VkResult err);

  private:
    static inline std::unique_ptr<GUI> _instance = nullptr;
    VkPipelineCache _pipelineCache = VK_NULL_HANDLE;
    bool _show_demo_window = true;
    bool _show_another_window = false;
    ImGuiIO *_io = nullptr;
};

} // namespace lpl::render::vk

#endif /* !GUI_HPP_ */
