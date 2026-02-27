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
 * @file Wrapper.hpp
 * @brief Wrapper class declaration.
 *
 * This class is a wrapper for the Vulkan API.
 * It is used to simplify the use of Vulkan in the Engine².
 *
 * @author @MasterLaplace
 * @version 0.0.4
 * @date 2024-10-15
 **************************************************************************/

#ifndef LPL_RENDER_VK_WRAPPER_HPP_
    #define LPL_RENDER_VK_WRAPPER_HPP_

#include "instance/Instance.hpp"
#include "wrapper/Model.hpp"
#include "buffers/Texture.hpp"
#include "shaderModule/ShaderModule.hpp"

namespace lpl::render::vk {

/**
 * @brief Wrapper class.
 *
 * This class is a wrapper for the Vulkan API.
 * It is used to simplify the use of Vulkan in the Engine².
 *
 * @example "Usage of the Wrapper class:"
 * @code
 * ::lpl::render::Window window(800, 600, "My Engine");
 *
 * lpl::render::vk::Wrapper vkWrapper;
 *
 * vkWrapper.CreateInstance(window.GetGLFWWindow(), "VkWrapper Test", 800, 600);
 *
 * uint32_t textureId;
 * vkWrapper.AddTexture("exemple.png", textureId);
 *
 * uint32_t modelId;
 * vkWrapper.AddModel("exemple.obj", modelId);
 *
 * vkWrapper.BinTexture(modelId, textureId);
 *
 * vkWrapper.AddShader(SHADER_DIR "vert.spv", "main", lpl::render::vk::Wrapper::ShaderType::VERTEX);
 * vkWrapper.AddShader(SHADER_DIR "frag.spv", "main", lpl::render::vk::Wrapper::ShaderType::FRAGMENT);
 *
 * vkWrapper.CreatePipeline();
 *
 * window.SetFramebufferSizeCallback((void *) &vkWrapper, lpl::render::vk::Wrapper::ResizeCallback);
 *
 * Wrapper::PrintConfig();
 * Wrapper::PrintAvailableExtensions();
 *
 * while (!window.ShouldClose())
 * {
 *     glfwPollEvents();
 *     auto result = vkWrapper.DrawFrame();
 *     if (result == lpl::render::vk::Wrapper::Result::NeedResize)
 *         Wrapper.Resize(window.GetGLFWWindow());
 * }
 *
 * vkWrapper.Destroy();
 * @endcode
 */
class Wrapper {
public:
    /**
     * @brief ShaderType enum class.
     *
     * This enum class is used to represent the type of the shader.
     * It can be either VERTEX, FRAGMENT, GEOMETRY, TESSELLATION_CONTROL, or
     * TESSELLATION_EVALUATION.
     */
    enum class ShaderType : uint8_t {
        VERTEX,
        FRAGMENT,
        GEOMETRY,
        TESSELLATION_CONTROL,
        TESSELLATION_EVALUATION
    };

    /**
     * @brief Result enum class.
     * The result of the draw operation.
     */
    enum class Result : uint8_t {
        Success,
        NeedResize
    };

  public:
    /**
     * @brief Create the Wrapper using the Vulkan API.
     *
     * This function creates the Wrapper using the Vulkan API. It creates the
     * instance, the surface, get the physical device, the logical device, the swap
     * chain, the image views.
     *
     * @param window  The GLFW window to create the Wrapper for.
     * @param applicationName  The name of the application.
     * @param width  The width of the window.
     * @param height  The height of the window.
     */
    void CreateInstance(GLFWwindow *window, const std::string &applicationName, const uint32_t width,
                        const uint32_t height);

    /**
     * @brief Create the GUI instance using the Vulkan implementation of ImGui.
     * ImGui is a bloat-free graphical user interface library for C++.
     *
     * @param window  The GLFW window to create the GUI instance for.
     */
    void CreateGuiInstance(GLFWwindow *window);

    /**
     * @brief Create the graphics pipeline using the Vulkan API.
     *
     * This function creates the graphics pipeline using the Vulkan API. It creates
     * the graphics pipeline with the specified shaders and textures.
     */
    void CreatePipeline();

    /**
     * @brief Destroy the Wrapper using the Vulkan API.
     *
     * This function destroys the Wrapper using the Vulkan API. It destroys the
     * fences, the semaphores, the command buffers, the command pool, the frame
     * buffers, the graphics pipeline, the render pass, the image views, the swap
     * chain, the logical device, the physical device, the surface, and the instance.
     */
    void Destroy();

    /**
     * @brief Add a texture to the Wrapper and get the texture id.
     *
     * @param texturePath  The path to the texture.
     * @param textureId  The id of the texture.
     */
    void AddTexture(const std::string &texturePath, uint32_t &textureId);

    /**
     * @brief Add a 3D model to the Wrapper and get the model id.
     *
     * @param modelPath  The path to the model.
     * @param modelId  The id of the model.
     */
    void AddModel(const std::string &modelPath, uint32_t &modelId);

    /**
     * @brief Bind a texture to a model using the texture id and the model id.
     *
     * @param textureId  The id of the texture.
     * @param modelId  The id of the model.
     */
    void AddModel(const ::lpl::render::Mesh &model, const std::string &modelName, uint32_t &modelId);

    /**
     * @brief Bind a texture to a model using the texture id and the model id.
     *
     * @param textureId  The id of the texture.
     * @param modelId  The id of the model.
     */
    void BindTexture(const uint32_t textureId, const uint32_t modelId);

    /**
     * @brief Add a shader to the Wrapper.
     *
     * @note if a shader with the same type is added, the new shader will replace the old one.
     *
     * @param shaderPath  The path to the shader.
     * @param fname  The name of the shader.
     * @param shaderType  The type of the shader.
     */
    void AddShader(const std::string &shaderPath, const std::string &fname, const ShaderType &shaderType);

    /**
     * @brief Enable the depth test using the Vulkan API.
     *
     * This function enables the depth test using the Vulkan API.
     *
     * @param enable  Enable or disable the depth test. The default value is true.
     */
    inline void EnableDepthTest(bool enable = true) { _isDepth = enable; }

    /**
     * @brief Draw a frame using the Vulkan API.
     *
     * This function draws a frame using the Vulkan API. It waits for the fences
     * to be signaled, resets the fences, acquires the next image in the swap chain,
     * records the command buffer, submits the command buffer to the graphics queue,
     * presents the image to the screen, and increments the current frame index.
     *
     * @return Wrapper::Result  The result of the draw operation. Success if the
     * image was drawn successfully, NeedResize if a Resize is needed.
     *
     * @see Wrapper::Result
     * @see Resize
     */
    [[nodiscard]] Wrapper::Result DrawFrame();

    /**
     * @brief Resize the window using the Vulkan API.
     *
     * This function resizes the window using the Vulkan API. It gets the new
     * width and height of the window, waits for the window to be resized, and
     * recreates the swap chain with the new dimensions.
     *
     * @param window  The GLFW window to resize.
     */
    void Resize(GLFWwindow *window);

    /**
     * @brief Set the framebuffer resized flag.
     *
     * This function sets the framebuffer resized flag to indicate that the
     * framebuffer needs to be resized.
     * The flag is used to trigger the recreation of the swap chain when the
     * window is resized.
     */
    inline void SetFramebufferResized() { _instance.SetFramebufferResized(true); }

    /**
     * @brief Set the clear color of the Wrapper.
     *
     * This function sets the clear color of the Wrapper.
     *
     * @param color  The color to set.
     */
    inline void ChangeClearColor(const glm::vec4 &color) { _instance.SetClearColor(color); }

    /**
     * @brief Callback function for the framebuffer resize event.
     *
     * This function is called when the framebuffer is resized.
     * It sets the framebuffer resized flag to indicate that the framebuffer
     * needs to be resized.
     *
     * @param window  The GLFW window that was resized.
     * @param width  The new width of the window. (unused)
     * @param height  The new height of the window. (unused)
     */
    static void ResizeCallback(GLFWwindow *window, int width, int height);

    /**
     * @brief Print the available extensions for the Vulkan API.
     *
     */
    static void PrintAvailableExtensions();

    /**
     * @brief Print the version of the VkWrapper.
     *
     */
    static void PrintVersion();

    /**
     * @brief Print the configuration of the VkWrapper.
     *
     */
    static void PrintConfig();

private:
    Instance _instance;
    ShaderModule::ShaderPaths _shaders;
    std::unordered_map<core::u32, std::unique_ptr<Texture>> _textures{};
    std::unordered_map<core::u32, std::unique_ptr<Model>> _models{};
    bool _isDepth = false;
    bool _isGui = false;
};

} // namespace lpl::render::vk

#endif /* !WRAPPER_HPP_ */
