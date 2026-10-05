/**************************************************************************
 * VkWrapper v0.0.4
 *
 * VkWrapper is a software package, part of LplPlugin.
 *
 * This file is part of the VkWrapper project that is under the MIT License.
 * https://opensource.org/license/mit
 * Copyright © 2024 by @MasterLaplace, All rights reserved.
 *
 * VkWrapper is free software: you can use, copy, modify, merge, publish and
 * distribute it under the terms of the MIT License, provided this copyright
 * notice and the permission notice are kept. See the LICENSE file.
 *
 * @file GraphicsPipeline.hpp
 * @brief GraphicsPipeline class declaration.
 *
 *
 * @author @MasterLaplace
 * @version 0.0.4
 * @date 2024-10-23
 **************************************************************************/

#ifndef LPL_RENDER_VK_GRAPHICSPIPELINE_HPP_
#define LPL_RENDER_VK_GRAPHICSPIPELINE_HPP_

#include "buffers/Buffer.hpp"
#include "shaderModule/ShaderModule.hpp"

namespace lpl::render::vk {

/**
 * @brief GraphicsPipeline class.
 *
 * @example
 * @code
 * GraphicsPipeline graphicsPipeline;
 * graphicsPipeline.Create(device, swapChainExtent, renderPass, shaders);
 * graphicsPipeline.Destroy(device);
 * @endcode
 */
class GraphicsPipeline {
public:
    /**
     * @brief Creates a graphics pipeline.
     *
     * This function creates a graphics pipeline from the device and the render pass.
     *
     * @param device  The Vulkan device.
     * @param renderPass  The render pass.
     * @param shaders  The shader paths.
     * @param descriptorLayout  The descriptor layout.
     * @param msaaSamples The MSAA maximum usable sample count.
     * @param isDepth  Whether to enable depth testing.
     */
    void Create(const VkDevice &device, const VkRenderPass &renderPass, const ShaderModule::ShaderPaths &shaders,
                const VkDescriptorSetLayout &descriptorLayout, const VkSampleCountFlagBits msaaSamples, bool isDepth);

    /**
     * @brief Destroys the graphics pipeline.
     *
     * This function destroys the graphics pipeline.
     *
     * @param device  The Vulkan device.
     */
    void Destroy(const VkDevice &device);

    /**
     * @brief Gets the graphics pipeline.
     *
     * This function returns the graphics pipeline.
     *
     * @return VkPipeline  The graphics pipeline.
     */
    [[nodiscard]] const VkPipeline &Get() const { return _graphicsPipeline; }

    /**
     * @brief Gets the pipeline layout.
     *
     * This function returns the pipeline layout.
     *
     * @return VkPipelineLayout  The pipeline layout.
     */
    [[nodiscard]] const VkPipelineLayout &GetLayout() const { return _pipelineLayout; }

private:
    /**
     * @brief Sets up the color blend attachment.
     *
     * The color blend attachment in vulkan is used to blend the color of the fragment with the color of the
     * framebuffer. It is used to create effects like transparency.
     *
     * @param colorBlendAttachment  The color blend attachment.
     * @param enableBlend  The enable blend.
     */
    void SetupColorBlendAttachment(VkPipelineColorBlendAttachmentState &colorBlendAttachment,
                                   const VkBool32 enableBlend);

private:
    VkPipelineLayout _pipelineLayout;
    VkPipeline _graphicsPipeline;
};

} // namespace lpl::render::vk

#endif /* !GRAPHICSPIPELINE_HPP_ */
