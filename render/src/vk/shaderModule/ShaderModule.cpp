#include "shaderModule/ShaderModule.hpp"
#include <lpl/core/Log.hpp>
#include <cstdlib>

namespace lpl::render::vk {

std::vector<char> ShaderModule::LoadSPVfile(const std::string &filename)
{
    if (!filename.ends_with(".spv"))
        { ::lpl::core::Log::fatal("file is not an spv file: " + filename); std::abort(); }

    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open())
        { ::lpl::core::Log::fatal("failed to open file: " + filename); std::abort(); }

    auto fileSize = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();

    return buffer;
}

VkShaderModule ShaderModule::Create(const VkDevice &device, const std::vector<char> &code)
{
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t *>(code.data());

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS)
        { ::lpl::core::Log::fatal("failed to create shader module!"); std::abort(); }

    return shaderModule;
}

void ShaderModule::Destroy(const VkDevice &device, const VkShaderModule &shaderModule)
{
    vkDestroyShaderModule(device, shaderModule, nullptr);
}

VkPipelineShaderStageCreateInfo ShaderModule::CreateShaderStage(const VkShaderModule &shaderModule,
                                                                const VkShaderStageFlagBits stage,
                                                                const std::string &pName)
{
    VkPipelineShaderStageCreateInfo shaderStageInfo{};
    shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageInfo.stage = stage;
    shaderStageInfo.module = shaderModule;
    shaderStageInfo.pName = pName.c_str();
    return shaderStageInfo;
}

} // namespace lpl::render::vk
