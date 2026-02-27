#include "wrapper/Wrapper.hpp"
#include <lpl/core/Log.hpp>
#include <vulkan/vulkan.hpp>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>
#include <unordered_map>

namespace lpl::render::vk {

void Wrapper::CreateInstance(GLFWwindow *window, const std::string &applicationName, const uint32_t width,
                             const uint32_t height)
{
    _instance.Create(applicationName);
    _instance.SetupDebugMessenger();
    _instance.CreateSurface(window);
    _instance.SetupDevices();
    _instance.CreateSwapChainImages(width, height);
}

void Wrapper::CreateGuiInstance(GLFWwindow *window)
{
    _instance.CreateGuiInstance(window);
    _isGui = true;
}

void Wrapper::CreatePipeline()
{
    _instance.CreateGraphicsPipeline(_shaders, _textures, _models, _isDepth);
    _instance.CreateSyncObjects();
}

void Wrapper::Destroy()
{
    _instance.Destroy(_textures);

    _textures.clear();
    _models.clear();

    if (_isGui)
        GUI::DestroyInstance();
}

void Wrapper::AddTexture(const std::string &texturePath, uint32_t &textureId)
{
    textureId = std::hash<std::string>{}(texturePath);

    auto texture = std::make_unique<Texture>();
    texture->Create(texturePath);
    _textures[textureId] = std::move(texture);
}

void Wrapper::AddModel(const std::string &modelPath, uint32_t &modelId)
{
    modelId = std::hash<std::string>{}(modelPath);

    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, modelPath.c_str()))
    {
        ::lpl::core::Log::fatal("failed to load model: " + warn + err);
        std::abort();
    }

    std::vector<::lpl::render::Vertex> vertices;
    std::vector<core::u32> indices;

    std::unordered_map<std::string, uint32_t> uniqueVertices{};

    for (const auto &shape : shapes)
    {
        for (const auto &index : shape.mesh.indices)
        {
            ::lpl::render::Vertex vertex{};

            vertex.position = {attrib.vertices[3 * index.vertex_index + 0], attrib.vertices[3 * index.vertex_index + 1],
                               attrib.vertices[3 * index.vertex_index + 2]};

            if (index.texcoord_index >= 0)
            {
                vertex.u = attrib.texcoords[2 * index.texcoord_index + 0];
                vertex.v = 1.0f - attrib.texcoords[2 * index.texcoord_index + 1];
            }

            if (index.normal_index >= 0)
            {
                vertex.normal = {attrib.normals[3 * index.normal_index + 0], attrib.normals[3 * index.normal_index + 1],
                                 attrib.normals[3 * index.normal_index + 2]};
            }

            std::string hashKey = std::to_string(vertex.position.x) + std::to_string(vertex.position.y) +
                                  std::to_string(vertex.position.z) + std::to_string(vertex.u) +
                                  std::to_string(vertex.v);

            if (!uniqueVertices.contains(hashKey))
            {
                uniqueVertices[hashKey] = static_cast<uint32_t>(vertices.size());
                vertices.push_back(vertex);
            }

            indices.push_back(uniqueVertices[hashKey]);
        }
    }

    auto model = std::make_unique<Model>();
    model->mesh.setVertices(std::move(vertices));
    model->mesh.setIndices(std::move(indices));

    _models[modelId] = std::move(model);
}

void Wrapper::AddModel(const ::lpl::render::Mesh &model, const std::string &modelName, uint32_t &modelId)
{
    modelId = std::hash<std::string>{}(modelName);

    auto newModel = std::make_unique<Model>();
    newModel->mesh = model;
    _models[modelId] = std::move(newModel);
}

void Wrapper::BindTexture(const uint32_t textureId, const uint32_t modelId)
{
    [[maybe_unused]] const auto &texture = _textures[textureId];
    const auto &model = _models[modelId];

    auto it = std::ranges::find(model->textures, textureId);

    if (it == model->textures.end())
        model->textures.emplace_back(textureId);
    else
        ::lpl::core::Log::warn("texture already bound to model");
}

void Wrapper::AddShader(const std::string &shaderPath, const std::string &fname, const ShaderType &shaderType)
{
    switch (shaderType)
    {
    case ShaderType::VERTEX: _shaders.vertex = {shaderPath, fname}; break;
    case ShaderType::FRAGMENT: _shaders.fragment = {shaderPath, fname}; break;

    default: break;
    }
}

Wrapper::Result Wrapper::DrawFrame() { 
    if (_instance.DrawNextImage() == lpl::render::vk::Result::NeedResize) {
        return Wrapper::Result::NeedResize;
    }
    return Wrapper::Result::Success; 
}

void Wrapper::Resize(GLFWwindow *window)
{
    int width = 0;
    int height = 0;

    glfwGetFramebufferSize(window, &width, &height);

    while (width == 0 || height == 0)
    {
        glfwGetFramebufferSize(window, &width, &height);
        glfwWaitEvents();
    }

    _instance.RecreateSwapChain(width, height);
}

void Wrapper::PrintAvailableExtensions()
{
    uint32_t extensionCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

    std::vector<VkExtensionProperties> extensions(extensionCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());

    std::string available_extensions = "available extensions (" + std::to_string(extensionCount) + "):";
    for (const auto &extension : extensions)
        available_extensions += "\n\t" + std::string(extension.extensionName);
    ::lpl::core::Log::info(available_extensions);
}

void Wrapper::PrintVersion() { ::lpl::core::Log::info("Wrapper version: " VKWRAPPER_VERSION_STRING); }

void Wrapper::PrintConfig() { ::lpl::core::Log::info("Wrapper config:\n" VKWRAPPER_CONFIG_STRING); }

void Wrapper::ResizeCallback(GLFWwindow *window, [[maybe_unused]] int width, [[maybe_unused]] int height)
{
    auto wrapperObj = static_cast<Wrapper *>(glfwGetWindowUserPointer(window));
    wrapperObj->SetFramebufferResized();
}

} // namespace lpl::render::vk
