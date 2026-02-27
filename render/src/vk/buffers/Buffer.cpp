#include "buffers/Buffer.hpp"
#include <lpl/core/Log.hpp>
#include <cstdlib>

namespace lpl::render::vk {

void Buffers::Create(const CreateInfo &info, const std::unordered_map<core::u32, std::unique_ptr<Texture>> &textures,
                     const std::unordered_map<core::u32, std::unique_ptr<Model>> &models)
{
    for (const auto &model : models)
    {
        if (!model.second)
            continue;

        auto &mesh = const_cast<::lpl::render::Mesh &>(model.second->mesh);

        CreateVertexBuffer(info.device, info.physicalDevice, info.commandPool, info.graphicsQueue, mesh);
        CreateIndexBuffer(info.device, info.physicalDevice, info.commandPool, info.graphicsQueue, mesh);

        CreateUniformBuffer(info.device, info.physicalDevice, info.swapChainImages);
    }

    for (const auto &texture : textures)
    {
        if (!texture.second)
            continue;

        auto &tex = const_cast<Texture &>(*texture.second);

        CreateTextureBuffer(info.device, info.physicalDevice, info.commandPool, info.graphicsQueue, tex);
        CreateTextureView(info.device, tex);
        CreateTextureSampler(info.device, info.physicalDevice, tex);
    }
}

void Buffers::CreateVertexBuffer(const VkDevice &device, const VkPhysicalDevice &physicalDevice,
                                 const VkCommandPool &commandPool, const VkQueue &graphicsQueue,
                                 const ::lpl::render::Mesh &mesh)
{
    VkDeviceSize bufferSize = sizeof(mesh.vertices()[0]) * mesh.vertices().size();

    VkBuffer stagingBuffer{};
    VkDeviceMemory stagingBufferMemory{};

    CreateBuffer(device, physicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer,
                 stagingBufferMemory);

    void *data = nullptr;
    vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, mesh.vertices().data(), bufferSize);
    vkUnmapMemory(device, stagingBufferMemory);

    CreateBuffer(device, physicalDevice, bufferSize,
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, _vertexBuffer, _vertexBufferMemory);

    CopyBuffer(device, commandPool, graphicsQueue, stagingBuffer, _vertexBuffer, bufferSize);

    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);
}

void Buffers::CreateIndexBuffer(const VkDevice &device, const VkPhysicalDevice &physicalDevice,
                                const VkCommandPool &commandPool, const VkQueue &graphicsQueue,
                                const ::lpl::render::Mesh &mesh)
{
    VkDeviceSize bufferSize = sizeof(mesh.indices()[0]) * mesh.indices().size();
    _indexCount = static_cast<uint32_t>(mesh.indices().size());

    VkBuffer stagingBuffer{};
    VkDeviceMemory stagingBufferMemory{};

    CreateBuffer(device, physicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer,
                 stagingBufferMemory);

    void *data = nullptr;
    vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, mesh.indices().data(), bufferSize);
    vkUnmapMemory(device, stagingBufferMemory);

    CreateBuffer(device, physicalDevice, bufferSize,
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, _indexBuffer, _indexBufferMemory);

    CopyBuffer(device, commandPool, graphicsQueue, stagingBuffer, _indexBuffer, bufferSize);

    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);
}

void Buffers::CreateUniformBuffer(const VkDevice &device, const VkPhysicalDevice &physicalDevice,
                                  [[maybe_unused]] const std::vector<VkImage> &swapChainImages)
{
    VkDeviceSize bufferSize = sizeof(UniformBufferObject);

    _uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
    _uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
    _uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        CreateBuffer(device, physicalDevice, bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, _uniformBuffers[i],
                     _uniformBuffersMemory[i]);

        vkMapMemory(device, _uniformBuffersMemory[i], 0, bufferSize, 0, &_uniformBuffersMapped[i]);
    }
}

void Buffers::CreateTextureBuffer(const VkDevice &device, const VkPhysicalDevice &physicalDevice,
                                  const VkCommandPool &commandPool, const VkQueue &graphicsQueue,
                                  Texture &texture) const
{
    VkBuffer stagingBuffer{};
    VkDeviceMemory stagingBufferMemory{};
    auto size = texture.GetSize();

    CreateBuffer(device, physicalDevice, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer,
                 stagingBufferMemory);

    void *data = nullptr;
    vkMapMemory(device, stagingBufferMemory, 0, size, 0, &data);
    memcpy(data, texture.GetPixels(), static_cast<size_t>(size));
    vkUnmapMemory(device, stagingBufferMemory);

    auto &image = texture.GetImage();

    CreateImage(device, physicalDevice, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL,
                VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, texture);

    TransitionImageLayout(device, commandPool, graphicsQueue, image, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, texture.GetMipLevels());

    CopyBufferToImage(device, commandPool, graphicsQueue, stagingBuffer, texture);
    // transitioned to VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL while generating mipmaps

    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingBufferMemory, nullptr);

    GenerateMipmaps(device, physicalDevice, commandPool, graphicsQueue, texture);
}

void Buffers::GenerateMipmaps(const VkDevice &device, const VkPhysicalDevice &physicalDevice,
                              const VkCommandPool &commandPool, const VkQueue &graphicsQueue, Texture &texture) const
{
    // Check if image format supports linear blitting
    VkFormatProperties formatProperties;
    vkGetPhysicalDeviceFormatProperties(physicalDevice, VK_FORMAT_R8G8B8A8_SRGB, &formatProperties);

    if (!(formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT))
        { ::lpl::core::Log::fatal("texture image format does not support linear blitting!"); std::abort(); }

    VkCommandBuffer commandBuffer = BeginSingleTimeCommands(device, commandPool);

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.image = texture.GetImage();
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.subresourceRange.levelCount = 1;

    int32_t mipWidth = texture.GetWidth();
    int32_t mipHeight = texture.GetHeight();

    for (uint32_t i = 1; i < texture.GetMipLevels(); i++)
    {
        barrier.subresourceRange.baseMipLevel = i - 1;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                             nullptr, 0, nullptr, 1, &barrier);

        VkImageBlit blit{};
        blit.srcOffsets[0] = {0, 0, 0};
        blit.srcOffsets[1] = {mipWidth, mipHeight, 1};
        blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.srcSubresource.mipLevel = i - 1;
        blit.srcSubresource.baseArrayLayer = 0;
        blit.srcSubresource.layerCount = 1;
        blit.dstOffsets[0] = {0, 0, 0};
        blit.dstOffsets[1] = {mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1};
        blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.dstSubresource.mipLevel = i;
        blit.dstSubresource.baseArrayLayer = 0;
        blit.dstSubresource.layerCount = 1;

        vkCmdBlitImage(commandBuffer, texture.GetImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, texture.GetImage(),
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);

        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
                             nullptr, 0, nullptr, 1, &barrier);

        if (mipWidth > 1)
            mipWidth /= 2;
        if (mipHeight > 1)
            mipHeight /= 2;
    }

    barrier.subresourceRange.baseMipLevel = texture.GetMipLevels() - 1;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
                         nullptr, 0, nullptr, 1, &barrier);

    EndSingleTimeCommands(device, commandPool, graphicsQueue, commandBuffer);
}

void Buffers::CreateTextureView(const VkDevice &device, Texture &texture) const
{
    texture.SetTextureView(
        ImageView::CreateImageView(device, texture.GetImage(), VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT, texture.GetMipLevels()));
}

void Buffers::CreateTextureSampler(const VkDevice &device, const VkPhysicalDevice &physicalDevice,
                                   Texture &texture) const
{
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physicalDevice, &properties);

    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.anisotropyEnable = VK_TRUE;
    samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.mipLodBias = 0.0f;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = static_cast<float>(texture.GetMipLevels());

    if (vkCreateSampler(device, &samplerInfo, nullptr, &texture.GetSampler()) != VK_SUCCESS)
        { ::lpl::core::Log::fatal("failed to create texture sampler!"); std::abort(); }
}

void Buffers::CreateDepthResources(const VkDevice &device, const VkPhysicalDevice &physicalDevice,
                                   const VkExtent2D &swapChainExtent, VkSampleCountFlagBits msaaSamples)
{
    VkFormat depthFormat = FindDepthFormat(physicalDevice);
    _depth.Create(swapChainExtent.width, swapChainExtent.height);
    CreateImage(device, physicalDevice, depthFormat, VK_IMAGE_TILING_OPTIMAL,
                VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, _depth, msaaSamples);
    _depth.SetTextureView(
        ImageView::CreateImageView(device, _depth.GetImage(), depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT));
}

void Buffers::DestroyDepthResources(const VkDevice &device) { _depth.Destroy(device); }

void Buffers::CreateColorResources(const VkDevice &device, const VkPhysicalDevice &physicalDevice,
                                   const VkExtent2D &swapChainExtent, VkFormat format, VkSampleCountFlagBits msaaSamples)
{
    _color.Create(swapChainExtent.width, swapChainExtent.height);
    CreateImage(device, physicalDevice, format, VK_IMAGE_TILING_OPTIMAL,
                VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, _color, msaaSamples);
    _color.SetTextureView(
        ImageView::CreateImageView(device, _color.GetImage(), format, VK_IMAGE_ASPECT_COLOR_BIT));
}

void Buffers::DestroyColorResources(const VkDevice &device) { _color.Destroy(device); }

void Buffers::Destroy(const VkDevice &device, [[maybe_unused]] const std::vector<VkImage> &swapChainImages)
{
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
    {
        vkDestroyBuffer(device, _uniformBuffers[i], nullptr);
        vkFreeMemory(device, _uniformBuffersMemory[i], nullptr);
    }

    vkDestroyBuffer(device, _indexBuffer, nullptr);
    vkFreeMemory(device, _indexBufferMemory, nullptr);

    vkDestroyBuffer(device, _vertexBuffer, nullptr);
    vkFreeMemory(device, _vertexBufferMemory, nullptr);
}

void Buffers::UpdateUniformBuffer(const VkDevice &/*device*/, const VkExtent2D swapChainExtent, const uint32_t currentImage)
{
    static auto startTime = std::chrono::high_resolution_clock::now();

    auto currentTime = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

    UniformBufferObject ubo{};
    ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    ubo.proj =
        glm::perspective(glm::radians(45.0f), swapChainExtent.width / (float) swapChainExtent.height, 0.1f, 10.0f);
    ubo.proj[1][1] *= -1;

    if (!_uniformBuffersMapped.empty() && currentImage < _uniformBuffersMapped.size() && _uniformBuffersMapped[currentImage] != nullptr)
    {
        memcpy(_uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
    }
}

void Buffers::CreateBuffer(const VkDevice &device, const VkPhysicalDevice &physicalDevice, const VkDeviceSize size,
                           const VkBufferUsageFlags usage, const VkMemoryPropertyFlags properties, VkBuffer &buffer,
                           VkDeviceMemory &bufferMemory) const
{
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS)
        { ::lpl::core::Log::fatal("failed to create vertex buffer!"); std::abort(); }

    VkMemoryRequirements memRequirements{};
    vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = FindMemoryType(physicalDevice, memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS)
        { ::lpl::core::Log::fatal("failed to allocate vertex buffer memory!"); std::abort(); }

    vkBindBufferMemory(device, buffer, bufferMemory, 0);
}

void Buffers::CreateImage(const VkDevice &device, const VkPhysicalDevice &physicalDevice, VkFormat format,
                          VkImageTiling tiling, VkImageUsageFlags usage, Texture &texture, VkSampleCountFlagBits numSamples) const
{
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = texture.GetWidth();
    imageInfo.extent.height = texture.GetHeight();
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = texture.GetMipLevels();
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = tiling;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.samples = numSamples;

    auto &image = texture.GetImage();
    auto &imageMemory = texture.GetMemory();

    if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS)
        { ::lpl::core::Log::fatal("failed to create image!"); std::abort(); }

    VkMemoryRequirements memRequirements{};
    vkGetImageMemoryRequirements(device, image, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex =
        FindMemoryType(physicalDevice, memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS)
        { ::lpl::core::Log::fatal("failed to allocate image memory!"); std::abort(); }

    vkBindImageMemory(device, image, imageMemory, 0);
}

uint32_t Buffers::FindMemoryType(const VkPhysicalDevice &physicalDevice, const uint32_t typeFilter,
                                 const VkMemoryPropertyFlags properties) const
{
    VkPhysicalDeviceMemoryProperties memProperties{};
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i)
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
            return i;

    { ::lpl::core::Log::fatal("failed to find suitable memory type!"); std::abort(); }
}

void Buffers::CopyBuffer(const VkDevice &device, const VkCommandPool &commandPool, const VkQueue &graphicsQueue,
                         const VkBuffer &srcBuffer, const VkBuffer &dstBuffer, VkDeviceSize size) const
{
    VkCommandBuffer commandBuffer = BeginSingleTimeCommands(device, commandPool);

    VkBufferCopy copyRegion{};
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

    EndSingleTimeCommands(device, commandPool, graphicsQueue, commandBuffer);
}

void Buffers::TransitionImageLayout(const VkDevice &device, const VkCommandPool &commandPool,
                                    const VkQueue &graphicsQueue, const VkImage &image, VkFormat /*format*/,
                                    VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels) const
{
    VkCommandBuffer commandBuffer = BeginSingleTimeCommands(device, commandPool);

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = mipLevels;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    VkPipelineStageFlags sourceStage{};
    VkPipelineStageFlags destinationStage{};

    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
    {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
    {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    else
    {
        { ::lpl::core::Log::fatal("unsupported layout transition!"); std::abort(); }
    }

    vkCmdPipelineBarrier(commandBuffer, sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);

    EndSingleTimeCommands(device, commandPool, graphicsQueue, commandBuffer);
}

void Buffers::CopyBufferToImage(const VkDevice &device, const VkCommandPool &commandPool, const VkQueue &graphicsQueue,
                                VkBuffer buffer, Texture &texture) const
{
    VkCommandBuffer commandBuffer = BeginSingleTimeCommands(device, commandPool);

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {texture.GetWidth(), texture.GetHeight(), 1};

    vkCmdCopyBufferToImage(commandBuffer, buffer, texture.GetImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    EndSingleTimeCommands(device, commandPool, graphicsQueue, commandBuffer);
}

VkCommandBuffer Buffers::BeginSingleTimeCommands(const VkDevice &device, const VkCommandPool &commandPool) const
{
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer{};
    vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    return commandBuffer;
}

void Buffers::EndSingleTimeCommands(const VkDevice &device, const VkCommandPool &commandPool,
                                    const VkQueue &graphicsQueue, VkCommandBuffer commandBuffer) const
{
    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue);

    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

VkFormat Buffers::FindSupportedFormat(const VkPhysicalDevice &physicalDevice, const std::vector<VkFormat> &candidates,
                                      const VkImageTiling tiling, const VkFormatFeatureFlags features) const
{
    for (VkFormat format : candidates)
    {
        VkFormatProperties props{};
        vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);

        if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features)
        {
            return format;
        }
        else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features)
        {
            return format;
        }
    }

    { ::lpl::core::Log::fatal("failed to find supported format!"); std::abort(); }
}

VkFormat Buffers::FindDepthFormat(const VkPhysicalDevice &physicalDevice) const
{
    return FindSupportedFormat(physicalDevice,
                               {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT},
                               VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
}

bool Buffers::HasStencilComponent(const VkFormat format) const
{
    return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
}

} // namespace lpl::render::vk
