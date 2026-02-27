#include "imageView/ImageView.hpp"
#include <lpl/core/Log.hpp>
#include <cstdlib>

namespace lpl::render::vk {

void ImageView::Create(const VkDevice &device, const std::vector<VkImage> &swapChainImages,
                       const VkSurfaceFormatKHR surfaceFormat, const uint32_t layerCount)
{
    _swapChainImageViews.resize(swapChainImages.size() * layerCount);

    for (size_t i = 0; i < swapChainImages.size(); ++i)
    {
        for (uint32_t layer = 0; layer < layerCount; ++layer)
        {
            _swapChainImageViews[i * layerCount + layer] =
                CreateImageView(device, swapChainImages[i], surfaceFormat.format, VK_IMAGE_ASPECT_COLOR_BIT, 1, layer);
        }
    }
}

void ImageView::Destroy(const VkDevice &device) const
{
    for (const auto &imageView : _swapChainImageViews)
        vkDestroyImageView(device, imageView, nullptr);
}

VkImageView ImageView::CreateImageView(const VkDevice &device, VkImage image, VkFormat format,
                                       VkImageAspectFlags aspectFlags, uint32_t mipLevels, uint32_t layer)
{
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
    viewInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
    viewInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
    viewInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
    viewInfo.subresourceRange.aspectMask = aspectFlags;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = mipLevels;
    viewInfo.subresourceRange.baseArrayLayer = layer;
    viewInfo.subresourceRange.layerCount = 1;

    VkImageView imageView{};
    if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS)
        { ::lpl::core::Log::fatal("failed to create image view!"); std::abort(); }

    return imageView;
}

} // namespace lpl::render::vk
