#include "framebuffer/Framebuffer.hpp"
#include <lpl/core/Log.hpp>
#include <cstdlib>
#include <array>

namespace lpl::render::vk {

void Framebuffer::Create(const VkDevice &device, const CreateInfo &info)
{
    _swapChainFramebuffers.resize(info.swapChainImageViews.size());

    for (size_t i = 0; i < info.swapChainImageViews.size(); ++i)
    {
        std::array<VkImageView, 3> attachments = {info.colorImageView, info.depthImageView, info.swapChainImageViews[i]};

        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = info.renderPass;
        framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        framebufferInfo.pAttachments = attachments.data();
        framebufferInfo.width = info.swapChainExtent.width;
        framebufferInfo.height = info.swapChainExtent.height;
        framebufferInfo.layers = 1;

        if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &_swapChainFramebuffers[i]) != VK_SUCCESS)
            { ::lpl::core::Log::fatal("failed to create framebuffer!"); std::abort(); }
    }
}

void Framebuffer::Destroy(const VkDevice &device) const
{
    for (auto framebuffer : _swapChainFramebuffers)
        vkDestroyFramebuffer(device, framebuffer, nullptr);
}

} // namespace lpl::render::vk
