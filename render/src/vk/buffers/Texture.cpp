/*
** EPITECH PROJECT, 2025
** VkWrapper-Test [WSL : Ubuntu]
** File description:
** Texture
*/

#include "buffers/Texture.hpp"
#include <lpl/core/Log.hpp>
#include <cstdlib>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

namespace lpl::render::vk {

void Texture::Create(const std::string &texturePath)
{
    _pixels = stbi_load(texturePath.c_str(), &_width, &_height, &_channels, STBI_rgb_alpha);

    if (!_pixels)
        { ::lpl::core::Log::fatal("failed to load texture image (" + texturePath + "): " + stbi_failure_reason()); std::abort(); }

    _mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(_width, _height)))) + 1;
}

void Texture::Create(const uint32_t width, const uint32_t height)
{
    _width = width;
    _height = height;
    _mipLevels = 1;
}

void Texture::Destroy(const VkDevice &_device)
{
    if (_image != VK_NULL_HANDLE && _imageMemory != VK_NULL_HANDLE && _imageView != VK_NULL_HANDLE)
    {
        vkDestroyImageView(_device, _imageView, nullptr);
        vkDestroyImage(_device, _image, nullptr);
        vkFreeMemory(_device, _imageMemory, nullptr);
    }

    if (!_pixels)
        return;

    stbi_image_free(_pixels);
    _pixels = nullptr;
    _width = 0;
    _height = 0;
    _channels = 0;
    _mipLevels = 1;
}

} // namespace lpl::render::vk
