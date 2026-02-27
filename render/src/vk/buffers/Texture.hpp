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
 * @file Texture.hpp
 * @brief Texture class declaration.
 *
 *
 * @author @MasterLaplace
 * @version 0.0.4
 * @date 2025-02-17
 **************************************************************************/

#ifndef LPL_RENDER_VK_TEXTURE_HPP_
    #define LPL_RENDER_VK_TEXTURE_HPP_

#include "debugMessenger/DebugMessenger.hpp"

namespace lpl::render::vk {

/**
 * @brief Texture class to handle Vulkan textures.
 *
 * This class is used to load and contain Vulkan texture data.
 */
class Texture {
  public:
    /**
     * @brief Construct a new Texture object.
     */
    explicit Texture() = default;

    /**
     * @brief Construct a new Texture object
     *
     * @param texturePath  Path to the texture file.
     */
    void Create(const std::string &texturePath);

    /**
     * @brief Create a new Texture object only used for depth images.
     *
     * @param width  The width of the texture.
     * @param height  The height of the texture.
     */
    void Create(const uint32_t width, const uint32_t height);

    /**
     * @brief Destroy the Texture object including the Vulkan image and memory and the pixels.
     *
     * @param _device  The Vulkan device.
     */
    void Destroy(const VkDevice &_device);

    /**
     * @brief Set the Texture View object
     *
     * @param textureView  The Vulkan texture view.
     */
    void SetTextureView(const VkImageView &textureView) { _imageView = textureView; }

    /**
     * @brief Get the Pixels object
     *
     * @return const uint8_t*  The pixels of the texture.
     */
    [[nodiscard]] const uint8_t *GetPixels() const { return _pixels; }

    /**
     * @brief Get the Size object
     *
     * @return VkDeviceSize  The size of the texture.
     */
    [[nodiscard]] VkDeviceSize GetSize() const { return _width * _height * 4; }

    /**
     * @brief Get the Width object
     *
     * @return uint32_t  The width of the texture.
     */
    [[nodiscard]] uint32_t GetWidth() const { return static_cast<uint32_t>(_width); }

    /**
     * @brief Get the Height object
     *
     * @return uint32_t  The height of the texture.
     */
    [[nodiscard]] uint32_t GetHeight() const { return static_cast<uint32_t>(_height); }

    /**
     * @brief Get the Mip levels
     * 
     * @return uint32_t  The number of mip levels. 
     */
    [[nodiscard]] uint32_t GetMipLevels() const { return _mipLevels; }

    /**
     * @brief Get the Channels object
     *
     * @return int  The number of channels of the texture.
     */
    [[nodiscard]] int GetChannels() const { return _channels; }

    /**
     * @brief Get the Image object
     *
     * @return VkImage&  The Vulkan image.
     */
    [[nodiscard]] VkImage &GetImage() { return _image; }

    /**
     * @brief Get the Image Memory object
     *
     * @return VkDeviceMemory&  The Vulkan image memory.
     */
    [[nodiscard]] VkDeviceMemory &GetMemory() { return _imageMemory; }

    /**
     * @brief Get the Image View object
     *
     * @return VkImageView  The Vulkan image view.
     */
    [[nodiscard]] VkImageView GetView() const { return _imageView; }

    /**
     * @brief Get the Texture Sampler object
     *
     * @return VkSampler  The Vulkan texture sampler.
     */
    [[nodiscard]] VkSampler &GetSampler() { return textureSampler; }

  protected:
  private:
    uint8_t *_pixels = nullptr;
    int _width = 0;
    int _height = 0;
    int _channels = 0;
    VkImage _image = VK_NULL_HANDLE;
    VkImageView _imageView = VK_NULL_HANDLE;
    VkSampler textureSampler = VK_NULL_HANDLE;
    VkDeviceMemory _imageMemory = VK_NULL_HANDLE;
    uint32_t _mipLevels = 1;
};

} // namespace lpl::render::vk

#endif /* !TEXTURE_HPP_ */
