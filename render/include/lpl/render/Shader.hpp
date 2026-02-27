/**
 * @file Shader.hpp
 * @brief Shader program abstraction.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_RENDER_SHADER_HPP
    #define LPL_RENDER_SHADER_HPP

#include <lpl/core/Types.hpp>
#include <lpl/core/Expected.hpp>

#include <string_view>

namespace lpl::render {

/**
 * @class Shader
 * @brief Represents a compiled vertex + fragment shader pair.
 *
 * Backend-agnostic; concrete renderers handle compilation.
 */
class Shader
{
public:
    virtual ~Shader() = default;

    /** @brief Compiles the shader from source strings. */
    [[nodiscard]] virtual core::Expected<void> compile(
        std::string_view vertexSrc,
        std::string_view fragmentSrc) = 0;

    /** @brief Activates this shader for subsequent draw calls. */
    virtual void bind() = 0;

    /** @brief Deactivates this shader. */
    virtual void unbind() = 0;

    /** @brief Sets an integer uniform. */
    virtual void setUniform(std::string_view name, core::i32 value) = 0;

    /** @brief Sets a float uniform. */
    virtual void setUniform(std::string_view name, core::f32 value) = 0;

    /** @brief Returns a human-readable name. */
    [[nodiscard]] virtual const char* name() const noexcept = 0;
};

} // namespace lpl::render

#endif // LPL_RENDER_SHADER_HPP
