/**
 * @file Mesh.hpp
 * @brief GPU mesh data descriptor.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_RENDER_MESH_HPP
    #define LPL_RENDER_MESH_HPP

#include <lpl/math/Vec3.hpp>
#include <lpl/core/Types.hpp>

#include <vector>

namespace lpl::render {

/**
 * @struct Vertex
 * @brief Per-vertex data for rendering.
 */
struct Vertex
{
    math::Vec3<core::f32> position;
    math::Vec3<core::f32> normal;
    core::f32             u{0.0f};
    core::f32             v{0.0f};
};

/**
 * @class Mesh
 * @brief CPU-side mesh representation (vertex + index data).
 */
class Mesh
{
public:
    Mesh() = default;
    ~Mesh() = default;

    /** @brief Sets vertex data. */
    void setVertices(std::vector<Vertex> vertices) { _vertices = std::move(vertices); }

    /** @brief Sets index data (triangle list). */
    void setIndices(std::vector<core::u32> indices) { _indices = std::move(indices); }

    [[nodiscard]] const std::vector<Vertex>&   vertices() const noexcept { return _vertices; }
    [[nodiscard]] const std::vector<core::u32>& indices() const noexcept  { return _indices; }
    [[nodiscard]] core::u32 vertexCount() const noexcept { return static_cast<core::u32>(_vertices.size()); }
    [[nodiscard]] core::u32 indexCount() const noexcept  { return static_cast<core::u32>(_indices.size()); }

private:
    std::vector<Vertex>    _vertices;
    std::vector<core::u32> _indices;
};

} // namespace lpl::render

#endif // LPL_RENDER_MESH_HPP
