// /////////////////////////////////////////////////////////////////////////////
/// @file Mesh.hpp
/// @brief GPU mesh data descriptor.
// /////////////////////////////////////////////////////////////////////////////

#pragma once

#include <lpl/math/Vec3.hpp>
#include <lpl/core/Types.hpp>

#include <vector>

namespace lpl::render {

// /////////////////////////////////////////////////////////////////////////////
/// @struct Vertex
/// @brief Per-vertex data for rendering.
// /////////////////////////////////////////////////////////////////////////////
struct Vertex
{
    math::Vec3<core::f32> position;
    math::Vec3<core::f32> normal;
    core::f32             u{0.0f};
    core::f32             v{0.0f};
};

// /////////////////////////////////////////////////////////////////////////////
/// @class Mesh
/// @brief CPU-side mesh representation (vertex + index data).
// /////////////////////////////////////////////////////////////////////////////
class Mesh
{
public:
    Mesh() = default;
    ~Mesh() = default;

    /// @brief Sets vertex data.
    void setVertices(std::vector<Vertex> vertices) { vertices_ = std::move(vertices); }

    /// @brief Sets index data (triangle list).
    void setIndices(std::vector<core::u32> indices) { indices_ = std::move(indices); }

    [[nodiscard]] const std::vector<Vertex>&   vertices() const noexcept { return vertices_; }
    [[nodiscard]] const std::vector<core::u32>& indices() const noexcept  { return indices_; }
    [[nodiscard]] core::u32 vertexCount() const noexcept { return static_cast<core::u32>(vertices_.size()); }
    [[nodiscard]] core::u32 indexCount() const noexcept  { return static_cast<core::u32>(indices_.size()); }

private:
    std::vector<Vertex>    vertices_;
    std::vector<core::u32> indices_;
};

} // namespace lpl::render
