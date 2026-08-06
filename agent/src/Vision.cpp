/**
 * @file Vision.cpp
 * @brief Implementation of the vision-language-action seam.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/agent/Vision.hpp>

#include <lpl/core/Error.hpp>
#include <lpl/ecs/Archetype.hpp>
#include <lpl/ecs/Component.hpp>
#include <lpl/ecs/Partition.hpp>
#include <lpl/ecs/Registry.hpp>
#include <lpl/image/Codec.hpp>
#include <lpl/math/Vec3.hpp>
#include <lpl/render/Box.hpp>
#include <lpl/render/Lighting.hpp>
#include <lpl/render/OrbitCamera.hpp>
#include <lpl/render/SoftwareRasterizer.hpp>

#include <cstdio>
#include <vector>

namespace lpl::agent {

namespace {

constexpr core::f32 kPi = 3.14159265358979323846f;
constexpr core::f32 kBackground = 0x000A0E18u; ///< The same near-black the terrain world clears to.

/// Half-extent used for an entity that declares no AABB.
constexpr core::f32 kDefaultHalfExtent = 0.5f;

/**
 * @brief A stable colour for a chunk's archetype.
 *
 * Cosmetic, so a hash is enough — but it must be STABLE, or two captures of the
 * same world would differ and the fold would stop meaning anything. Derived from
 * the archetype's component mask rather than from a pointer or an index, both of
 * which move between runs.
 */
core::u32 archetypeColour(const ecs::Archetype::Mask &mask) noexcept
{
    core::u32 hash = 0x811C9DC5u;
    for (std::size_t bit = 0u; bit < mask.size(); ++bit)
    {
        hash ^= mask.test(bit) ? 1u : 0u;
        hash *= 0x01000193u;
    }
    // Keep it bright enough to read against the background: a world rendered in
    // near-black tells a viewer nothing, which defeats the point of looking.
    const core::u32 r = 0x60u + (hash & 0x7Fu);
    const core::u32 g = 0x60u + ((hash >> 8u) & 0x7Fu);
    const core::u32 b = 0x60u + ((hash >> 16u) & 0x7Fu);
    return (r << 16u) | (g << 8u) | b;
}

/// One positioned, sized thing to draw.
struct Box {
    core::f32 cx{}, cy{}, cz{};
    core::f32 hx{}, hy{}, hz{};
    core::u32 colour{};
};

/// Collects every entity that has somewhere to be.
std::vector<Box> collectBoxes(const ecs::Registry &registry)
{
    std::vector<Box> boxes;
    for (const auto &part : registry.partitions())
    {
        if (!part)
            continue;
        // readComponent answers a pointer for EVERY component id, allocated or
        // not, so the archetype is what must be asked whether a component is
        // really there — the trap that made entities read garbage before.
        const bool hasAabb = part->archetype().has(ecs::ComponentId::AABB);
        const core::u32 colour = archetypeColour(part->archetype().mask());

        for (const auto &chunk : part->chunks())
        {
            if (!chunk)
                continue;
            const auto *positions =
                static_cast<const math::Vec3<math::Fixed32> *>(chunk->readComponent(ecs::ComponentId::Position));
            if (positions == nullptr || !part->archetype().has(ecs::ComponentId::Position))
                continue;
            const auto *extents =
                hasAabb ? static_cast<const math::Vec3<math::Fixed32> *>(chunk->readComponent(ecs::ComponentId::AABB)) :
                          nullptr;

            const core::u32 count = chunk->count();
            for (core::u32 i = 0u; i < count; ++i)
            {
                Box box;
                box.cx = positions[i].x.toFloat();
                box.cy = positions[i].y.toFloat();
                box.cz = positions[i].z.toFloat();
                box.hx = extents != nullptr ? extents[i].x.toFloat() : kDefaultHalfExtent;
                box.hy = extents != nullptr ? extents[i].y.toFloat() : kDefaultHalfExtent;
                box.hz = extents != nullptr ? extents[i].z.toFloat() : kDefaultHalfExtent;
                box.colour = colour;
                boxes.push_back(box);
            }
        }
    }
    return boxes;
}

/// Draws one axis-aligned box as six shaded quads.
core::u32 drawBox(const render::RenderTarget &rt, const math::Mat4<core::f32> &mvp, const Box &box)
{
    // Fixed per-face brightness rather than a light, and that is why this does not call
    // render::drawBox: the picture exists to be READ, and a consistent key light makes a
    // box legible at any orientation. What IS shared is the geometry — the same six faces
    // wound the same way — because two hand-written copies of a box disagree about winding
    // and then one of them culls its own front faces.
    //
    // The table is indexed by FACE ORDER, which render::forEachBoxFace documents as part of
    // its contract: top, bottom, +Z, -Z, +X, -X.
    constexpr core::f32 kKeyLight[6] = {1.00f, 0.35f, 0.78f, 0.60f, 0.70f, 0.50f};

    core::u32 triangles = 0u;
    core::u32 face = 0u;
    render::forEachBoxFace(box.cx - box.hx, box.cy - box.hy, box.cz - box.hz, box.cx + box.hx, box.cy + box.hy,
                           box.cz + box.hz, [&](const core::f32 *quad, core::f32, core::f32, core::f32) {
                               triangles += render::fillPolygonClipped(rt, mvp, quad, 4u,
                                                                       render::modulate(box.colour, kKeyLight[face]));
                               ++face;
                           });
    return triangles;
}

} // namespace

Screenshot renderWorld(const ecs::Registry &registry, core::u32 width, core::u32 height, const CameraPose &pose,
                       image::Image &out)
{
    Screenshot shot;
    shot.width = width;
    shot.height = height;
    if (width == 0u || height == 0u)
        return shot;

    std::vector<core::u32> colour(static_cast<std::size_t>(width) * height, 0u);
    std::vector<core::f32> depth(static_cast<std::size_t>(width) * height, 0.0f);
    const render::RenderTarget target{colour.data(), depth.data(), width, height};
    render::clearTarget(target, kBackground);

    const std::vector<Box> boxes = collectBoxes(registry);
    shot.entitiesDrawn = static_cast<core::u32>(boxes.size());

    if (!boxes.empty())
    {
        // Frame what is actually there. A fixed camera over a world whose extent
        // the caller did not state photographs the void just as happily as the
        // world, and a blank picture is the least useful observation there is.
        core::f32 minX = boxes[0].cx, maxX = boxes[0].cx;
        core::f32 minY = boxes[0].cy, maxY = boxes[0].cy;
        core::f32 minZ = boxes[0].cz, maxZ = boxes[0].cz;
        for (const Box &box : boxes)
        {
            minX = box.cx < minX ? box.cx : minX;
            maxX = box.cx > maxX ? box.cx : maxX;
            minY = box.cy < minY ? box.cy : minY;
            maxY = box.cy > maxY ? box.cy : maxY;
            minZ = box.cz < minZ ? box.cz : minZ;
            maxZ = box.cz > maxZ ? box.cz : maxZ;
        }
        const core::f32 spanX = maxX - minX;
        const core::f32 spanZ = maxZ - minZ;
        const core::f32 span = (spanX > spanZ ? spanX : spanZ) + 1.0f;

        render::OrbitCamera camera;
        camera.setFocus((minX + maxX) * 0.5f, (minZ + maxZ) * 0.5f);
        camera.setEyeHeight(0.0f);
        camera.setYaw(pose.yawDeg * kPi / 180.0f);
        camera.setPitch(pose.pitchDeg * kPi / 180.0f);
        camera.setDistance(pose.distance > 0.0f ? pose.distance : span * 1.4f);

        render::CameraBasis basis{};
        const core::f32 aspect = static_cast<core::f32>(width) / static_cast<core::f32>(height);
        // The far plane follows the framing, or a world larger than the default
        // 600 units would be clipped away by the camera meant to photograph it.
        const core::f32 farPlane = camera.distance() * 4.0f + span * 2.0f + 10.0f;
        const math::Mat4<core::f32> mvp =
            camera.viewProjection((minY + maxY) * 0.5f, aspect, render::CameraLens{1.04719755f, 0.4f, farPlane}, basis);

        for (const Box &box : boxes)
            shot.triangles += drawBox(target, mvp, box);
    }

    shot.fold = render::foldTarget(target);

    out.resize(width, height);
    for (core::u32 y = 0u; y < height; ++y)
        for (core::u32 x = 0u; x < width; ++x)
            out.set(static_cast<core::i32>(x), static_cast<core::i32>(y),
                    0xFF000000u | colour[static_cast<std::size_t>(y) * width + x]);
    return shot;
}

core::Expected<Screenshot> captureToFile(const ecs::Registry &registry, std::string_view path, core::u32 width,
                                         core::u32 height, const CameraPose &pose)
{
    image::Image picture;
    const Screenshot shot = renderWorld(registry, width, height, pose, picture);

    pmr::vector<core::u8> encoded;
    if (!image::writePpm(picture, encoded))
        return std::unexpected(
            core::makeError(core::ErrorCode::kSerializationFailed, lpl::pmr::string{"the frame did not encode"})
                .error());

    const std::string file{path};
    std::FILE *handle = std::fopen(file.c_str(), "wb");
    if (handle == nullptr)
        return std::unexpected(
            core::makeError(core::ErrorCode::kIoError, lpl::pmr::string{"cannot open " + file}).error());
    const std::size_t written = std::fwrite(encoded.data(), 1u, encoded.size(), handle);
    std::fclose(handle);
    if (written != encoded.size())
        return std::unexpected(
            core::makeError(core::ErrorCode::kIoError, lpl::pmr::string{"short write to " + file}).error());
    return shot;
}

} // namespace lpl::agent
