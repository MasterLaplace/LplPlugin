#include <cmath>
#include <lpl/ecs/Partition.hpp>
#include <lpl/ecs/Registry.hpp>
#include <lpl/engine/systems/CameraSystem.hpp>

#ifdef LPL_HAS_RENDERER
#    include <GLFW/glfw3.h>
#endif

namespace lpl::engine::systems {

static const ecs::ComponentAccess kCameraAccesses[] = {
    {ecs::ComponentId::Position, ecs::AccessMode::ReadOnly},
};

static const ecs::SystemDescriptor kCameraSystemDesc{"CameraSystem", ecs::SchedulePhase::Render,
                                                     std::span<const ecs::ComponentAccess>{kCameraAccesses}};

static void updateCameraDirection(CameraData &cam)
{
    float frontX = std::cos(cam.yaw * (3.14159f / 180.f)) * std::cos(cam.pitch * (3.14159f / 180.f));
    float frontY = std::sin(cam.pitch * (3.14159f / 180.f));
    float frontZ = std::sin(cam.yaw * (3.14159f / 180.f)) * std::cos(cam.pitch * (3.14159f / 180.f));

    float len = std::sqrt(frontX * frontX + frontY * frontY + frontZ * frontZ);
    if (len > 0.0f)
    {
        cam.front.x = frontX / len;
        cam.front.y = frontY / len;
        cam.front.z = frontZ / len;
    }
}

static bool getEntityPosition(ecs::Registry &registry, core::u32 id, math::Vec3<float> &outPos)
{
    auto refResult = registry.resolve(ecs::EntityId{id});
    if (!refResult.has_value())
        return false;

    auto ref = refResult.value();
    const auto &partitions = registry.partitions();

    // Find the partition that owns this entity
    for (const auto &part : partitions)
    {
        if (!part)
            continue;
        const auto &chunks = part->chunks();
        if (ref.chunkIndex >= static_cast<core::u32>(chunks.size()))
            continue;

        auto &chunk = *chunks[ref.chunkIndex];
        auto entityIds = chunk.entities();

        for (core::u32 i = 0; i < chunk.count(); ++i)
        {
            if (entityIds[i].raw() == id)
            {
                if (auto *rpos =
                        static_cast<const math::Vec3<float> *>(chunk.readComponent(ecs::ComponentId::Position)))
                {
                    outPos = rpos[i];
                    return true;
                }
                break;
            }
        }
    }
    return false;
}

CameraSystem::CameraSystem(CameraData &cameraData, ecs::Registry &registry, GLFWwindow *window,
                           const core::u32 &myEntityId, const bool &connected)
    : _cameraData(cameraData), _registry(registry), _window(window), _myEntityId(myEntityId), _connected(connected)
{
}

const ecs::SystemDescriptor &CameraSystem::descriptor() const noexcept { return kCameraSystemDesc; }

void CameraSystem::execute(core::f32 dt)
{
#ifdef LPL_HAS_RENDERER
    if (!_window)
        return;

    constexpr float ROT_SPEED = 60.f;

    if (glfwGetKey(_window, GLFW_KEY_LEFT) == GLFW_PRESS)
    {
        _cameraData.yaw -= ROT_SPEED * dt;
        updateCameraDirection(_cameraData);
    }
    if (glfwGetKey(_window, GLFW_KEY_RIGHT) == GLFW_PRESS)
    {
        _cameraData.yaw += ROT_SPEED * dt;
        updateCameraDirection(_cameraData);
    }
    if (glfwGetKey(_window, GLFW_KEY_UP) == GLFW_PRESS)
    {
        _cameraData.pitch += ROT_SPEED * dt;
        if (_cameraData.pitch > 89.f)
            _cameraData.pitch = 89.f;
        updateCameraDirection(_cameraData);
    }
    if (glfwGetKey(_window, GLFW_KEY_DOWN) == GLFW_PRESS)
    {
        _cameraData.pitch -= ROT_SPEED * dt;
        if (_cameraData.pitch < -89.f)
            _cameraData.pitch = -89.f;
        updateCameraDirection(_cameraData);
    }

    if (_connected && _myEntityId != 0)
    {
        math::Vec3<float> pos;
        if (getEntityPosition(_registry, _myEntityId, pos))
        {
            _cameraData.position.x = pos.x - _cameraData.front.x * 60.f;
            _cameraData.position.y = pos.y - _cameraData.front.y * 60.f + 30.f;
            _cameraData.position.z = pos.z - _cameraData.front.z * 60.f;
            return;
        }
    }

    // Free cam
    math::Vec3<float> movement{0.f, 0.f, 0.f};
    if (glfwGetKey(_window, GLFW_KEY_W) == GLFW_PRESS)
    {
        movement.x += _cameraData.front.x;
        movement.y += _cameraData.front.y;
        movement.z += _cameraData.front.z;
    }
    if (glfwGetKey(_window, GLFW_KEY_S) == GLFW_PRESS)
    {
        movement.x -= _cameraData.front.x;
        movement.y -= _cameraData.front.y;
        movement.z -= _cameraData.front.z;
    }

    if (movement.x != 0.f || movement.y != 0.f || movement.z != 0.f)
    {
        float speed = _cameraData.speed * dt;
        _cameraData.position.x += movement.x * speed;
        _cameraData.position.y += movement.y * speed;
        _cameraData.position.z += movement.z * speed;
    }
#endif
}

} // namespace lpl::engine::systems
