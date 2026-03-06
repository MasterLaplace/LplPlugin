/**
 * @file ServerMonitorSystem.hpp
 * @brief ECS system that logs server performance metrics periodically.
 *
 * Runs in the Network phase. Logs physics timing, entity counts,
 * chunk counts, and client counts every N frames.
 *
 * Server-side only.
 *
 * @author MasterLaplace
 * @version 0.2.0
 * @date 2026-02-27
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ENGINE_SYSTEMS_SERVERMONITORSYSTEM_HPP
#    define LPL_ENGINE_SYSTEMS_SERVERMONITORSYSTEM_HPP

#    include <lpl/ecs/System.hpp>
#    include <lpl/ecs/WorldPartition.hpp>
#    include <lpl/net/session/SessionManager.hpp>

#    include <memory>

namespace lpl::engine::systems {

/**
 * @class ServerMonitorSystem
 * @brief Periodic server performance logging (server).
 */
class ServerMonitorSystem final : public ecs::ISystem {
public:
    /**
     * @param sessionManager Session registry for client count.
     * @param world          World partition for entity/chunk counts.
     * @param logInterval    Frames between log outputs (default: 300).
     */
    ServerMonitorSystem(net::session::SessionManager &sessionManager, ecs::WorldPartition &world,
                        core::u32 logInterval = 300);
    ~ServerMonitorSystem() override;

    [[nodiscard]] const ecs::SystemDescriptor &descriptor() const noexcept override;
    void execute(core::f32 dt) override;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace lpl::engine::systems

#endif // LPL_ENGINE_SYSTEMS_SERVERMONITORSYSTEM_HPP
