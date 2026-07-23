/**
 * @file StateHashReportSystem.hpp
 * @brief Client-side desync reporting: hash our world, tell the server.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-07-22
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ENGINE_SYSTEMS_STATEHASHREPORTSYSTEM_HPP
#    define LPL_ENGINE_SYSTEMS_STATEHASHREPORTSYSTEM_HPP

#    include <lpl/core/Platform.hpp>

#    ifdef LPL_HAS_NET

#        include <lpl/core/Types.hpp>
#        include <lpl/ecs/System.hpp>

#        include <memory>

namespace lpl::net::transport {
class ITransport;
}

namespace lpl::engine {
class World;
}

namespace lpl::engine::systems {

/**
 * @class StateHashReportSystem
 * @brief Periodically sends this client's state digest to the server (§6.4).
 *
 * The client folds its own authoritative state and reports the digest with the
 * tick it belongs to; the server looks that tick up in its history and answers
 * Match or Diverged. Reporting every tick would be wasteful for a 16-byte
 * payload's worth of information, so it goes out every @c interval ticks.
 */
class StateHashReportSystem final : public ecs::ISystem {
public:
    /**
     * @param world     The instance whose state is hashed.
     * @param transport Transport to the server.
     * @param connected Set once the handshake completed; nothing is sent before.
     * @param interval  Ticks between reports.
     */
    StateHashReportSystem(const World &world, std::shared_ptr<net::transport::ITransport> transport,
                          const bool &connected, core::u32 interval = 60);
    ~StateHashReportSystem() override;

    [[nodiscard]] const ecs::SystemDescriptor &descriptor() const noexcept override;
    void execute(core::f32 dt) override;

    /** @brief Ticks this system has counted; the tick a report refers to. */
    [[nodiscard]] core::u64 currentTick() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace lpl::engine::systems

#    endif // LPL_HAS_NET

#endif // LPL_ENGINE_SYSTEMS_STATEHASHREPORTSYSTEM_HPP
