/**
 * @file Config.hpp
 * @brief Engine configuration (Builder pattern).
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */
#pragma once

#ifndef LPL_ENGINE_CONFIG_HPP
#    define LPL_ENGINE_CONFIG_HPP

#    include <lpl/core/Constants.hpp>
#    include <lpl/core/Types.hpp>
#    include <lpl/ecs/WorldPartition.hpp>
#    include <lpl/std/string.hpp>

namespace lpl::engine {

/** @brief Immutable engine configuration. */
class Config {
public:
    /** @brief Fluent builder for Config. */
    class Builder {
    public:
        Builder &tickRate(core::u32 hz) noexcept;
        Builder &maxEntities(core::u32 n) noexcept;
        Builder &maxChunks(core::u32 n) noexcept;
        Builder &worldCellCapacity(core::u32 n) noexcept;
        Builder &serverMode(bool enabled) noexcept;
        Builder &headless(bool enabled) noexcept;
        Builder &arenaSize(core::usize bytes) noexcept;
        Builder &worldArenaSize(core::usize bytes) noexcept;
        Builder &enableBci(bool enabled) noexcept;
        Builder &enablePhysics(bool enabled) noexcept;
        Builder &enableNetworking(bool enabled) noexcept;
        Builder &enableRendering(bool enabled) noexcept;
        Builder &enableRealTimeGuard(bool enabled) noexcept;
        Builder &enableGpu(bool enabled) noexcept;
        Builder &serverAddress(pmr::string addr) noexcept;
        Builder &serverPort(core::u16 port) noexcept;

        [[nodiscard]] Config build() const noexcept;

    private:
        core::u32 _tickRate{core::kTickRate};
        core::u32 _maxEntities{core::kMaxEntities};
        core::u32 _maxChunks{core::kMaxChunks};
        core::u32 _worldCellCapacity{ecs::WorldPartition::kDefaultCellCapacity};
        bool _serverMode{false};
        bool _headless{false};
        core::usize _arenaSize{64 * 1024 * 1024};
        core::usize _worldArenaSize{64 * 1024 * 1024};
        bool _enableBci{false};
        bool _enableGpu{false};
        bool _enablePhysics{true};
        bool _enableNetworking{true};
        bool _enableRendering{true};
        bool _enableRealTimeGuard{false};
        pmr::string _serverAddress{"127.0.0.1"};
        core::u16 _serverPort{4242};
    };

    [[nodiscard]] core::u32 tickRate() const noexcept { return _tickRate; }
    [[nodiscard]] core::u32 maxEntities() const noexcept { return _maxEntities; }
    [[nodiscard]] core::u32 maxChunks() const noexcept { return _maxChunks; }

    /**
     * @brief Returns the number of spatial cells budgeted by WorldPartition.
     * @return Number of spatial cells budgeted by WorldPartition.
     */
    [[nodiscard]] core::u32 worldCellCapacity() const noexcept { return _worldCellCapacity; }
    [[nodiscard]] bool serverMode() const noexcept { return _serverMode; }
    [[nodiscard]] bool headless() const noexcept { return _headless; }

    /**
     * @brief Per-frame scratch arena; reset every frame.
     * @return Size in bytes of the per-frame scratch arena.
     */
    [[nodiscard]] core::usize arenaSize() const noexcept { return _arenaSize; }

    /**
     * @brief Persistent arena backing the World's ECS storage. NEVER reset while the
     *        World lives — chunks allocated from it must outlive the frame.
     * @return Size in bytes of the persistent arena backing the World's ECS storage.
     */
    [[nodiscard]] core::usize worldArenaSize() const noexcept { return _worldArenaSize; }
    [[nodiscard]] bool enableBci() const noexcept { return _enableBci; }

    /**
     * @brief Tests whether physics is enabled.
     * @return True if physics is enabled, false otherwise.
     */
    [[nodiscard]] bool enablePhysics() const noexcept { return _enablePhysics; }
    /**
     * @brief Tests whether networking is enabled.
     * @return True if networking is enabled, false otherwise.
     */
    [[nodiscard]] bool enableNetworking() const noexcept { return _enableNetworking; }
    /**
     * @brief Tests whether rendering is enabled.
     * @return True if rendering is enabled, false otherwise.
     */
    [[nodiscard]] bool enableRendering() const noexcept { return _enableRendering; }

    /**
     * @brief Runs the authoritative tick inside a platform real-time section, where a
     *        backend that enforces it makes heap allocation FAIL. Off by default: turn
     *        it on only once the tick is known to be allocation-free, or the first
     *        allocation takes the process (or the kernel) down.
     * @return True if the real-time guard is enabled, false otherwise.
     */
    [[nodiscard]] bool enableRealTimeGuard() const noexcept { return _enableRealTimeGuard; }
    [[nodiscard]] bool enableGpu() const noexcept { return _enableGpu; }
    [[nodiscard]] const pmr::string &serverAddress() const noexcept { return _serverAddress; }
    [[nodiscard]] core::u16 serverPort() const noexcept { return _serverPort; }

private:
    friend class Builder;

    core::u32 _tickRate{core::kTickRate};
    core::u32 _maxEntities{core::kMaxEntities};
    core::u32 _maxChunks{core::kMaxChunks};
    core::u32 _worldCellCapacity{ecs::WorldPartition::kDefaultCellCapacity};
    bool _serverMode{false};
    bool _headless{false};
    core::usize _arenaSize{64 * 1024 * 1024};
    core::usize _worldArenaSize{64 * 1024 * 1024};
    bool _enableBci{false};
    bool _enableGpu{false};
    bool _enablePhysics{true};
    bool _enableNetworking{true};
    bool _enableRendering{true};
    bool _enableRealTimeGuard{false};
    pmr::string _serverAddress{"127.0.0.1"};
    core::u16 _serverPort{4242};
};

} // namespace lpl::engine

#endif // LPL_ENGINE_CONFIG_HPP
