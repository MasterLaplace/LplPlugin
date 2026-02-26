// /////////////////////////////////////////////////////////////////////////////
/// @file StateSnapshot.hpp
/// @brief Complete world-state snapshot (Memento pattern).
///
/// Captures/restores the full deterministic simulation state
/// for rollback, replay, or save-game functionality.
// /////////////////////////////////////////////////////////////////////////////
#pragma once

#include <lpl/serial/ISerializable.hpp>
#include <lpl/math/StateHash.hpp>
#include <lpl/core/Types.hpp>
#include <vector>

namespace lpl::serial {

/// @brief Per-entity state blob within a snapshot.
struct EntityBlob
{
    core::u32 entityId{0};
    std::vector<core::byte> data;
};

/// @brief Full world snapshot at a given tick.
///
/// Memento pattern: encapsulates all simulation state needed
/// to restore or compare a game tick deterministically.
class StateSnapshot : public ISerializable
{
public:
    StateSnapshot();
    ~StateSnapshot() override;

    /// @brief Tick number this snapshot represents.
    [[nodiscard]] core::u64 tick() const noexcept;
    void setTick(core::u64 tick) noexcept;

    /// @brief StateHash (FNV-1a) of the full snapshot payload.
    [[nodiscard]] core::u64 hash() const noexcept;

    /// @brief Add an entity's serialised component data.
    void addEntityBlob(core::u32 entityId,
                       const core::byte* data, core::usize size);

    /// @brief Number of entity blobs in this snapshot.
    [[nodiscard]] core::usize entityCount() const noexcept;

    /// @brief Access entity blob by index.
    [[nodiscard]] const EntityBlob& blob(core::usize index) const;

    /// @brief Clear all data (reuse allocation).
    void clear() noexcept;

    /// @brief Recompute the hash from current payload.
    void rehash();

    // ISerializable ──────────────────────────────────────────────────────────
    [[nodiscard]] core::Expected<void> serialize(
        net::protocol::Bitstream& stream) const override;
    [[nodiscard]] core::Expected<void> deserialize(
        net::protocol::Bitstream& stream) override;

private:
    core::u64 tick_{0};
    core::u64 hash_{0};
    std::vector<EntityBlob> blobs_;
};

} // namespace lpl::serial
