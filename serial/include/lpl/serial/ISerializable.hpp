/**
 * @file ISerializable.hpp
 * @brief Abstract serialization interface.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */
#pragma once

#ifndef LPL_SERIAL_ISERIALIZABLE_HPP
    #define LPL_SERIAL_ISERIALIZABLE_HPP

#include <lpl/core/Types.hpp>
#include <lpl/core/Expected.hpp>

namespace lpl::net::protocol { class Bitstream; }

namespace lpl::serial {

/** @brief Interface for types that support binary serialization. */
class ISerializable
{
public:
    virtual ~ISerializable() = default;

    /**
     * @brief Serialize this object into the bitstream.
     * @param stream Output bitstream.
     * @return Success or error.
     */
    [[nodiscard]] virtual core::Expected<void> serialize(
        net::protocol::Bitstream& stream) const = 0;

    /**
     * @brief Deserialize this object from the bitstream.
     * @param stream Input bitstream.
     * @return Success or error.
     */
    [[nodiscard]] virtual core::Expected<void> deserialize(
        net::protocol::Bitstream& stream) = 0;

    /** @brief Size in bytes when serialized (0 = variable). */
    [[nodiscard]] virtual core::usize serializedSize() const noexcept { return 0; }
};

} // namespace lpl::serial

#endif // LPL_SERIAL_ISERIALIZABLE_HPP
