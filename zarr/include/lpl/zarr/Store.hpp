/**
 * @file Store.hpp
 * @brief Where the bytes of a chunk come from, without saying how they get here.
 *
 * @warning **Zarr is worth supporting on its own merits, not because one corpus uses it.** It is
 * the open standard for chunked N-dimensional arrays -- the format behind OME-NGFF bioimaging,
 * large microscopy atlases, climate and simulation output -- and every property it has is a
 * property a streaming volume renderer needs: a chunk is an independently addressable unit, the
 * multiscale pyramid is published rather than computed, the metadata is self-describing JSON, and
 * a "store" is nothing more than a map from key to bytes. That last point is the one this header
 * exists for: because a store is only a key-value map, a filesystem, an HTTP endpoint, an object
 * bucket and a zip file are all stores, and none of them belongs in the code that reads the format.
 *
 * @warning **The seam is here so the transport can live where its dependencies already are.** This
 * module parses and decodes and links nothing but a codec; whoever has a socket implements
 * @ref IZarrStore over it. Same reason `net::Endpoint` exists rather than a `sockaddr` crossing
 * module boundaries.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ZARR_STORE_HPP
#    define LPL_ZARR_STORE_HPP

#    include <lpl/core/Types.hpp>

namespace lpl::zarr {

/**
 * @enum FetchStatus
 * @brief What happened when a key was asked for.
 *
 * @warning **Absent and failed are different, and collapsing them is how a viewer reports an empty
 * subject.** A zarr array does not store chunks that are entirely fill value, so a missing key is
 * a normal, meaningful answer meaning "all fill". A transport error is not. A reader that treated
 * a timeout as absence would render a hole and say nothing.
 */
enum class FetchStatus : core::u8 {
    Ok = 0,   ///< Bytes were written.
    Absent,   ///< The store does not have this key: the chunk is entirely fill value.
    TooLarge, ///< The value does not fit the buffer offered.
    Failed,   ///< Transport or decode error. Not absence.
};

/**
 * @struct FetchResult
 * @brief Outcome of one key lookup.
 */
struct FetchResult final {
    FetchStatus status{FetchStatus::Failed};
    core::usize size{0}; ///< Bytes written on Ok; bytes needed on TooLarge.

    /**
     * The bytes are the array's fill value because the store had no such key.
     *
     * @warning **Successful and yet not fetched, and a caller has to be able to tell.** Filling an
     * absent chunk is the correct reading of the format, so the read succeeds -- but "this region
     * is empty" and "this region was downloaded" are different facts about a run, and a streamer
     * that could not separate them would report a sparse volume and a stalled link identically.
     */
    bool filled{false};

    [[nodiscard]] constexpr bool ok() const noexcept { return status == FetchStatus::Ok; }
};

/**
 * @class IZarrStore
 * @brief A map from key to bytes.
 *
 * @warning The buffer is the caller's, on purpose. A store that allocated would put an allocator on
 * the hot path of a streamer whose entire design is a bounded, reused resident set.
 */
class IZarrStore {
public:
    virtual ~IZarrStore() = default;

    /**
     * @brief Reads the value for @p key into @p buffer.
     * @param key      Store key, for example "0/12/3/4" or ".zarray".
     * @param buffer   Destination.
     * @param capacity Bytes available.
     */
    [[nodiscard]] virtual FetchResult read(const char *key, core::u8 *buffer, core::usize capacity) noexcept = 0;

    /// @brief Human-readable name of the transport, for logs that have to say where bytes failed.
    [[nodiscard]] virtual const char *name() const noexcept = 0;
};

} // namespace lpl::zarr

#endif // LPL_ZARR_STORE_HPP
