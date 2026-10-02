/**
 * @file FileStore.hpp
 * @brief A zarr store backed by a directory, which is also what a local cache is.
 *
 * @warning **This is the store a cache uses, and a cache is not a compromise.** Pages backed by a
 * file are RAM -- read at memory speed, zero copy -- and unlike an anonymous cache they are
 * reclaimable, so a working set larger than memory degrades instead of being killed. The
 * alternative, holding everything in the process, is the arrangement that runs out.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_ZARR_FILESTORE_HPP
#    define LPL_ZARR_FILESTORE_HPP

#    include <lpl/zarr/Store.hpp>

namespace lpl::zarr {

/**
 * @class FileStore
 * @brief Reads keys as paths under a root directory.
 */
class FileStore final : public IZarrStore {
public:
    /// @param root Directory the keys are relative to. Copied; must fit.
    explicit FileStore(const char *root) noexcept;

    [[nodiscard]] FetchResult read(const char *key, core::u8 *buffer, core::usize capacity) noexcept override;
    [[nodiscard]] const char *name() const noexcept override { return "FileStore"; }

    [[nodiscard]] bool valid() const noexcept { return _root[0] != '\0'; }

    /**
     * @brief Writes a value, creating parent directories.
     *
     * A store that can only read cannot be a cache, and a cache written by a separate code path
     * would be free to disagree with the reader about where a key lives.
     */
    [[nodiscard]] bool write(const char *key, const core::u8 *bytes, core::usize size) noexcept;

private:
    [[nodiscard]] bool pathFor(const char *key, char *out, core::usize capacity) const noexcept;

    char _root[512]{};
};

} // namespace lpl::zarr

#endif // LPL_ZARR_FILESTORE_HPP
