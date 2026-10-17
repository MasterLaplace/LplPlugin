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
    /**
     * @param root Directory the keys are relative to, copied. A null, empty, or 510-byte-or-longer
     * root leaves the store invalid (@ref valid).
     *
     * The copy ends with a separator, so that every path below is one concatenation rather than a
     * conditional: a conditional separator is how a cache ends up with two spellings of the same key.
     */
    explicit FileStore(const char *root) noexcept;

    /**
     * @brief Absent only when no file exists at the key. A directory, or a file that cannot be opened, is Failed.
     */
    [[nodiscard]] FetchResult read(const char *key, std::span<core::u8> buffer) noexcept override;
    [[nodiscard]] const char *name() const noexcept override { return "FileStore"; }

    /**
     * @return Whether the store has a root to read and write under.
     */
    [[nodiscard]] bool valid() const noexcept { return _root[0] != '\0'; }

    /**
     * @brief Writes a value, creating parent directories.
     *
     * A store that can only read cannot be a cache, and a cache written by a separate code path
     * would be free to disagree with the reader about where a key lives.
     *
     * Every parent is created, INCLUDING the root: a cache directory that does not exist yet is the
     * normal state on a first run, and a store that only created directories below the root would
     * fail on exactly that. The walk starts after the leading separator, so that an absolute path does
     * not try to create "".
     *
     * The value is written beside its key, then renamed onto it: a reader that finds a half-written
     * chunk has no way to tell it from a short one, and rename is the only step that is atomic.
     *
     * @return Whether the value is now at @p key. A reader sees the old value or the new one, never
     * part of one; on false, a value already at @p key is untouched.
     */
    [[nodiscard]] bool write(const char *key, std::span<const core::u8> bytes) noexcept;

private:
    /**
     * @brief Writes the path of @p key under the root into @p out.
     *
     * A key that climbs out of the root is refused. Keys come from metadata, metadata can come from a
     * network, and a store that resolves ".." writes wherever the document says.
     */
    [[nodiscard]] bool pathFor(const char *key, char *out, core::usize capacity) const noexcept;

    char _root[512]{};
};

} // namespace lpl::zarr

#endif // LPL_ZARR_FILESTORE_HPP
