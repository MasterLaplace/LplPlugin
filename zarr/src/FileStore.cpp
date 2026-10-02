/**
 * @file FileStore.cpp
 * @brief Keys as paths.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/zarr/FileStore.hpp>

#include <cstdio>
#include <cstring>
#include <sys/stat.h>
#include <sys/types.h>

namespace lpl::zarr {

FileStore::FileStore(const char *root) noexcept
{
    if (root == nullptr)
        return;
    const core::usize n = std::strlen(root);
    if (n == 0u || n + 2u >= sizeof(_root))
        return;
    std::memcpy(_root, root, n);
    // A trailing separator here means every path below is one concatenation rather than a
    // conditional, and a conditional separator is how a cache ends up with two spellings of the
    // same key.
    _root[n] = (root[n - 1u] == '/') ? '\0' : '/';
    _root[n + 1u] = '\0';
}

bool FileStore::pathFor(const char *key, char *out, core::usize capacity) const noexcept
{
    if (key == nullptr || !valid())
        return false;
    // A key that climbs out of the root is refused. Keys come from metadata, metadata can come
    // from a network, and a store that resolves ".." writes wherever the document says.
    for (const char *p = key; *p != '\0'; ++p)
    {
        if (p[0] == '.' && p[1] == '.')
            return false;
    }
    const core::usize r = std::strlen(_root);
    const core::usize k = std::strlen(key);
    if (r + k + 1u > capacity)
        return false;
    std::memcpy(out, _root, r);
    std::memcpy(out + r, key, k);
    out[r + k] = '\0';
    return true;
}

FetchResult FileStore::read(const char *key, core::u8 *buffer, core::usize capacity) noexcept
{
    char path[1024];
    if (!pathFor(key, path, sizeof(path)))
        return FetchResult{FetchStatus::Failed, 0u};

    std::FILE *f = std::fopen(path, "rb");
    if (f == nullptr)
        return FetchResult{FetchStatus::Absent, 0u};

    if (std::fseek(f, 0, SEEK_END) != 0)
    {
        std::fclose(f);
        return FetchResult{FetchStatus::Failed, 0u};
    }
    const long size = std::ftell(f);
    std::rewind(f);
    if (size < 0)
    {
        std::fclose(f);
        return FetchResult{FetchStatus::Failed, 0u};
    }
    if (static_cast<core::usize>(size) > capacity)
    {
        std::fclose(f);
        return FetchResult{FetchStatus::TooLarge, static_cast<core::usize>(size)};
    }
    const core::usize got = std::fread(buffer, 1u, static_cast<core::usize>(size), f);
    std::fclose(f);
    if (got != static_cast<core::usize>(size))
        return FetchResult{FetchStatus::Failed, got};
    return FetchResult{FetchStatus::Ok, got};
}

bool FileStore::write(const char *key, const core::u8 *bytes, core::usize size) noexcept
{
    char path[1024];
    if (!pathFor(key, path, sizeof(path)) || bytes == nullptr)
        return false;

    // Create every parent, INCLUDING the root: a cache directory that does not exist yet is the
    // normal state on a first run, and a store that only created directories below the root would
    // fail on exactly that. Start after the leading separator so an absolute path does not try to
    // create "".
    for (char *p = path + 1; *p != '\0'; ++p)
    {
        if (*p != '/')
            continue;
        *p = '\0';
        ::mkdir(path, 0755);
        *p = '/';
    }

    // Write beside, then rename. A reader that finds a half-written chunk has no way to tell it
    // from a short one, and rename is the only step that is atomic.
    char tmp[1088];
    const int n = std::snprintf(tmp, sizeof(tmp), "%s.part", path);
    if (n <= 0 || static_cast<core::usize>(n) >= sizeof(tmp))
        return false;

    std::FILE *f = std::fopen(tmp, "wb");
    if (f == nullptr)
        return false;
    const core::usize put = std::fwrite(bytes, 1u, size, f);
    const bool closed = std::fclose(f) == 0;
    if (put != size || !closed)
    {
        std::remove(tmp);
        return false;
    }
    return std::rename(tmp, path) == 0;
}

} // namespace lpl::zarr
