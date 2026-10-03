/**
 * @file FileStore.cpp
 * @brief Keys as paths.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/zarr/FileStore.hpp>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <memory>
#include <sys/stat.h>
#include <sys/types.h>

namespace lpl::zarr {

namespace {

constexpr core::usize kMaxPathLength = 1024u;
constexpr char kPartialSuffix[] = ".part";

struct FileCloser final {
    void operator()(std::FILE *file) const noexcept { std::fclose(file); }
};

using File = std::unique_ptr<std::FILE, FileCloser>;

} // namespace

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

FetchResult FileStore::read(const char *key, std::span<core::u8> buffer) noexcept
{
    char path[kMaxPathLength];
    if (!pathFor(key, path, sizeof(path)))
        return FetchResult{FetchStatus::Failed, 0u};

    const File file{std::fopen(path, "rb")};
    if (file == nullptr)
        return FetchResult{errno == ENOENT ? FetchStatus::Absent : FetchStatus::Failed, 0u};
    struct stat info {};
    if (::fstat(::fileno(file.get()), &info) != 0 || !S_ISREG(info.st_mode))
        return FetchResult{FetchStatus::Failed, 0u};
    const auto bytes = static_cast<core::usize>(info.st_size);
    if (bytes > buffer.size())
        return FetchResult{FetchStatus::TooLarge, bytes};

    const core::usize got = std::fread(buffer.data(), 1u, bytes, file.get());
    if (got != bytes)
        return FetchResult{FetchStatus::Failed, got};
    return FetchResult{FetchStatus::Ok, got};
}

bool FileStore::write(const char *key, std::span<const core::u8> bytes) noexcept
{
    char path[kMaxPathLength];
    if (!pathFor(key, path, sizeof(path)))
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
    char partial[kMaxPathLength + sizeof(kPartialSuffix)];
    const int n = std::snprintf(partial, sizeof(partial), "%s%s", path, kPartialSuffix);
    if (n <= 0 || static_cast<core::usize>(n) >= sizeof(partial))
        return false;

    File file{std::fopen(partial, "wb")};
    if (file == nullptr)
        return false;
    const core::usize put = std::fwrite(bytes.data(), 1u, bytes.size(), file.get());
    const bool closed = std::fclose(file.release()) == 0;
    if (put != bytes.size() || !closed)
    {
        std::remove(partial);
        return false;
    }
    return std::rename(partial, path) == 0;
}

} // namespace lpl::zarr
