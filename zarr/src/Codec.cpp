/**
 * @file Codec.cpp
 * @brief blosc and zstd, when the build has them.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/zarr/Codec.hpp>

#include <cstring>

#if defined(LPL_ZARR_HAS_BLOSC)
#    include <blosc.h>
#endif
#if defined(LPL_ZARR_HAS_ZSTD)
#    include <zstd.h>
#endif

namespace lpl::zarr {

bool codecAvailable(Codec codec) noexcept
{
    switch (codec)
    {
    case Codec::Raw: return true;
    case Codec::Blosc:
#if defined(LPL_ZARR_HAS_BLOSC)
        return true;
#else
        return false;
#endif
    case Codec::Zstd:
#if defined(LPL_ZARR_HAS_ZSTD)
        return true;
#else
        return false;
#endif
    case Codec::Unsupported:
    default: return false;
    }
}

const char *codecName(Codec codec) noexcept
{
    switch (codec)
    {
    case Codec::Raw: return "raw";
    case Codec::Blosc: return "blosc";
    case Codec::Zstd: return "zstd";
    case Codec::Unsupported:
    default: return "unsupported";
    }
}

core::usize decodeChunk(Codec codec, std::span<const core::u8> in, std::span<core::u8> out) noexcept
{
    if (in.empty() || out.empty())
        return 0u;

    switch (codec)
    {
    case Codec::Raw: {
        if (in.size() > out.size())
            return 0u;
        std::memcpy(out.data(), in.data(), in.size());
        return in.size();
    }

    case Codec::Blosc: {
#if defined(LPL_ZARR_HAS_BLOSC)
        // The blosc container names its own sub-codec in its header, so nothing here has to know
        // whether the writer chose zstd, lz4 or blosclz -- which is exactly why a corpus can
        // change its mind between volumes without breaking a reader.
        core::usize nbytes = 0u;
        // The decompressor reads as many bytes as the header announces and is never told how many
        // arrived, so a short download would decode whatever lies after it in the buffer.
        if (blosc_cbuffer_validate(in.data(), in.size(), &nbytes) != 0 || nbytes == 0u || nbytes > out.size())
            return 0u;
        constexpr int threads = 1;
        const int produced = blosc_decompress_ctx(in.data(), out.data(), out.size(), threads);
        return produced > 0 ? static_cast<core::usize>(produced) : 0u;
#else
        return 0u;
#endif
    }

    case Codec::Zstd: {
#if defined(LPL_ZARR_HAS_ZSTD)
        const core::usize produced = ZSTD_decompress(out.data(), out.size(), in.data(), in.size());
        return ZSTD_isError(produced) ? 0u : produced;
#else
        return 0u;
#endif
    }

    case Codec::Unsupported:
    default: return 0u;
    }
}

} // namespace lpl::zarr
