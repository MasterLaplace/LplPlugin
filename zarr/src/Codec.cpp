/**
 * @file Codec.cpp
 * @brief blosc and zstd, when the build has them.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @copyright MIT License
 */

#include <lpl/zarr/Codec.hpp>

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

core::usize decodeChunk([[maybe_unused]] Codec codec, [[maybe_unused]] const core::u8 *in,
                        [[maybe_unused]] core::usize size, [[maybe_unused]] core::u8 *out,
                        [[maybe_unused]] core::usize capacity) noexcept
{
    if (in == nullptr || out == nullptr || size == 0u || capacity == 0u)
        return 0u;

    switch (codec)
    {
    case Codec::Raw: return 0u; // Raw never reaches here: the caller reads it straight into place.

    case Codec::Blosc: {
#if defined(LPL_ZARR_HAS_BLOSC)
        // The blosc container names its own sub-codec in its header, so nothing here has to know
        // whether the writer chose zstd, lz4 or blosclz -- which is exactly why a corpus can
        // change its mind between volumes without breaking a reader.
        core::usize nbytes = 0;
        core::usize cbytes = 0;
        core::usize blocksize = 0;
        blosc_cbuffer_sizes(in, &nbytes, &cbytes, &blocksize);
        if (nbytes == 0u || nbytes > capacity)
            return 0u;
        const int produced = blosc_decompress_ctx(in, out, capacity, 1);
        return produced > 0 ? static_cast<core::usize>(produced) : 0u;
#else
        return 0u;
#endif
    }

    case Codec::Zstd: {
#if defined(LPL_ZARR_HAS_ZSTD)
        const core::usize produced = ZSTD_decompress(out, capacity, in, size);
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
