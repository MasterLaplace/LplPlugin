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
        core::usize nbytes = 0u;
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
