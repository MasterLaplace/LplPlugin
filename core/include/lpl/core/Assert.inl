#ifndef LPL_CORE_ASSERT_INL
#define LPL_CORE_ASSERT_INL

#if !LPL_TARGET_KERNEL

namespace lpl::core::detail {

inline void assertFail(const char *expr, std::source_location loc)
{
    std::fprintf(stderr, "[LPL ASSERT] %s:%u in %s — \"%s\" failed\n", loc.file_name(), loc.line(), loc.function_name(),
                 expr);
    std::abort();
}

} // namespace lpl::core::detail

#endif

#endif // LPL_CORE_ASSERT_INL
