-- /////////////////////////////////////////////////////////////////////////////
-- /// @file xmake.lua
-- /// @brief Build configuration for the lpl::platform module.
-- ///
-- /// Pure platform-seam interfaces (IPlatform + backend strategies) plus their
-- /// hosted implementations. The kernel backends (src/kernel) compile to nothing
-- /// here (guarded by LPL_TARGET_KERNEL); only the Linux backends are built into
-- /// the hosted oracle. libengine builds the kernel side from the same sources.
-- /////////////////////////////////////////////////////////////////////////////

target("lpl-platform")
    set_kind("static")
    set_group("modules")
    add_deps("lpl-core")
    add_headerfiles("include/(lpl/platform/**.hpp)")
    add_includedirs("include", {public = true})
    add_files("src/**.cpp")
