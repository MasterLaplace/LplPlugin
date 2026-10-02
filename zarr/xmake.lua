-- /////////////////////////////////////////////////////////////////////////////
-- @file xmake.lua
-- @brief Build configuration for the lpl::zarr module.
--
-- zarr/ reads the open standard for chunked N-dimensional arrays: the format
-- behind OME-NGFF bioimaging, microscopy atlases and simulation output. It
-- parses, it decodes, and it knows nothing about transport -- a store is a
-- key-to-bytes map, so a filesystem, an HTTP endpoint and an object bucket are
-- all stores and none of them belongs in here.
--
-- The compressed paths are optional and REFUSED rather than approximated when
-- absent: handing compressed bytes to the raw path yields plausible garbage,
-- and garbage renders.
-- /////////////////////////////////////////////////////////////////////////////

target("lpl-zarr")
    set_kind("static")
    set_group("modules")
    add_deps("lpl-core")
    add_includedirs("include", { public = true })
    add_files("src/**.cpp")
    add_headerfiles("include/(lpl/zarr/**.hpp)")

    if os.isfile("/usr/include/blosc.h") then
        add_defines("LPL_ZARR_HAS_BLOSC")
        add_syslinks("blosc", { public = true })
    end
    if os.isfile("/usr/include/zstd.h") then
        add_defines("LPL_ZARR_HAS_ZSTD")
        add_syslinks("zstd", { public = true })
    end
target_end()
