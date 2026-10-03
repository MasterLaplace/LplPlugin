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
-- The compressed paths are optional: a codec the system does not provide is
-- refused at open time, never approximated (Codec.hpp says why).
-- /////////////////////////////////////////////////////////////////////////////

add_requires("pkgconfig::blosc", { optional = true })
add_requires("pkgconfig::libzstd", { optional = true })

target("lpl-zarr")
    set_kind("static")
    set_group("modules")
    add_deps("lpl-core")
    add_includedirs("include", { public = true })
    add_files("src/**.cpp")
    add_headerfiles("include/(lpl/zarr/**.hpp)")
    add_packages("pkgconfig::blosc", "pkgconfig::libzstd", { public = true })

    on_config(function (target)
        if target:pkg("pkgconfig::blosc") then
            target:add("defines", "LPL_ZARR_HAS_BLOSC")
        end
        if target:pkg("pkgconfig::libzstd") then
            target:add("defines", "LPL_ZARR_HAS_ZSTD")
        end
    end)
target_end()
