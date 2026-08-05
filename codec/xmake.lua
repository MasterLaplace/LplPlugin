-- /////////////////////////////////////////////////////////////////////////////
-- @file xmake.lua
-- @brief Build configuration for the lpl::codec module.
-- codec/ build configuration — erasure coding and the algebra it needs.
-- Integer-only by construction (XOR, AND, table lookups, shifts): no float, no
-- libm, so it links in ring 0 without a single derogation to the determinism
-- contract. Three consumers share one engine — net/ for UDP erasure coding under
-- jitter, pack/ for cartridges that survive a bad sector, and the DNA archival
-- research. The host path may use SIMD; the i686 path takes the scalar kernel and
-- BOTH must fold the same signature.
-- /////////////////////////////////////////////////////////////////////////////

target("lpl-codec")
    set_kind("static")
    set_group("modules")
    add_deps("lpl-core", "lpl-math")
    add_includedirs("include", { public = true })
    add_files("src/**.cpp")
    add_headerfiles("include/(lpl/codec/**.hpp)")
target_end()
