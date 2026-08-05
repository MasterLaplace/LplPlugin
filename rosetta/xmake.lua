-- /////////////////////////////////////////////////////////////////////////////
-- @file xmake.lua
-- @brief Build configuration for the lpl::rosetta module.
-- rosetta/ build configuration — a format that carries the specification of its own reader.
-- The archival end of 'store the generator, not the result'. A cartridge assumes
-- a reader compiled from this repository; a Rosetta artifact assumes nothing but
-- an observer with a microscope. Ten opcodes, an interpreter for them, and a
-- decompressor written in that instruction set — enough for someone with none of
-- our machines to rebuild the reader and open the archive.
-- /////////////////////////////////////////////////////////////////////////////

target("lpl-rosetta")
    set_kind("static")
    set_group("modules")
    add_deps("lpl-core", "lpl-codec")
    add_includedirs("include", { public = true })
    add_files("src/**.cpp")
    add_headerfiles("include/(lpl/rosetta/**.hpp)")
target_end()
