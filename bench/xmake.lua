-- /////////////////////////////////////////////////////////////////////////////
-- @file xmake.lua
-- @brief Build configuration for the lpl::bench module (micro-benchmark harness
--        and host introspection).
-- /////////////////////////////////////////////////////////////////////////////

target("lpl-bench")
    set_kind("static")
    set_group("modules")
    add_deps("lpl-core", "lpl-math", "lpl-voxel")
    add_headerfiles("include/(lpl/bench/*.hpp)")
    add_includedirs("include", {public = true})
    add_files("src/*.cpp")

    -- The commit this module is built from, with -dirty when tracked files differ from it,
    -- stamped as LPLPLUGIN_COMMIT into SystemInfo.cpp alone, the file that reports it, so a
    -- new commit recompiles that file and nothing else. LplKernel stamps its identity the
    -- same way.
    on_load(function (target)
        local function git(arguments)
            local output = try { function ()
                return os.iorunv("git", table.join({"-C", target:scriptdir()}, arguments))
            end }
            return output and output:trim() or ""
        end
        local commit = git({"rev-parse", "--short=7", "HEAD"})
        if commit == "" then
            commit = "unknown"
        elseif git({"status", "--porcelain", "--untracked-files=no"}) ~= "" then
            commit = commit .. "-dirty"
        end
        local stampedFile = path.relative(path.join(target:scriptdir(), "src/SystemInfo.cpp"), os.projectdir())
        target:fileconfig_add(stampedFile, {defines = 'LPLPLUGIN_COMMIT="' .. commit .. '"'})
    end)
