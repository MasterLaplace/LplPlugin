-- /////////////////////////////////////////////////////////////////////////////
-- /// @file xmake.lua
-- /// @brief Kernel module build & management targets.
-- ///
-- /// These are phony targets that wrap the kernel build system and
-- /// insmod/rmmod commands for the LPL kernel module.
-- ///
-- /// Usage:
-- ///   xmake kmod-build      — Build lpl_kmod.ko
-- ///   xmake kmod-install    — Load module + set /dev/lpl0 permissions
-- ///   xmake kmod-uninstall  — Unload module
-- ///   xmake kmod-logs       — Show recent kernel logs
-- ///   xmake kmod-clean      — Clean kernel build artifacts
-- /////////////////////////////////////////////////////////////////////////////

local kernel_dir = path.absolute(os.scriptdir())

task("kmod-build")
    on_run(function ()
        local old_dir = os.cd(kernel_dir)
        os.exec("make")
        os.cd(old_dir)
        print("lpl_kmod.ko built successfully")
    end)
    set_menu {
        usage = "xmake kmod-build",
        description = "Build the LPL kernel module (lpl_kmod.ko)"
    }
task_end()

task("kmod-install")
    set_category("plugin")
    on_run(function ()
        local ko_path = path.join(kernel_dir, "lpl_kmod.ko")
        if not os.isfile(ko_path) then
            raise("lpl_kmod.ko not found — run 'xmake kmod-build' first")
        end
        local _, status = os.execv("sudo", {"insmod", ko_path})
        if status == nil then
            print("[kmod-install] insmod returned no status, continuing anyway")
        elseif status ~= 0 then
            print("[kmod-install] insmod failed (status " .. status .. ")")
            print("[kmod-install] continuing anyway; module may already be loaded")
        end
        os.execv("sudo", {"chmod", "666", "/dev/lpl0"})
        os.execv("sudo", {"setfacl", "-m", "u::rw,g::rw,o::rw", "/dev/lpl0"})
        print("lpl_kmod loaded (or already present), /dev/lpl0 permissions/ACLs set")
    end)
    set_menu {
        usage = "xmake kmod-install",
        description = "Load the LPL kernel module (insmod + chmod /dev/lpl0)"
    }
task_end()

task("kmod-uninstall")
    set_category("plugin")
    on_run(function ()
        os.execv("sudo", {"rmmod", "lpl_kmod"})
        print("lpl_kmod unloaded")
    end)
    set_menu {
        usage = "xmake kmod-uninstall",
        description = "Unload the LPL kernel module (rmmod)"
    }
task_end()

task("kmod-logs")
    set_category("plugin")
    on_run(function ()
        os.exec("sudo sh -c 'dmesg | tail -20'")
    end)
    set_menu {
        usage = "xmake kmod-logs",
        description = "Show recent kernel logs (dmesg tail)"
    }
task_end()

task("kmod-clean")
    set_category("plugin")
    on_run(function ()
        local old_dir = os.cd(kernel_dir)
        os.exec("make clean")
        os.cd(old_dir)
        print("Kernel build artifacts cleaned")
    end)
    set_menu {
        usage = "xmake kmod-clean",
        description = "Clean kernel module build artifacts"
    }
task_end()
