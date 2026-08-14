local musa_dir = get_config("musa")
if musa_dir == nil or musa_dir == "" then
    musa_dir = os.getenv("MUSA_HOME") or "/usr/local/musa"
end

local mcc = path.join(musa_dir, "bin", "mcc")
local mcc_tool = "gcc@" .. mcc

target("llaisys-device-moore")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")
    set_toolset("cxx", mcc_tool)

    add_includedirs(path.join(musa_dir, "include"))
    add_linkdirs(path.join(musa_dir, "lib"), path.join(musa_dir, "lib64"))
    add_links("musa")

    add_cxflags("-x", "musa")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("../src/device/moore/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-ops-moore")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")
    set_toolset("cxx", mcc_tool)

    add_includedirs(path.join(musa_dir, "include"))
    add_linkdirs(path.join(musa_dir, "lib"), path.join(musa_dir, "lib64"))
    add_links("musa", "mublas")

    add_cxflags("-x", "musa")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("../src/ops/*/moore/*.cpp")

    on_install(function (target) end)
target_end()
