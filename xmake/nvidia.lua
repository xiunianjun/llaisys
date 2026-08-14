target("llaisys-device-nvidia")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")

    add_rules("cuda")
    set_policy("build.cuda.devlink", true)
    add_links("cudart")

    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        add_cuflags("-Xcompiler=-fPIC")
        add_culdflags("-Xcompiler=-fPIC")
    end

    add_files("../src/device/nvidia/*.cu")

    on_install(function (target) end)
target_end()

target("llaisys-ops-nvidia")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")

    add_rules("cuda")
    set_policy("build.cuda.devlink", true)
    add_links("cudart")

    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
        add_cuflags("-Xcompiler=-fPIC")
        add_culdflags("-Xcompiler=-fPIC")
    end

    add_files("../src/ops/*/nvidia/*.cu")

    on_install(function (target) end)
target_end()
