add_rules("mode.debug", "mode.release")
set_encodings("utf-8")

add_includedirs("include")

-- CPU --
includes("xmake/cpu.lua")

-- NVIDIA --
option("nv-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile implementations for Nvidia GPU")
option_end()

if has_config("nv-gpu") then
    add_defines("ENABLE_NVIDIA_API")
    includes("xmake/nvidia.lua")
end

-- Moore Threads MUSA --
option("mt-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile implementations for Moore Threads MUSA GPU")
option_end()

option("musa")
    set_default("")
    set_showmenu(true)
    set_description("MUSA SDK directory, e.g. /usr/local/musa")
option_end()

if has_config("mt-gpu") then
    add_defines("ENABLE_MOORE_API")
    includes("xmake/moore.lua")
end

target("llaisys-utils")
    set_kind("static")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/utils/*.cpp")

    on_install(function (target) end)
target_end()


target("llaisys-device")
    set_kind("static")
    add_deps("llaisys-utils")
    add_deps("llaisys-device-cpu")
    if has_config("nv-gpu") then
        add_deps("llaisys-device-nvidia")
    end
    if has_config("mt-gpu") then
        add_deps("llaisys-device-moore")
    end

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/device/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-core")
    set_kind("static")
    add_deps("llaisys-utils")
    add_deps("llaisys-device")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/core/*/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-tensor")
    set_kind("static")
    add_deps("llaisys-core")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/tensor/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-ops")
    set_kind("static")
    add_deps("llaisys-ops-cpu")
    if has_config("nv-gpu") then
        add_deps("llaisys-ops-nvidia")
    end
    if has_config("mt-gpu") then
        add_deps("llaisys-ops-moore")
    end

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
    
    add_files("src/ops/*/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys")
    set_kind("shared")
    if has_config("nv-gpu") then
        add_rules("cuda")
        add_links("cudart", "cublas")
    end
    if has_config("mt-gpu") then
        local musa_dir = get_config("musa")
        if musa_dir == nil or musa_dir == "" then
            musa_dir = os.getenv("MUSA_HOME") or "/usr/local/musa"
        end
        add_linkdirs(path.join(musa_dir, "lib"), path.join(musa_dir, "lib64"))
        add_rpathdirs(path.join(musa_dir, "lib"), path.join(musa_dir, "lib64"))
        add_links("musa", "mublas")
    end
    add_deps("llaisys-utils")
    add_deps("llaisys-device")
    add_deps("llaisys-core")
    add_deps("llaisys-tensor")
    add_deps("llaisys-ops")

    set_languages("cxx17")
    set_warnings("all", "error")
    add_files("src/llaisys/*.cc")
    set_installdir(".")

    
    after_install(function (target)
        -- copy shared library to python package
        print("Copying llaisys to python/llaisys/libllaisys/ ..")
        if is_plat("windows") then
            os.cp("bin/*.dll", "python/llaisys/libllaisys/")
            for _, dir in ipairs(os.dirs(".venv/Lib/site-packages/llaisys/libllaisys")) do
                print("Copying llaisys to " .. dir .. " ..")
                os.cp("bin/*.dll", dir)
            end
        end
        if is_plat("linux") then
            os.cp("lib/*.so", "python/llaisys/libllaisys/")
            for _, dir in ipairs(os.dirs(".venv/lib/python*/site-packages/llaisys/libllaisys")) do
                print("Copying llaisys to " .. dir .. " ..")
                os.cp("lib/*.so", dir)
            end
        end
    end)
target_end()
