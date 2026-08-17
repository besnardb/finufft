include_guard(GLOBAL)

function(detect_hip_architecture)

    find_program(ROCMINFO_EXECUTABLE rocminfo)

    if(ROCMINFO_EXECUTABLE)

        execute_process(
            COMMAND ${ROCMINFO_EXECUTABLE}
            OUTPUT_VARIABLE rocminfo_output
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )

        string(REGEX MATCH "Name:\\s+gfx[0-9]+" GPU_ARCH "${rocminfo_output}")

        if(GPU_ARCH)
            string(REGEX REPLACE ".*(gfx[0-9]+).*" "\\1" GPU_ARCH "${GPU_ARCH}")

            message(STATUS "Detected HIP architecture: ${GPU_ARCH}")

            set(CMAKE_HIP_ARCHITECTURES
                "${GPU_ARCH}"
                CACHE STRING "HIP architectures"
                FORCE
            )

            return()
        endif()

    endif()

    message(STATUS
        "Could not detect AMD GPU architecture (no GPU visible on this host). "
        "Falling back to CMAKE_HIP_ARCHITECTURES=all. Set it explicitly "
        "(e.g. -DCMAKE_HIP_ARCHITECTURES=gfx90a) to target a specific GPU."
    )
    set(CMAKE_HIP_ARCHITECTURES "all" CACHE STRING "HIP architectures" FORCE)

endfunction()


if(NOT DEFINED CMAKE_HIP_ARCHITECTURES
   OR CMAKE_HIP_ARCHITECTURES STREQUAL "")

    detect_hip_architecture()

else()

    message(
        STATUS
        "Using user-specified CMAKE_HIP_ARCHITECTURES=${CMAKE_HIP_ARCHITECTURES}"
    )

endif()

if(DEFINED ENV{ROCM_PATH})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{ROCM_PATH}")
elseif(CMAKE_CXX_COMPILER MATCHES "^(.*)/llvm/bin/clang\\+\\+$")
    list(APPEND CMAKE_PREFIX_PATH "${CMAKE_MATCH_1}")
elseif(EXISTS "/opt/rocm")
    list(APPEND CMAKE_PREFIX_PATH "/opt/rocm")
endif()

enable_language(HIP)

find_package(hip REQUIRED)
find_package(hipfft REQUIRED)