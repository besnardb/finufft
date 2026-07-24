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

    message(WARNING
        "Could not detect AMD GPU architecture. "
        "Set CMAKE_HIP_ARCHITECTURES manually."
    )

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


enable_language(HIP)

find_package(hip REQUIRED)
find_package(hipfft REQUIRED)