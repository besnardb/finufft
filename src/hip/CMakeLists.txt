include(staticAnalysis)

set(SRC_HIP
    fft.hip
    heuristics.hip
    makeplan.hip
    setpts.hip
    execute.hip
    spread_blockgather_inst.hip
)
# Pure-host TUs: dispatch + C-API shims with no kernel launches and no
# __global__/__device__ tokens. Compile with the host C++ compiler so they
# don't pay nvcc's parsing cost and so MSVC/Windows toolchains see plain C++.
set(SRC_CXX c_interface.cpp spreadinterp.cpp)
set(SRC ${SRC_HIP} ${SRC_CXX})

# Per-method per-dim instantiation TUs. Each *_inst.cu is compiled once
# per dim (1, 2, 3) with -DCUFINUFFT_DIM=<dim> via configure_file copies
# in the build dir. Splitting per (method, dim) maximizes nvcc parallelism
# without duplicating source files in the tree.
set(CUFINUFFT_PER_DIM_METHODS
    spread_nupts_driven
    spread_subprob
    spread_output_driven
    interp_nupts_driven
    interp_subprob
)
foreach(method ${CUFINUFFT_PER_DIM_METHODS})
    foreach(dim 1 2 3)
        set(_gen ${CMAKE_CURRENT_BINARY_DIR}/${method}_${dim}d.hip)
        configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${method}_inst.hip ${_gen} COPYONLY)
        set_source_files_properties(${_gen} PROPERTIES COMPILE_DEFINITIONS "CUFINUFFT_DIM=${dim}")
        list(APPEND SRC ${_gen})
    endforeach()
endforeach()

set(CUFINUFFT_INCLUDE_DIRS
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/contrib>
    $<INSTALL_INTERFACE:include>
)

set(CUFINUFFT_INCLUDE_DIRS ${CUFINUFFT_INCLUDE_DIRS} PARENT_SCOPE)

list(APPEND FINUFFT_HIP_FLAGS $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=${CMAKE_HIP_ARCHITECTURES}>)
# -Wextra / -Wall are GCC/Clang spellings; nvcc forwards them to the host
# compiler via -forward-unknown-to-host-compiler, where MSVC rejects /Wextra
# with D8021. Only emit them when the host compiler understands them.
if(NOT MSVC)
    list(APPEND FINUFFT_HIP_FLAGS $<$<COMPILE_LANGUAGE:HIP>:-Wextra -Wall>)
endif()

if(FINUFFT_SHARED_LINKING)
    add_library(cufinufft SHARED ${SRC})
else()
    add_library(cufinufft STATIC ${SRC})
endif()
set_source_files_properties(${SRC_CXX} PROPERTIES LANGUAGE CXX)
set_source_files_properties(
    ${SRC_HIP}
    PROPERTIES
    LANGUAGE HIP
)

enable_static_analysis(cufinufft)
set_property(TARGET ${target} PROPERTY INTERPROCEDURAL_OPTIMIZATION ${FINUFFT_INTERPROCEDURAL_OPTIMIZATION})
add_library(finufft::cufinufft ALIAS cufinufft)
target_include_directories(cufinufft PUBLIC ${CUFINUFFT_INCLUDE_DIRS})
# Generated per-dim instantiation TUs in build dir need to find
# spreadinterp_common.cuh (shared device-helper header) in the source dir.
target_include_directories(cufinufft PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
# set target build location
set_target_properties(cufinufft PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

if(FINUFFT_SHARED_LINKING AND FINUFFT_BUILD_TESTS)
    set(FINUFFT_CUDA_VISIBILITY_PRESET default)
else()
    set(FINUFFT_CUDA_VISIBILITY_PRESET hidden)
endif()

set_target_properties(
    cufinufft
    PROPERTIES
        HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}"
        # RDC is disabled: no cross-TU __device__ calls exist (all device
        # helpers are inline/__forceinline__/template in headers, all
        # __global__ kernels are launched from the same TU that defines
        # them). Turning this off skips the device-link step, enables
        # whole-program inlining within each TU, and avoids MSVC + nvcc
        # toolchain quirks. Reintroduce only if a future feature genuinely
        # needs cross-TU device calls.
        HIP_SEPARABLE_COMPILATION OFF
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
        INTERPROCEDURAL_OPTIMIZATION ${FINUFFT_INTERPROCEDURAL_OPTIMIZATION}
        POSITION_INDEPENDENT_CODE ${FINUFFT_POSITION_INDEPENDENT_CODE}
        CXX_VISIBILITY_PRESET ${FINUFFT_CUDA_VISIBILITY_PRESET}
        VISIBILITY_INLINES_HIDDEN YES
        WINDOWS_EXPORT_ALL_SYMBOLS NO
)

if(DEFINED ENV{GITHUB_ACTIONS} AND "$ENV{GITHUB_ACTIONS}" STREQUAL "true" AND MSVC)
    message(STATUS "GITHUB_ACTIONS on Windows using c++20 for cufinufft")
    target_compile_features(cufinufft PRIVATE cxx_std_20 hip_std_20)
else()
    target_compile_features(cufinufft PRIVATE cxx_std_17 hip_std_17)
endif()
target_compile_options(cufinufft PUBLIC ${FINUFFT_HIP_FLAGS})

if(FINUFFT_SHARED_LINKING)
    target_compile_definitions(cufinufft PRIVATE FINUFFT_DLL)
    if(WIN32)
        target_compile_definitions(cufinufft PRIVATE dll_EXPORTS)
    endif()
endif()

target_compile_definitions(cufinufft
    PUBLIC CUFINUFFT_USE_HIP
)

# BUILD_INTERFACE exposes deps to the library itself and to all build consumers
# (tests, examples) without requiring them to list dependencies explicitly.
# Shared library consumers don't need these re-exposed: the .so encodes its own
# runtime deps and CCCL is header-only with nothing to link.
target_link_libraries(
    cufinufft
    PUBLIC
        $<BUILD_INTERFACE:hip::device>
        $<BUILD_INTERFACE:hip::hipfft>
        $<BUILD_INTERFACE:finufft_common>
        $<BUILD_INTERFACE:poet::poet>
)

check_cxx_compiler_flag(-Wno-deprecated-declarations FINUFFT_HAS_NO_DEPRECATED_DECLARATIONS)
# disable deprecated warnings for tests if supported
if(FINUFFT_HAS_NO_DEPRECATED_DECLARATIONS)
    target_compile_options(cufinufft PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-declarations>)
endif()

file(GLOB CUFINUFFT_PUBLIC_HEADERS "${CMAKE_SOURCE_DIR}/include/cufinufft*.h")
set_target_properties(cufinufft PROPERTIES PUBLIC_HEADER "${CUFINUFFT_PUBLIC_HEADERS}")
