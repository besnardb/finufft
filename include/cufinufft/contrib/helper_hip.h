#ifndef COMMON_HELPER_HIP_H_
#define COMMON_HELPER_HIP_H_

/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <string>
#include <hip/hip_runtime.h>
#include <hip/hip_math.h>
#include <finufft_common/safe_call.h>
#include <finufft_errors.h>

namespace cufinufft {

// Helper function to get HIP error string
inline const char *hipGetErrorString(hipError_t error) {
    switch (error) {
    case hipSuccess:
        return "HIP_SUCCESS";
    case hipErrorInvalidValue:
        return "HIP_ERROR_INVALID_VALUE";
    case hipErrorMemoryAllocation:
        return "HIP_ERROR_MEMORY_ALLOCATION";
    case hipErrorInitializationError:
        return "HIP_ERROR_INITIALIZATION_ERROR";
    case hipErrorNoDevice:
        return "HIP_ERROR_NO_DEVICE";
    case hipErrorInvalidDevice:
        return "HIP_ERROR_INVALID_DEVICE";
    case hipErrorInvalidConfiguration:
        return "HIP_ERROR_INVALID_CONFIGURATION";
    case hipErrorInvalidResourceHandle:
        return "HIP_ERROR_INVALID_RESOURCE_HANDLE";
    case hipErrorOutOfResources:
        return "HIP_ERROR_OUT_OF_RESOURCES";
    case hipErrorNotFound:
        return "HIP_ERROR_NOT_FOUND";
    case hipErrorNotReady:
        return "HIP_ERROR_NOT_READY";
    case hipErrorIllegalAddress:
        return "HIP_ERROR_ILLEGAL_ADDRESS";
    case hipErrorLaunchFailure:
        return "HIP_ERROR_LAUNCH_FAILURE";
    case hipErrorLaunchTimeout:
        return "HIP_ERROR_LAUNCH_TIMEOUT";
    case hipErrorLaunchOutOfResources:
        return "HIP_ERROR_LAUNCH_OUT_OF_RESOURCES";
    case hipErrorNotSupported:
        return "HIP_ERROR_NOT_SUPPORTED";
    case hipErrorUnknown:
        return "HIP_ERROR_UNKNOWN";
    case hipErrorMapFailed:
        return "HIP_ERROR_MAP_FAILED";
    case hipErrorUnmappedResource:
        return "HIP_ERROR_UNMAPPED_RESOURCE";
    case hipErrorMemoryValueTooLarge:
        return "HIP_ERROR_MEMORY_VALUE_TOO_LARGE";
    case hipErrorInvalidHostPointer:
        return "HIP_ERROR_INVALID_HOST_POINTER";
    case hipErrorInvalidDevicePointer:
        return "HIP_ERROR_INVALID_DEVICE_POINTER";
    case hipErrorInvalidTexture:
        return "HIP_ERROR_INVALID_TEXTURE";
    case hipErrorArrayIsMapped:
        return "HIP_ERROR_ARRAY_IS_MAPPED";
    case hipErrorNotMapped:
        return "HIP_ERROR_NOT_MAPPED";
    case hipErrorInvalidPitchValue:
        return "HIP_ERROR_INVALID_PITCH_VALUE";
    case hipErrorInvalidSymbol:
        return "HIP_ERROR_INVALID_SYMBOL";
    case hipErrorInvalidExtAddr:
        return "HIP_ERROR_INVALID_EXT_ADDR";
    case hipErrorInvalidMemcpyDirection:
        return "HIP_ERROR_INVALID_MEMCPY_DIRECTION";
    case hipErrorAddressOfConstant:
        return "HIP_ERROR_ADDRESS_OF_CONSTANT";
    case hipErrorTextureFetchFailed:
        return "HIP_ERROR_TEXTURE_FETCH_FAILED";
    case hipErrorTextureNotBound:
        return "HIP_ERROR_TEXTURE_NOT_BOUND";
    case hipErrorSharedObjectSymbolNotFound:
        return "HIP_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND";
    case hipErrorSharedObjectInitFailed:
        return "HIP_ERROR_SHARED_OBJECT_INIT_FAILED";
    case hipErrorUnsatisfiedLinkOrder:
        return "HIP_ERROR_UNSATISFIED_LINK_ORDER";
    case hipErrorUnsatisfiedLinkText:
        return "HIP_ERROR_UNSATISFIED_LINK_TEXT";
    case hipErrorDriverMismatch:
        return "HIP_ERROR_DRIVER_MISMATCH";
    case hipErrorECCUncorrectable:
        return "HIP_ERROR_ECC_UNCORRECTABLE";
    case hipErrorSharedObjectFormat:
        return "HIP_ERROR_SHARED_OBJECT_FORMAT";
    case hipErrorCudartUnloading:
        return "HIP_ERROR_CUDART_UNLOADING";
    case hipErrorUnknownFailure:
        return "HIP_ERROR_UNKNOWN_FAILURE";
    case hipErrorSurrogateHasError:
        return "HIP_ERROR_SURROGATE_HAS_ERROR";
    case hipErrorIllegalInstruction:
        return "HIP_ERROR_ILLEGAL_INSTRUCTION";
    case hipErrorInvalidPreemption:
        return "HIP_ERROR_INVALID_PREEMPTION";
    case hipErrorInvalidAddressSpace:
        return "HIP_ERROR_INVALID_ADDRESS_SPACE";
    case hipErrorInvalidDeviceFunction:
        return "HIP_ERROR_INVALID_DEVICE_FUNCTION";
    case hipErrorInvalidKernelImage:
        return "HIP_ERROR_INVALID_KERNEL_IMAGE";
    case hipErrorInvalidExecConfiguration:
        return "HIP_ERROR_INVALID_EXEC_CONFIGURATION";
    case hipErrorInvalidKernelTooFewArgs:
        return "HIP_ERROR_INVALID_KERNEL_TOO_FEW_ARGS";
    case hipErrorInvalidKernelTooManyArgs:
        return "HIP_ERROR_INVALID_KERNEL_TOO_MANY_ARGS";
    case hipErrorNoKernelImageForDevice:
        return "HIP_ERROR_NO_KERNEL_IMAGE_FOR_DEVICE";
    case hipErrorHostMemoryAlreadyRegistered:
        return "HIP_ERROR_HOST_MEMORY_ALREADY_REGISTERED";
    case hipErrorHostMemoryNotRegistered:
        return "HIP_ERROR_HOST_MEMORY_NOT_REGISTERED";
    default:
        return "<unknown>";
    }
}

// Typed exception carrying a hipError_t plus a caller-supplied operation tag
// and optional source location.
class hip_exception final : public finufft::exception {
public:
    hip_exception(hipError_t err, const char *op, const char *file = nullptr,
                  int line = 0)
        : finufft::exception(FINUFFT_ERR_HIP_FAILURE, format(err, op, file, line)),
          hip_code_(err) {}

    hipError_t hip_code() const noexcept { return hip_code_; }

private:
    hipError_t hip_code_;

    static std::string format(hipError_t e, const char *op, const char *file, int line) {
        std::string s = op ? op : "<unknown>";
        if (file) {
            s += " @ ";
            s += file;
            s += ":";
            s += std::to_string(line);
        }
        s += ": ";
        s += hipGetErrorName(e);
        s += " (";
        s += hipGetErrorString(e);
        s += ")";
        return s;
    }
};

namespace detail {

// Sample (and clear) the sticky HIP error flag. Throws if a kernel launch or
// prior async call left an error on the stream.
inline void throw_if_hip_error(const char *op, const char *file = nullptr,
                               int line = 0) {
    if (const hipError_t e = hipGetLastError(); e != hipSuccess)
        throw hip_exception(e, op, file, line);
}

// Check a synchronous HIP runtime return code.
inline void check_hip_status(hipError_t s, const char *op, const char *file = nullptr,
                             int line = 0) {
    if (s != hipSuccess) throw hip_exception(s, op, file, line);
}

} // namespace detail

} // namespace cufinufft

// Wrap a HIP runtime call; `#call` stringifies the expression for the tag and
// `__FILE__`/`__LINE__` pinpoint the call site for debugging.
#define checkHipErrors(call) \
    ::cufinufft::detail::check_hip_status((call), #call, __FILE__, __LINE__)

// Sample the sticky HIP error flag and tag it with the enclosing function
// name and source location.
#define THROW_IF_HIP_ERROR() \
    ::cufinufft::detail::throw_if_hip_error(__func__, __FILE__, __LINE__)

#endif // COMMON_HELPER_HIP_H_
