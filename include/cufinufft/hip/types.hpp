#ifndef CUFINUFFT_TYPES_H
#define CUFINUFFT_TYPES_H

#include <hipfft/hipfft.h>
#include <cufinufft_opts.h>
#include <finufft_common/common.h>

#include <hip/hip_complex.h>
#include <limits>
#include <type_traits>

// FIXME: If hipfft ever takes N > INT_MAX...
constexpr int32_t MAX_NF = std::numeric_limits<int32_t>::max();

using CUFINUFFT_BIGINT = int;


// Define hip_complex<T>
// if T is float, hip_complex<T> is hipFloatComplex
// if T is double, hip_complex<T> is hipDoubleComplex
template<typename T>
using hip_complex = typename std::conditional<
    std::is_same<T, float>::value,
    hipFloatComplex,
    typename std::conditional<
        std::is_same<T, double>::value,
        hipDoubleComplex,
        void
    >::type
>::type;


// FFT types
template<typename T>
static inline constexpr hipfftType_t hipfft_type();

template<>
inline constexpr hipfftType_t hipfft_type<float>()
{
    return HIPFFT_C2C;
}

template<>
inline constexpr hipfftType_t hipfft_type<double>()
{
    return HIPFFT_Z2Z;
}


// FFT execution helpers
static inline hipfftResult hipfft_ex(
    hipfftHandle plan,
    hipFloatComplex *idata,
    hipFloatComplex *odata,
    int direction)
{
    return hipfftExecC2C(plan, idata, odata, direction);
}


static inline hipfftResult hipfft_ex(
    hipfftHandle plan,
    hipDoubleComplex *idata,
    hipDoubleComplex *odata,
    int direction)
{
    return hipfftExecZ2Z(plan, idata, odata, direction);
}


#endif