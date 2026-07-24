#ifndef FINUFFT_INCLUDE_CUFINUFFT_CONTRIB_HELPER_MATH_H
#define FINUFFT_INCLUDE_CUFINUFFT_CONTRIB_HELPER_MATH_H

#include <hip/hip_complex.h>

// This header provides some helper functions for cuComplex types.
// It mainly wraps existing CUDA implementations to provide operator overloads
// e.g. cuAdd, cuSub, cuMul, cuDiv, cuCreal, cuCimag, cuCabs, cuCarg, cuConj are all
// provided by CUDA

__device__ __forceinline__ hipFloatComplex complex_mul_real(
    hipFloatComplex a, float b)
{
    hipFloatComplex r;
    r.x = a.x * b;
    r.y = a.y * b;
    return r;
}


__device__ __forceinline__ hipDoubleComplex complex_mul_real(
    hipDoubleComplex a, double b)
{
    hipDoubleComplex r;
    r.x = a.x * b;
    r.y = a.y * b;
    return r;
}

__device__ __forceinline__ hipFloatComplex complex_add(
    hipFloatComplex a,
    hipFloatComplex b)
{
    hipFloatComplex r;
    r.x = a.x + b.x;
    r.y = a.y + b.y;
    return r;
}


__device__ __forceinline__ hipDoubleComplex complex_add(
    hipDoubleComplex a,
    hipDoubleComplex b)
{
    hipDoubleComplex r;
    r.x = a.x + b.x;
    r.y = a.y + b.y;
    return r;
}

/* Auxiliary func to compute power of complex number */
__device__ __forceinline__ double cabs(const hipDoubleComplex &z) noexcept {
  return double(hipCabs(z));
}

#endif // FINUFFT_INCLUDE_CUFINUFFT_CONTRIB_HELPER_MATH_H
