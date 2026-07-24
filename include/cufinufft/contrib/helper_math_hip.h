#ifndef FINUFFT_INCLUDE_CUFINUFFT_CONTRIB_HELPER_MATH_H
#define FINUFFT_INCLUDE_CUFINUFFT_CONTRIB_HELPER_MATH_H

#include <hip/hip_complex.h>

// This header provides some helper functions for cuComplex types.
// It mainly wraps existing CUDA implementations to provide operator overloads
// e.g. cuAdd, cuSub, cuMul, cuDiv, cuCreal, cuCimag, cuCabs, cuCarg, cuConj are all
// provided by CUDA

// Addition for cuDoubleComplex (double) with cuDoubleComplex (double)
__host__ __device__ __forceinline__ hipDoubleComplex operator+(
    const hipDoubleComplex &a, const hipDoubleComplex &b) noexcept {
  return hipCadd(a, b);
}

// Subtraction for cuDoubleComplex (double) with cuDoubleComplex (double)
__host__ __device__ __forceinline__ hipDoubleComplex operator-(
    const hipDoubleComplex &a, const hipDoubleComplex &b) noexcept {
  return hipCsub(a, b);
}

// Multiplication for cuDoubleComplex (double) with cuDoubleComplex (double)
__host__ __device__ __forceinline__ hipDoubleComplex operator*(
    const hipDoubleComplex &a, const hipDoubleComplex &b) noexcept {
  return hipCmul(a, b);
}

// Division for cuDoubleComplex (double) with cuDoubleComplex (double)
__host__ __device__ __forceinline__ hipDoubleComplex operator/(
    const hipDoubleComplex &a, const hipDoubleComplex &b) noexcept {
  return hipCdiv(a, b);
}

// Equality for cuDoubleComplex (double) with cuDoubleComplex (double)
__host__ __device__ __forceinline__ bool operator==(const hipDoubleComplex &a,
                                                    const hipDoubleComplex &b) noexcept {
  return hipCreal(a) == hipCreal(b) && hipCimag(a) == hipCimag(b);
}

// Inequality for cuDoubleComplex (double) with cuDoubleComplex (double)
__host__ __device__ __forceinline__ bool operator!=(const hipDoubleComplex &a,
                                                    const hipDoubleComplex &b) noexcept {
  return !(a == b);
}

// Addition for cuDoubleComplex (double) with double
__host__ __device__ __forceinline__ hipDoubleComplex operator+(const hipDoubleComplex &a,
                                                              double b) noexcept {
  return make_hipDoubleComplex(hipCreal(a) + b, hipCimag(a));
}

__host__ __device__ __forceinline__ hipDoubleComplex operator+(
    double a, const hipDoubleComplex &b) noexcept {
  return make_hipDoubleComplex(a + hipCreal(b), hipCimag(b));
}

// Subtraction for cuDoubleComplex (double) with double
__host__ __device__ __forceinline__ hipDoubleComplex operator-(const hipDoubleComplex &a,
                                                              double b) noexcept {
  return make_hipDoubleComplex(hipCreal(a) - b, hipCimag(a));
}

__host__ __device__ __forceinline__ hipDoubleComplex operator-(
    double a, const hipDoubleComplex &b) noexcept {
  return make_hipDoubleComplex(a - hipCreal(b), -hipCimag(b));
}

// Multiplication for cuDoubleComplex (double) with double
__host__ __device__ __forceinline__ hipDoubleComplex operator*(const hipDoubleComplex &a,
                                                              double b) noexcept {
  return make_hipDoubleComplex(hipCreal(a) * b, hipCimag(a) * b);
}

__host__ __device__ __forceinline__ hipDoubleComplex operator*(
    double a, const hipDoubleComplex &b) noexcept {
  return make_hipDoubleComplex(a * hipCreal(b), a * hipCimag(b));
}

// Division for cuDoubleComplex (double) with double
__host__ __device__ __forceinline__ hipDoubleComplex operator/(const hipDoubleComplex &a,
                                                              double b) noexcept {
  return make_hipDoubleComplex(hipCreal(a) / b, hipCimag(a) / b);
}

__host__ __device__ __forceinline__ hipDoubleComplex operator/(
    double a, const hipDoubleComplex &b) noexcept {
  double denom = hipCreal(b) * hipCreal(b) + hipCimag(b) * hipCimag(b);
  return make_hipDoubleComplex((a * hipCreal(b)) / denom, (-a * hipCimag(b)) / denom);
}

// Addition for cuFloatComplex (float) with cuFloatComplex (float)
__host__ __device__ __forceinline__ hipFloatComplex operator+(
    const hipFloatComplex &a, const hipFloatComplex &b) noexcept {
  return hipCaddf(a, b);
}

// Subtraction for cuFloatComplex (float) with cuFloatComplex (float)
__host__ __device__ __forceinline__ hipFloatComplex operator-(
    const hipFloatComplex &a, const hipFloatComplex &b) noexcept {
  return hipCsubf(a, b);
}

// Multiplication for cuFloatComplex (float) with cuFloatComplex (float)
__host__ __device__ __forceinline__ hipFloatComplex operator*(
    const hipFloatComplex &a, const hipFloatComplex &b) noexcept {
  return hipCmulf(a, b);
}

// Division for cuFloatComplex (float) with cuFloatComplex (float)
__host__ __device__ __forceinline__ hipFloatComplex operator/(
    const hipFloatComplex &a, const hipFloatComplex &b) noexcept {
  return hipCdivf(a, b);
}

// Equality for cuFloatComplex (float) with cuFloatComplex (float)
__host__ __device__ __forceinline__ bool operator==(const hipFloatComplex &a,
                                                    const hipFloatComplex &b) noexcept {
  return hipCrealf(a) == hipCrealf(b) && hipCimagf(a) == hipCimagf(b);
}

// Inequality for cuFloatComplex (float) with cuFloatComplex (float)
__host__ __device__ __forceinline__ bool operator!=(const hipFloatComplex &a,
                                                    const hipFloatComplex &b) noexcept {
  return !(a == b);
}

// Addition for cuFloatComplex (float) with float
__host__ __device__ __forceinline__ hipFloatComplex operator+(const hipFloatComplex &a,
                                                             float b) noexcept {
  return make_hipFloatComplex(hipCrealf(a) + b, hipCimagf(a));
}

__host__ __device__ __forceinline__ hipFloatComplex operator+(
    float a, const hipFloatComplex &b) noexcept {
  return make_hipFloatComplex(a + hipCrealf(b), hipCimagf(b));
}

// Subtraction for cuFloatComplex (float) with float
__host__ __device__ __forceinline__ hipFloatComplex operator-(const hipFloatComplex &a,
                                                             float b) noexcept {
  return make_hipFloatComplex(hipCrealf(a) - b, hipCimagf(a));
}

__host__ __device__ __forceinline__ hipFloatComplex operator-(
    float a, const hipFloatComplex &b) noexcept {
  return make_hipFloatComplex(a - hipCrealf(b), -hipCimagf(b));
}

// Multiplication for cuFloatComplex (float) with float
__host__ __device__ __forceinline__ hipFloatComplex operator*(const hipFloatComplex &a,
                                                             float b) noexcept {
  return make_hipFloatComplex(hipCrealf(a) * b, hipCimagf(a) * b);
}

__host__ __device__ __forceinline__ hipFloatComplex operator*(
    float a, const hipFloatComplex &b) noexcept {
  return make_hipFloatComplex(a * hipCrealf(b), a * hipCimagf(b));
}

// Division for cuFloatComplex (float) with float
__host__ __device__ __forceinline__ hipFloatComplex operator/(const hipFloatComplex &a,
                                                             float b) noexcept {
  return make_hipFloatComplex(hipCrealf(a) / b, hipCimagf(a) / b);
}

__host__ __device__ __forceinline__ hipFloatComplex operator/(
    float a, const hipFloatComplex &b) noexcept {
  float denom = hipCrealf(b) * hipCrealf(b) + hipCimagf(b) * hipCimagf(b);
  return make_hipFloatComplex((a * hipCrealf(b)) / denom, (-a * hipCimagf(b)) / denom);
}

template<typename T>
__device__ inline hip_complex<T> complex_mul_real(
    hip_complex<T> a, T b);


template<>
__device__ inline hipFloatComplex complex_mul_real(
    hipFloatComplex a, float b)
{
    return hipCmulf(a, make_hipFloatComplex(b,0));
}


template<>
__device__ inline hipDoubleComplex complex_mul_real(
    hipDoubleComplex a, double b)
{
    return hipCmul(a, make_hipDoubleComplex(b,0));
}


template<typename T>
__device__ inline hip_complex<T> complex_add(
    hip_complex<T> a,
    hip_complex<T> b);


template<>
__device__ inline hipFloatComplex complex_add<float>(
    hipFloatComplex a,
    hipFloatComplex b)
{
    return hipCaddf(a, b);
}


template<>
__device__ inline hipDoubleComplex complex_add<double>(
    hipDoubleComplex a,
    hipDoubleComplex b)
{
    return hipCadd(a, b);
}

/* Auxiliary func to compute power of complex number */
__device__ __forceinline__ double cabs(const hipDoubleComplex &z) noexcept {
  return double(hipCabs(z));
}

#endif // FINUFFT_INCLUDE_CUFINUFFT_CONTRIB_HELPER_MATH_H
