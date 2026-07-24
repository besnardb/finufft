#include <hip/hip_complex.h>
// Per-dim instantiation TU: nupts-driven interp (gpu_method = 1).
// Compiled three times via CMake foreach with -DCUFINUFFT_DIM={1,2,3}.

#include <cufinufft/spreadinterp.hpp>
#include <poet/poet.hpp>
#ifndef CUFINUFFT_DIM
#error "CUFINUFFT_DIM must be defined to 1, 2, or 3 (set by CMake)"
#endif

namespace cufinufft {
namespace spreadinterp {

// Nupts-driven interpolation kernel
template<typename T, int KEREVALMETH, int ndim, int ns>
__global__ FINUFFT_FLATTEN void interp_nupts_driven(
    cufinufft_gpu_data<T> p, hipComplex<T> *c, const hipComplex<T> *fw) {
  T es_c    = p.es_c;
  T es_beta = p.es_beta;
  T sigma   = p.sigma;

  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < p.M;
       i += blockDim.x * gridDim.x) {
    const auto nuptsidx = loadReadOnly(p.idxnupts + i);

    auto [ker, start] = get_kerval_and_startpos_nuptsdriven<T, KEREVALMETH, ndim, ns>(
        nuptsidx, p.xyz, p.nf123, sigma, es_c, es_beta);

    hipComplex<T> cnow{0, 0};
    if constexpr (ndim == 1) {
      for (int x0 = 0, ix = start[0]; x0 < ns;
           ++x0, ix       = (ix + 1 >= p.nf123[0]) ? 0 : ix + 1)
        cnow += loadReadOnly(fw + ix) * ker[0][x0];
    } else if constexpr (ndim == 2) {
      for (int y0 = 0, iy = start[1]; y0 < ns;
           ++y0, iy       = (iy + 1 >= p.nf123[1]) ? 0 : iy + 1) {
        const auto inidx0 = iy * p.nf123[0];
        hipComplex<T> cnowx{0, 0};
        for (int x0 = 0, ix = start[0]; x0 < ns;
             ++x0, ix       = (ix + 1 >= p.nf123[0]) ? 0 : ix + 1)
          cnowx += loadReadOnly(fw + inidx0 + ix) * ker[0][x0];
        cnow += cnowx * ker[1][y0];
      }
    }
    if constexpr (ndim == 3) {
      hip::std::array<int, ns> xidx;
      for (int x0 = 0, ix = start[0]; x0 < ns;
           ++x0, ix       = (ix + 1 >= p.nf123[0]) ? 0 : ix + 1)
        xidx[x0] = ix;
      for (int z0 = 0, iz = start[2]; z0 < ns;
           ++z0, iz       = (iz + 1 >= p.nf123[2]) ? 0 : iz + 1) {
        const auto inidx0 = iz * p.nf123[1] * p.nf123[0];
        hipComplex<T> cnowy{0, 0};
        for (int y0 = 0, iy = start[1]; y0 < ns;
             ++y0, iy       = (iy + 1 >= p.nf123[1]) ? 0 : iy + 1) {
          const auto inidx1 = inidx0 + iy * p.nf123[0];
          hipComplex<T> cnowx{0, 0};
          for (int x0 = 0; x0 < ns; ++x0)
            cnowx += loadReadOnly(fw + inidx1 + xidx[x0]) * ker[0][x0];
          cnowy += cnowx * ker[1][y0];
        }
        cnow += cnowy * ker[2][z0];
      }
    }
    storeCacheStreaming(c + nuptsidx, cnow);
  }
}

// Nupts-driven interpolation CPU driver
template<typename T, int ndim, int ns>
void interp_nupts_driven_launch(const cufinufft_plan_t<T> &d_plan, hipComplex<T> *c,
                                const hipComplex<T> *fw, int blksize) {
  const dim3 threadsPerBlock{
      optimal_interp_block_threads(d_plan.opts.gpu_device_id, d_plan.M), 1u, 1u};
  const dim3 blocks{(d_plan.M + threadsPerBlock.x - 1) / threadsPerBlock.x, 1, 1};

  const auto launch = [&](auto kernel) {
    for (int t = 0; t < blksize; t++) {
      kernel<<<blocks, threadsPerBlock, 0, d_plan.stream>>>(d_plan, c + t * d_plan.M,
                                                            fw + t * d_plan.nf);
      THROW_IF_CUDA_ERROR();
    }
  };
  (d_plan.opts.gpu_kerevalmeth == 1) ? launch(interp_nupts_driven<T, 1, ndim, ns>)
                                     : launch(interp_nupts_driven<T, 0, ndim, ns>);
}

template<typename T, int Ndim> struct InterpNuptsDrivenCaller {
  const cufinufft_plan_t<T> &p;
  hipComplex<T> *c;
  const hipComplex<T> *fw;
  int blksize;
  template<int Ns> void operator()() const {
    interp_nupts_driven_launch<T, Ndim, Ns>(p, c, fw, blksize);
  }
};

template<typename T, int Ndim>
void do_interp_nupts_driven(const cufinufft_plan_t<T> &p, hipComplex<T> *c,
                            const hipComplex<T> *fw, int blksize) {
  using namespace finufft::common;
  InterpNuptsDrivenCaller<T, Ndim> caller{p, c, fw, blksize};
  using NsSeq = poet::inclusive_range<MIN_NSPREAD, MAX_NSPREAD<T>>;
  poet::dispatch(caller, std::make_tuple(poet::dispatch_param<NsSeq>{p.spopts.nspread}));
}

template void do_interp_nupts_driven<float, CUFINUFFT_DIM>(
    const cufinufft_plan_t<float> &, hipComplex<float> *, const hipComplex<float> *,
    int);
template void do_interp_nupts_driven<double, CUFINUFFT_DIM>(
    const cufinufft_plan_t<double> &, hipComplex<double> *,
    const hipComplex<double> *, int);

} // namespace spreadinterp
} // namespace cufinufft
