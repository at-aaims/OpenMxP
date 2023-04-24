#ifndef __fp16GPUKer
#define __fp16GPUKer

#include <assert.h>
#include "panel.hpp"
//#include <cublas_v2.h>
//#include <cuda_runtime.h>
#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include "hpl_rand.hpp"
#include "grid.hpp"
//#include <chrono>

#include "device_macros.h"

using namespace std;


__global__ void copyCoalesced(__half *odata, const float *idata, int olda, int ilda);
/// HUGE ASSUMTION OF ilda = olda for copy lower and upper
__global__ void copyCoalesced_lower(__half *odata, const float *idata, int olda, int ilda);
__global__ void copyCoalesced_upper(__half *odata, const float *idata, int olda, int ilda);
__global__ void copyCoalesced_lower(float *odata, const __half *idata, int olda, int ilda);
__global__ void copyCoalesced_upper(float *odata, const __half *idata, int olda, int ilda);
__global__ void transposeCoalesced(__half *odata, const float *idata, int olda, int ilda);

// HOST CALLS TO LAUNCH KERNELS (LEFT)
__host__ void half_conversion_left( const float *C, int b, int plda, __half *lp, int lplda);

// HOST CALLS TO LAUNCH KERNELS (RIGHT TRANS)
__host__ void half_conversion_right_trans(const float *C, int b, int plda, __half * rp, int rplda);

// Copy cast kernel
__global__ void copyFtoH(__half *odata, const float *idata, long olda, long ilda);
__host__ void downCast_copy_general(__half* out, long olda, long nrow, long ncol, float* C, long ilda);
__host__ void downCast_copy_lower(__half* out, long olda, long nrow, long ncol, float* C, long ilda);
__host__ void downCast_copy_upper(__half* out, long olda, long nrow, long ncol, float* C, long ilda);

// Debug purpose
__host__ void upCast_copy_lower(float* out, long olda, long nrow, long ncol, __half * C, long ilda);
__host__ void upCast_copy_upper(float* out, long olda, long nrow, long ncol, __half * C, long ilda);

// Alt solver
__host__ void gen_identity_mat( float * out, long nrow, long ncol);
__global__ void gen_identity_mat_kernel( float * out, long nrow, long ncol);
__global__ void gen_identity_mat_kernel( __half * out, long nrow, long ncol);
__host__ void gen_identity_mat( __half * out, long nrow, long ncol);

// trans cast kernel
__host__ void downCast_trans_general(__half* out, long olda, long nrow, long ncol , float* C, long ilda);
__host__ void downCast_trans_general(__half* out, int olda, int nrow, int ncol , float* C, int ilda);


#define VECTOR_OP_THREADS 128
template<typename F>
void copycolv_h(Panels<F>const& p, double const* x, double* y );

__global__ void divcolv_d(int b, double const* __restrict__ x, double* __restrict__ y);

template<typename F>
void divcolv_h(Panels<F>const& p, double const* x, double* y );

__global__ void colv2rowv_d(int b, double const* __restrict__ x, double* __restrict__ y );

template<typename F>
void colv2rowv_h(Panels<F>const& p, double const* x, double* y );

__global__ void addcolv_d(int b, double const* __restrict__ x, double* __restrict__ y);

template<typename F>
void addcolv_h(Panels<F>const& p, double const* x, double* y );



#define MV_THREADS 256
__global__ void otf_gemv_diag_d(int n, int b, double alpha, double const* __restrict__ x, double * y, 
        RandStat stat_ij, RandCoeff incl1, RandCoeff jump_thread, double const* diag);

__global__ void otf_gemv_d(int n, int b, double alpha, double const* __restrict__ x, double * y,
        RandStat stat_ij, RandCoeff incl1, RandCoeff jump_thread);

template<typename FPanel>
void otf_gemv_h(Matgen<double> const& mg, Panels<FPanel>const& p,
                int rowstart, int rowend, int colstart, int colend, double alpha, double const* x, double* y);



#define NORM_THREADS 1024
__global__ void calc_infnorm_d(int b, double* __restrict__ x, double* __restrict__ result);

template <typename F>
double colv_infnorm_h(Panels<F> const &p, double *dx, Grid &g, double* workspace);


// Trsv
#define TILE_DIM_TRSV 32
#define BLOCK_ROWS_TRSV 4
#define TRSV_MV_THREADS 128
#define GEMV_USE_MED_KNL 1

template <int DIM_X, int DIM_Y, typename T, typename U>
static __global__
void gemv_kernel(const int m,
                     const int n,
                     const U   alpha,
                     const T* __restrict__ A,
                     const int lda,
                     const U* __restrict__ x,
                     const int incx,
                     const U   beta,
                     U* __restrict__ y,
                     const int incy);

template <typename T, typename U>
void HPLMXP_gemv(const int m,
                 const int n,
                 const T   alpha,
                 const U*  A,
                 const int lda,
                 const T*  x,
                 const T   beta,
                 T*        y);

template <typename Fx, typename Fy>
__global__ void convert_precision (Fy * odata, const Fx * idata, size_t olda,
                              size_t ilda);

template <typename Fx, typename Fy>
__host__ void cast_copy_general(Fy* out, size_t olda, size_t nrow, size_t ncol , Fx* C, size_t ilda);

template <typename Fx, typename Fy>
__global__ void matrix_vector_multiply_0(int rows, int cols, double alpha,
        Fx const* __restrict__ A, int lda, Fy * __restrict__ x, 
        Fy * __restrict__ y);

#define MATGEM_THREADS 512
template <typename T>
__global__ void fill_panel_diag_dev(int n, int b, T * __restrict__ A, int lda, RandStat stat_ij, RandCoeff incl1,
        RandCoeff jump_thread, RandStat stat_00, int row_start, double* Diag);
template <typename T>
__global__ void fill_panel_dev(int n, int b, T * __restrict__ A,
        int lda, RandStat stat_ij, RandCoeff incl1, RandCoeff jump_thread);
template <typename F>
void panel_matgen_dev(Matgen<F> const& mg, Panels<F>& p, double* Diag);


#endif


