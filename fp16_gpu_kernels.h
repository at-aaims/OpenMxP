#ifndef __fp16GPUKer
#define __fp16GPUKer

#include <assert.h>
//#include <cublas_v2.h>
//#include <cuda_runtime.h>
#include <iostream>
#include <stdlib.h>
#include <unistd.h>
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

#endif


