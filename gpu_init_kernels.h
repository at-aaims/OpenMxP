#ifndef GPU_INIT_KERNELS
#define GPU_INIT_KERNELS
#include <unistd.h>
#include <iostream>
#include <stdlib.h>
#include <assert.h>

#include "hpl_rand.hpp"
#include "device_macros.h"

#define MATGEM_THREADS 512

using namespace std;
/*
template <typename T>
void fill_panel_diag_host(int n, int b, T *A, int lda, RandStat stat_ij, RandCoeff incl1,
        RandCoeff jump_thread, int rowstart, RandStat stat_00, double* Diag, int nthreads);

template <typename T>
void fill_panel_host(int n, int b, T *A, int lda,
        RandStat stat_ij, RandCoeff incl1, RandCoeff jump_thread, int nthreads );

template <typename T>
__global__ void fill_panel_dev(int n, int b, T * __restrict__ A,
        int lda, RandStat stat_ij, RandCoeff incl1, RandCoeff jump_dn);

template <typename T>
__global__ void fill_panel_diag_dev(int n, int nb, T * __restrict__ A, int lda, RandStat stat_ij, RandCoeff incl1,
        RandCoeff jump_dn, RandStat stat_00, int rowstart, double* Diag);
*/

void fill_random(float *A, long m, long n, int n_threads, int blocksize_x, int work_per_thread);

void fill_random(double *v, long m, int n_threads);

__host__ void fill_random_fugaku(uint64_t N, RandStat stat0, RandCoeff inc1, float *A, double *row_sums, long m, long n, long i1, long j1, long b, long istride, long jstride, int n_threads, int blocksize_x, int work_per_thread);

template <typename F> __global__ void fill_random_fugaku_d(uint64_t N, RandStat stat0, RandCoeff inc1, F *A, double *row_sums, long m, long n, long i1, long j1, long b, long istride, long jstride, int work_per_thread);
    
__host__ void compute_row_sums(float *A, double *row_sums, long m, long n, long i1, long j1, long b, long istride, long jstride, int n_threads, int blocksize_x, int work_per_thread);

__global__ void compute_row_sums_d(float *A, double *row_sums, long m, long n, long i1, long j1, long b, long istride, long jstride, int work_per_thread);

__host__ void fill_diag_rhs(int my_init, uint64_t N, RandStat stat0, RandCoeff inc1, float *A, double *row_sums_d, double *rhs, double *rhs_d, long m, long *diag_i_steps, long *diag_j_steps, long n_diag_blocks, long i1, long b, long istride, int n_threads);

__global__ void fill_diag_rhs_d(int my_init, uint64_t N, RandStat stat0, RandCoeff inc1, float *A, double *row_sums_d, double *rhs, double *rhs_d, long m, long *diag_i_steps, long *diag_j_steps, long n_diag_entries, long i1, long b, long istride);

template <typename F> __global__ void minus_05(F *A, long m, long n, int work_per_thread);

#endif
