
#ifndef PANEL_GEMV_HPP
#define PANEL_GEMV_HPP
#include "highammgen.hpp"
#include "hpl_rand.hpp"
#include "panel.hpp"
#include "grid.hpp"
#include "timer.hpp"
#include <mpi.h>


#define MV_THREADS 256
__global__ void otf_gemv_diag_d(int n, int b, double alpha, double const* __restrict__ x, double * y, 
        RandStat stat_ij, RandCoeff incl1, RandCoeff jump_thread, double const* diag)
{
    int cur_row = blockIdx.x;
    int id = threadIdx.x;
    __shared__ double sh[MV_THREADS];
    double t = 0;
    double d = diag[cur_row];

    if (cur_row < b) {
        RandCoeff jump_id = incl1.pow(id * n + cur_row);
        RandStat stat_tj = jump_id*stat_ij;
        for (int j = 0; j + id < b; j += blockDim.x) {
            double aij = (cur_row == j + id ? d : static_cast<double>(stat_tj));
            t += aij*x[j+id];;
            stat_tj = jump_thread * stat_tj;
        }
    }
    sh[id] = t;
    __syncthreads();

    if (id < 128) sh[id] += sh[id + 128];
    __syncthreads();
    if (id < 64) sh[id] += sh[id + 64];
    __syncthreads();
    if (id < 32) sh[id] += sh[id + 32];
    __syncthreads();
    if (id < 16) sh[id] += sh[id + 16];
    __syncthreads();
    if (id < 8) sh[id] += sh[id + 8];
    __syncthreads();
    if (id < 4) sh[id] += sh[id + 4];
    __syncthreads();
    if (id == 0) y[cur_row] = y[cur_row] + alpha * (sh[0] + sh[1] + sh[2] + sh[3]);
}

__global__ void otf_gemv_d(int n, int b, double alpha, double const* __restrict__ x, double * y,
        RandStat stat_ij, RandCoeff incl1, RandCoeff jump_thread)
{
    int cur_row = blockIdx.x;
    int id = threadIdx.x;
    __shared__ double sh[MV_THREADS];
    double t = 0;

    if (cur_row < b) {
        RandCoeff jump_id = incl1.pow(id * n + cur_row);
        RandStat stat_tj = jump_id*stat_ij;
        for (int j = 0; j + id < b; j += blockDim.x) {
            t += static_cast<double>(stat_tj)*x[j+id];
            stat_tj = jump_thread * stat_tj;
        }
    }
    sh[id] = t;
    __syncthreads();

    if (id < 128) sh[id] += sh[id + 128];
    __syncthreads();
    if (id < 64) sh[id] += sh[id + 64];
    __syncthreads();
    if (id < 32) sh[id] += sh[id + 32];
    __syncthreads();
    if (id < 16) sh[id] += sh[id + 16];
    __syncthreads();
    if (id < 8) sh[id] += sh[id + 8];
    __syncthreads();
    if (id < 4) sh[id] += sh[id + 4];
    __syncthreads();
    if (id == 0) y[cur_row] = y[cur_row] + alpha * (sh[0] + sh[1] + sh[2] + sh[3]);
}

template<typename FPanel>
void otf_gemv_h(Matgen<double> const& mg, Panels<FPanel>const& p,
                int rowstart, int rowend, int colstart, int colend, double alpha, double const* x, double* y)
{
    int const n = mg.n;
    int const b = p.b;
    int const i1 = p.i1;
    int const j1 = p.j1;
    int const istride = p.istride;
    int const jstride = p.jstride;
    int const nprow = p.nprow;
    int const npcol = p.npcol;
    RandCoeff incl1 = mg.incl1;
    RandStat stat_00 = RandStat::initialize(mg.seed);
    RandCoeff jump_thread = incl1.pow(MV_THREADS * n);
    double const* diag = mg.diag_dev;

    for (int pj = colstart; pj < colend; ++pj) {
        int j0 = j1 + pj * jstride;

        for (int pi = 0; pi < nprow; ++pi) {
            int i0 = i1 + pi * istride;
            RandCoeff jump_ij = mg.jump(b * static_cast<uint64_t>(i0), b * static_cast<uint64_t>(j0));
            RandStat stat_ij = jump_ij * stat_00;

            if (i0 != j0) {
                otf_gemv_d<<<b, MV_THREADS, 0>>>(n, b, alpha, x + b * pj, y + b * pi, stat_ij, incl1, jump_thread);
            }
            else {
                otf_gemv_diag_d<<<b, MV_THREADS, 0>>>(n, b, alpha, x + b * pj, y + b * pi, stat_ij, incl1, jump_thread, diag + b * pi);
            }
            GPU_THREAD_SYNCHRONIZE(0);
        }
    }
}

template <typename FPanel, typename FAcc, template<class> class Matgen>
void panel_gemv_h(FAcc alpha, Panels<FPanel>const& p, Matgen<FAcc>const& mg, 
        bool x_is_full, FAcc* x, FAcc beta, FAcc* y, Grid& grid)
{
    int const b = p.b;
    int const i1 = p.i1;
    int const j1 = p.j1;
    int const istride = p.istride;
    int const jstride = p.jstride;
    int const nprow = p.nprow;
    int const npcol = p.npcol;
    FAcc const dzero = static_cast<FAcc> (0);
    FAcc *sx = NULL;

    // first: initialize y data
    for (int i = 0; i < nprow; ++i) {
        int ipos = i1 + i * istride;

        if ((ipos % jstride) == j1) {
            GPU_dscal(p.cuHandle, b, &beta, y+i*b, 1);
        }
        else {
            GPU_setValue(y+i*b, dzero, b*sizeof(double)); 
        }
    }
    sx = x;

    if (x_is_full || grid.nrow == 1) {
        // easy case. just call GEMV.
        Timer::beg (Timer::IR_GEMV);
        otf_gemv_h (mg, p, 0, nprow, 0, npcol, alpha, x, y);
        Timer::end (Timer::IR_GEMV, false, 2ull * nprow * npcol * b * b);
    }
    else {
        //printf("hard case\n");
        // multi-step bcast gemv
        MPI_Request req[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
        int rootrow = (j1) % istride;
        GPU_THREAD_SYNCHRONIZE(0);
        Timer::beg (Timer::IR_GEMV_COMM);
        if (npcol > 0) MPI_Ibcast (x, b, T2MPI<FAcc>::type, rootrow, grid.vcomm, req);
        Timer::end (Timer::IR_GEMV_COMM);

        for (int j = 0; j < npcol; ++j) {
            int nextrootrow = (j1 + (j + 1) * jstride) % istride;
            Timer::beg (Timer::IR_GEMV_COMM, true);
            if (j != npcol - 1) MPI_Ibcast (x + b * (j + 1), b, T2MPI<FAcc>::type, nextrootrow, grid.vcomm, req + (j + 1) % 2);
            MPI_Wait (req + j % 2, MPI_STATUS_IGNORE);
            Timer::end (Timer::IR_GEMV_COMM);

            Timer::beg (Timer::IR_GEMV, true);
            otf_gemv_h (mg, p, 0, nprow, j, j + 1, alpha, x, y);
            Timer::end (Timer::IR_GEMV, false, 2ull * nprow * b * b);
        }
        MPI_Waitall (2, req, MPI_STATUSES_IGNORE);
    }
    GPU_THREAD_SYNCHRONIZE(0);
    Timer::beg (Timer::IR_GEMV_COMM, true);
    MPI_Allreduce (MPI_IN_PLACE, y, b * nprow, T2MPI<FAcc>::type, MPI_SUM, grid.hcomm);
    Timer::end (Timer::IR_GEMV_COMM);
}




template <typename FPanel>
static void mem_gemv_kernel(int mb, int nb, double alpha,
                                double const *__restrict__ x,
                                double *__restrict__ y,
                                Panels<FPanel> const &p, uint64_t row, uint64_t col) {
    // memory-based GEMV computes y = y + alpha * A * x
    // n is the dimension of the whole matrix.
    // mb \times nb is the dimension of the sub-matrix to compute.
    // Note that the sub-matrix cannot have diagonals.
    uint64_t p_idx, p_idx0 = p.lda*col + row;
    for (uint64_t i = 0; i < mb; i++)
    {
        double t = 0.;
        p_idx = p_idx0 + i;
        for (uint64_t j = 0; j < nb; j++)
        {
            t += p.p_init[p_idx] * x[j];
            p_idx += p.lda;
        }
        y[i] += alpha * t;
    }
}

// on-the-fly computation of gemv.
extern "C" void otf_gemv_kernel(int64_t n, int mb, int nb, double alpha,
                                double const *__restrict__ x,
                                double *__restrict__ y, uint64_t seed);
extern "C" void hmg_gemv_up(int i, double a, double b, int mb, int nb,
                            double alpha, double const *__restrict__ x,
                            double *__restrict__ y);
extern "C" void hmg_gemv_low(int j, double a, double b, int mb, int nb,
                             double alpha, double const *__restrict__ x,
                             double *__restrict__ y);
extern "C" void hmg_gemv_diag(int i, int j, double a, double b, int mb, int nb,
                              double alpha, double const *__restrict__ x,
                              double *__restrict__ y);

static void otf_gemv_small(int b, double alpha, double const *__restrict__ x,
                           double *__restrict__ y,
                           double const *__restrict__ diag, RandStat stat_00,
                           RandCoeff incl1, RandCoeff jumpn) {
    for (int i = 0; i < b; ++i) {
        RandStat stat_j = stat_00;
        double d = diag[i];
        double t = 0.;
        for (int j = 0; j < b; ++j) {
            double aij = (i == j ? d : static_cast<double>(stat_j));
            t += aij * x[j];
            stat_j = jumpn * stat_j;
        }
        y[i] += alpha * t;
        stat_00 = incl1 * stat_00;
    }
}

template <typename FPanel>
static void mem_gemv_small(int b, double alpha, double const *__restrict__ x,
                           double *__restrict__ y,
                           double const *__restrict__ diag,
                           Panels<FPanel> const &p, long row, long col) {
    uint64_t p_idx, p_idx0 = p.lda*col + row;
    for (int i = 0; i < b; ++i) {
        //double d = diag[i];
        double t = 0.;
        p_idx = p_idx0 + i;
        for (int j = 0; j < b; ++j) {
            // p[row + i, col + j]
            double aij = p.p_init[p_idx];//(i == j ? d : p.p_init[p_idx]);
            t += aij * x[j];
            p_idx += p.lda;
        }
        y[i] += alpha * t;
    }
}

static void otf_gemv_kernel_diag(int64_t n, int b, double alpha,
                                 double const *__restrict__ x,
                                 double *__restrict__ y,
                                 double const *__restrict__ diag,
                                 RandStat stat_00, RandCoeff incl1,
                                 RandCoeff jumpn) {
    int microb = 64;
    RandCoeff jumpmb = incl1.pow(microb);
    RandCoeff jumpmbn = jumpn.pow(microb);
    RandCoeff jumppi = jumpn.pow(0);
    for (int pi = 0; pi < b; pi += microb) {
        RandStat stat_j = stat_00;
        int bi = (b - pi < microb ? b - pi : microb);
        if (pi) {
            otf_gemv_kernel(n, bi, pi, alpha, x, y + pi, stat_j.x);
            stat_j = jumppi * stat_j;
        }
        otf_gemv_small(bi, alpha, x + pi, y + pi, diag + pi, stat_j, incl1,
                       jumpn);
        if (b > pi + microb) {
            stat_j = jumpmbn * stat_j;
            otf_gemv_kernel(n, bi, b - pi - microb, alpha, x + pi + microb,
                            y + pi, stat_j.x);
        }

        stat_00 = jumpmb * stat_00;
        jumppi = jumppi * jumpmbn;
    }
}

// row, col are coords in p
// diag is offset by row, i.e. so that first b entries correspond to the block
template <typename FPanel>
static void mem_gemv_kernel_diag(int b, double alpha,
                                 double const *__restrict__ x,
                                 double *__restrict__ y,
                                 double const *__restrict__ diag,
                                 Panels<FPanel> const &p, long row, long col) {
    int microb = 64;
    for (int pi = 0; pi < b; pi += microb) {
        int bi = (b - pi < microb ? b - pi : microb);
        if (pi) // not first iteration
            mem_gemv_kernel(bi, pi, alpha, x, y + pi, p, row + pi, col);
        mem_gemv_small(bi, alpha, x + pi, y + pi, diag + pi, p, row + pi, col + pi);
        if (b > pi + microb) // not last iteration
            mem_gemv_kernel(bi, b - pi - microb, alpha, x + pi + microb,
                            y + pi, p, row + pi, col + pi + microb);

    }
}

template <typename FPanel>
void otf_gemv(Matgen<double> const &mg, Panels<FPanel> const &p, int rowstart,
              int rowend, int colstart, int colend, double alpha,
              double const *x, double *y) {
    // assuming x is full row.
    int const b = p.b;
    int const i1 = p.i1;
    int const j1 = p.j1;
    int const istride = p.istride;
    int const jstride = p.jstride;
    RandCoeff incl1 = mg.incl1;
    RandCoeff jumpn = mg.jumpn;
    double const *diag = mg.diag;
    for (int pj = colstart; pj < colend; ++pj) {
        int j0 = j1 + pj * jstride;
#pragma omp parallel for schedule(dynamic, 1)
        for (int pi = rowstart; pi < rowend; ++pi) {
            int i0 = i1 + pi * istride;
            RandCoeff jump_ij = mg.jump(b * i0, b * j0);
            RandStat stat_00 = jump_ij * RandStat::initialize(mg.seed);
            if (i0 != j0)
                otf_gemv_kernel(mg.n, b, b, alpha, x + b * pj, y + b * pi,
                                stat_00.x);
            else
                otf_gemv_kernel_diag(mg.n, b, alpha, x + b * pj, y + b * pi,
                                     diag + b * pi, stat_00, incl1, jumpn);
        }
    }
}

template <typename FPanel>
void mem_gemv(Matgen<double> const &mg, Panels<FPanel> const &p, int rowstart,
              int rowend, int colstart, int colend, double alpha,
              double const *x, double *y) {
    long const b = p.b;
    long const i1 = p.i1;
    long const j1 = p.j1;
    long const istride = p.istride;
    long const jstride = p.jstride;
    double const *diag = mg.diag;

    for (long pj = colstart; pj < colend; ++pj) { // pj is block col
#pragma omp parallel for schedule(dynamic, 1)
        for (long pi = rowstart; pi < rowend; ++pi) { // pi is block row
            if (i1 + pi * istride != j1 + pj * jstride)
                mem_gemv_kernel(b, b, alpha, x + b * pj, y + b * pi, p, b*pi, b*pj);
            else
                mem_gemv_kernel_diag(b, alpha, x + b * pj, y + b * pi, diag + b * pi, p, b*pi, b*pj);
        }
    }
}

template <typename FPanel>
void otf_gemv(HMGen<double> const &mg, Panels<FPanel> const &p, int rowstart,
              int rowend, int colstart, int colend, double alpha,
              double const *x, double *y) {
    // assuming x is full row.
    int const b = p.b;
    int const i1 = p.i1;
    int const j1 = p.j1;
    int const istride = p.istride;
    int const jstride = p.jstride;
    for (int pj = colstart; pj < colend; ++pj) {
        int j0 = j1 + pj * jstride;
#pragma omp parallel for schedule(dynamic, 1)
        for (int pi = rowstart; pi < rowend; ++pi) {
            int i0 = i1 + pi * istride;
            if (i0 < j0)
                hmg_gemv_up(b * i0, mg.alpha, mg.beta, b, b, alpha, x + b * pj,
                            y + b * pi);
            else if (i0 > j0)
                hmg_gemv_low(b * j0, mg.alpha, mg.beta, b, b, alpha, x + b * pj,
                             y + b * pi);
            else
                hmg_gemv_diag(b * i0, b * j0, mg.alpha, mg.beta, b, b, alpha,
                              x + b * pj, y + b * pi);
        }
    }
}

template <typename FPanel, typename FAcc, template <class> class Matgen>
void panel_gemv(int my_init, FAcc alpha, Panels<FPanel> const &p, Matgen<FAcc> const &mg,
                bool x_is_full, FAcc *x, FAcc beta, FAcc *y, Grid &grid) {
    // compute y = beta*y + alpha * p * x
    // it can be p.nprow != p.npcol.

    // in: p
    // inout: x(M), y(M)
    // where M = max(p.nprow, p.npcol) * b

    // x is the (partial) row vector. x_j = X_{j1+(j-1)*jstride}. where
    // 1<=j<=npcol. x_j is valid iff j0+(j-1)*jstride >= i0 and
    // (j0+(j-1)*jstride-i0)%istride == 0. If !!x_is_valid, it is assumed that
    // all the data is valid. x was modified and become full at the end of the
    // process

    // y is the (partial) column vector. x_i = Y_{i1+(i-1)*istride}, where
    // 1<=i<=nprow. y_i is valid iff i0+(i-1)*istride >= j0 and
    // (i0+(i-1)*istride-j0)%jstride == 0. Other part of vector are invalid;
    // they have arbitrary values but MODIFIED after the computation.

    int const b = p.b;
    int const i1 = p.i1;
    int const j1 = p.j1;
    int const istride = p.istride;
    int const jstride = p.jstride;
    int const nprow = p.nprow;
    int const npcol = p.npcol;
    FAcc const dzero = static_cast<FAcc>(0);
    // first: initialize y data
    for (int i = 0; i < nprow; ++i) {
        int ipos = i1 + i * istride;
        if ((ipos % jstride) == j1) {
            // it is a valid block
            #pragma omp parallel for simd
            for (int k = i * b; k < i * b + b; ++k)
                y[k] *= beta;
        } else {
            // it is an invalid block
            #pragma omp parallel for simd
            for (int k = i * b; k < i * b + b; ++k)
                y[k] = dzero;
        }
    }

    if (x_is_full || grid.nrow == 1) {
        // easy case. just call GEMV.
        Timer::beg(Timer::IR_GEMV);
        if (my_init == 2001)
            mem_gemv(mg, p, 0, nprow, 0, npcol, alpha, x, y);
        else
            otf_gemv(mg, p, 0, nprow, 0, npcol, alpha, x, y);
        Timer::end(Timer::IR_GEMV, false, 2ull * nprow * npcol * b * b);
    } else {
        // multi-step bcast gemv
        MPI_Request req[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
        int rootrow = (j1) % istride;
        Timer::beg(Timer::IR_GEMV_COMM);
        if (npcol > 0)
            MPI_Ibcast(x, b, T2MPI<FAcc>::type, rootrow, grid.vcomm, req);
        Timer::end(Timer::IR_GEMV_COMM);
        for (int j = 0; j < npcol; ++j) {
            int nextrootrow = (j1 + (j + 1) * jstride) % istride;
            Timer::beg(Timer::IR_GEMV_COMM, true);
            if (j != npcol - 1)
                MPI_Ibcast(x + b * (j + 1), b, T2MPI<FAcc>::type, nextrootrow,
                           grid.vcomm, req + (j + 1) % 2);
            MPI_Wait(req + j % 2, MPI_STATUS_IGNORE);
            Timer::end(Timer::IR_GEMV_COMM);
            Timer::beg(Timer::IR_GEMV, true);
            if (my_init ==20001)
                mem_gemv(mg, p, 0, nprow, j, j + 1, alpha, x, y);
            else
                otf_gemv(mg, p, 0, nprow, j, j + 1, alpha, x, y);
            Timer::end(Timer::IR_GEMV, false, 2ull * nprow * b * b);
        }
        MPI_Waitall(2, req, MPI_STATUSES_IGNORE);
    }
    Timer::beg(Timer::IR_GEMV_COMM, true);
    MPI_Allreduce(MPI_IN_PLACE, y, b * nprow, T2MPI<FAcc>::type, MPI_SUM,
                  grid.hcomm);
    Timer::end(Timer::IR_GEMV_COMM);
}

template <typename FPanel, typename FAcc, template<class> class Matgen>
void panel_gemv(FAcc alpha, Panels<FPanel>const& p, Matgen<FAcc>const& mg, bool x_is_full, FAcc* x, FAcc beta, FAcc* y, Grid& grid)
{
    int const b = p.b;
    int const i1 = p.i1;
    int const j1 = p.j1;
    int const istride = p.istride;
    int const jstride = p.jstride;
    int const nprow = p.nprow;
    int const npcol = p.npcol;
    FAcc const dzero = static_cast<FAcc> (0);

    // first: initialize y data
    for (int i = 0; i < nprow; ++i) {
        int ipos = i1 + i * istride;

        if ((ipos % jstride) == j1) {
            // it is a valid block
            #pragma omp parallel for simd
            for (int k = i * b; k < i * b + b; ++k) y[k] *= beta;
        }
        else {
            // it is an invalid block
            #pragma omp parallel for simd
            for (int k = i * b; k < i * b + b; ++k) y[k] = dzero;
        }
    }

    if (x_is_full || grid.nrow == 1) {
        // easy case. just call GEMV.
        Timer::beg (Timer::IR_GEMV);
        otf_gemv (mg, p, 0, nprow, 0, npcol, alpha, x, y);
        Timer::end (Timer::IR_GEMV, false, 2ull * nprow * npcol * b * b);
    }
    else {
        // multi-step bcast gemv
        MPI_Request req[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
        int rootrow = (j1) % istride;
        
        Timer::beg (Timer::IR_GEMV_COMM);
        if (npcol > 0) MPI_Ibcast (x, b, T2MPI<FAcc>::type, rootrow, grid.vcomm, req);
        Timer::end (Timer::IR_GEMV_COMM);

        for (int j = 0; j < npcol; ++j) {
            int nextrootrow = (j1 + (j + 1) * jstride) % istride;

            Timer::beg (Timer::IR_GEMV_COMM, true);
            if (j != npcol - 1) MPI_Ibcast (x + b * (j + 1), b, T2MPI<FAcc>::type, nextrootrow, grid.vcomm, req + (j + 1) % 2);
            MPI_Wait (req + j % 2, MPI_STATUS_IGNORE);
            Timer::end (Timer::IR_GEMV_COMM);

            Timer::beg (Timer::IR_GEMV, true);
            otf_gemv (mg, p, 0, nprow, j, j + 1, alpha, x, y);
            Timer::end (Timer::IR_GEMV, false, 2ull * nprow * b * b);
        }

        MPI_Waitall (2, req, MPI_STATUSES_IGNORE);
    }

    Timer::beg (Timer::IR_GEMV_COMM, true);
    MPI_Allreduce (MPI_IN_PLACE, y, b * nprow, T2MPI<FAcc>::type, MPI_SUM, grid.hcomm);
    Timer::end (Timer::IR_GEMV_COMM);
}


#endif
