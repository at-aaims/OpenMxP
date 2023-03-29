#ifndef PANEL_TRF_HPP
#define PANEL_TRF_HPP

#include <cmath>

#include <mpi.h>

#include "grid.hpp"
#include "hpl_rand.hpp"
#include "panel.hpp"
#include "getrf_nopiv.hpp"

#include "timer.hpp"
#include "device_macros.h"


struct RequestStack {
    int nreq;
    int maxnreq;
    MPI_Request *reqs;
    RequestStack(int num) : nreq(0), maxnreq(num) {
        reqs = (MPI_Request *)malloc(sizeof(MPI_Request) * maxnreq);
    }
    ~RequestStack() { free(reqs); }
    RequestStack(RequestStack const &) = delete;
    RequestStack &operator=(RequestStack const &) = delete;
    MPI_Request *get_request() {
        if (nreq == maxnreq)
            std::abort(); // Die imidiately
        return reqs + nreq++;
    }
    void wait_all() {
        if (!nreq)
            return;
        MPI_Waitall(nreq, reqs, MPI_STATUSES_IGNORE);
        nreq = 0;
    }
    bool test_all() {
        if (nreq == 0)
            return true;
        int flags = 0;
        MPI_Testall(nreq, reqs, &flags, MPI_STATUSES_IGNORE);
        if (flags) {
            nreq = 0;
            return true;
        } else
            return false;
    }
    void wait_all(Timer::Items item) {
        Timer::beg(item);
        wait_all();
        Timer::end(item);
    }
    bool test_all(Timer::Items item) {
        Timer::beg(item);
        bool ret = test_all();
        Timer::end(item);
        return ret;
    }
};

// communications
template <typename F>
void broadcast_pivlu(bool is_tile, int b, F *lu, size_t lda, F *piv,
                     size_t ldpiv, Grid &grid, RequestStack &req) {
    // broadcast the pivot block to row and column

    Timer::beg(Timer::DIAG_BCAST);
    req.wait_all();
    if (is_tile) {
        MPI_Ibcast(const_cast<F *>(lu), lda * b, T2MPI<F>::type, grid.row,
                   grid.vcomm, req.get_request());
        MPI_Ibcast(const_cast<F *>(lu), lda * b, T2MPI<F>::type, grid.col,
                   grid.hcomm, req.get_request());
    } else {
        MPI_Ibcast(piv, ldpiv * b, T2MPI<F>::type, grid.row, grid.vcomm,
                   req.get_request());
        MPI_Ibcast(piv, ldpiv * b, T2MPI<F>::type, grid.col, grid.hcomm,
                   req.get_request());
#pragma omp parallel for
        for (int j = 0; j < b; ++j)
            for (int i = 0; i < b; ++i)
                lu[j * lda + i] = piv[j * ldpiv + i];
    }
    Timer::end(Timer::DIAG_BCAST);
}

template <typename F>
void broadcast_pivlu_wGPU(bool is_tile, int b, F *lu, size_t lda, F *piv,
                          size_t ldpiv, Grid &grid, RequestStack &req) {
    // broadcast the pivot block to row and column

    Timer::beg(Timer::DIAG_BCAST);
    req.wait_all();
    if (is_tile) {
        MPI_Ibcast(const_cast<F *>(lu), lda * b, T2MPI<F>::type, grid.row,
                   grid.vcomm, req.get_request());
        MPI_Ibcast(const_cast<F *>(lu), lda * b, T2MPI<F>::type, grid.col,
                   grid.hcomm, req.get_request());
    } else {
        MPI_Ibcast(piv, ldpiv * b, T2MPI<F>::type, grid.row, grid.vcomm,
                   req.get_request());
        MPI_Ibcast(piv, ldpiv * b, T2MPI<F>::type, grid.col, grid.hcomm,
                   req.get_request());
    }
    Timer::end(Timer::DIAG_BCAST);
}

template <typename F>
void receive_pivu(int b, F *piv, int ldpiv, int root, Grid &grid,
                  RequestStack &req) {
    Timer::beg(Timer::DIAG_BCAST);
    req.wait_all();
    MPI_Ibcast(piv, ldpiv * b, T2MPI<F>::type, root, grid.vcomm,
               req.get_request());
    Timer::end(Timer::DIAG_BCAST);
}
template <typename F>
void receive_pivl(int b, F *piv, int ldpiv, int root, Grid &grid,
                  RequestStack &req) {
    Timer::beg(Timer::DIAG_BCAST);
    req.wait_all();
    MPI_Ibcast(piv, ldpiv * b, T2MPI<F>::type, root, grid.hcomm,
               req.get_request());
    Timer::end(Timer::DIAG_BCAST);
}

template <typename F>
void broadcast_left_panels(int b, const LRPanels<F> &lp, int rowstart, int nrow,
                           Grid &grid, RequestStack &req) {
    Timer::beg(Timer::LCOL_BCAST);
    // broadcast the left panels to row
    if (lp.is_tile)
        MPI_Ibcast(const_cast<F *>(lp(rowstart)), lp.ldp * (nrow - rowstart),
                   T2MPI<F>::type, grid.col, grid.hcomm, req.get_request());
    else
        MPI_Ibcast(const_cast<F *>(lp(rowstart)), lp.get_lda() * b,
                   T2MPI<F>::type, grid.col, grid.hcomm, req.get_request());
    Timer::end(Timer::LCOL_BCAST);
}
template <typename F>
void receive_left_panels(int b, LRPanels<F> &lp, int rowstart, int nrow,
                         int root, Grid &grid, RequestStack &req) {
    Timer::beg(Timer::LCOL_BCAST);
    lp.set_start(rowstart);
    if (lp.is_tile)
        MPI_Ibcast(const_cast<F *>(lp(rowstart)), lp.ldp * (nrow - rowstart),
                   T2MPI<F>::type, root, grid.hcomm, req.get_request());
    else
        MPI_Ibcast(const_cast<F *>(lp(rowstart)), lp.get_lda() * b,
                   T2MPI<F>::type, root, grid.hcomm, req.get_request());
    Timer::end(Timer::LCOL_BCAST);
}
template <typename F>
void broadcast_right_panels(int b, LRPanels<F> &rp, int colstart, int ncol,
                            Grid &grid, RequestStack &req) {
    Timer::beg(Timer::RROW_BCAST);
    MPI_Ibcast(const_cast<F *>(rp(colstart)), rp.get_lda() * b, T2MPI<F>::type,
               grid.row, grid.vcomm, req.get_request());
    Timer::end(Timer::RROW_BCAST);
}
template <typename F>
void receive_right_panels(int b, LRPanels<F> &rp, int colstart, int ncol,
                          int root, Grid &grid, RequestStack &req) {
    Timer::beg(Timer::RROW_BCAST);
    rp.set_start(colstart);
    MPI_Ibcast(const_cast<F *>(rp(colstart)), rp.get_lda() * b, T2MPI<F>::type,
               root, grid.vcomm, req.get_request());
    Timer::end(Timer::RROW_BCAST);
}

template <typename F>
void update_left_panels(const F *u, int ldu, Panels<F> &p, int rowstart,
                        int col) {
    if (rowstart == p.nprow)
        return;
    Timer::beg(Timer::TRSM_L);
    if (p.is_tile) {
        for (int i = rowstart; i < p.nprow; ++i) {
            trsmR(p.b, p.b, u, ldu, p(i, col), p.lda);
        }
    } else {
        trsmR(p.b * (p.nprow - rowstart), p.b, u, ldu, p(rowstart, col), p.lda);
    }
    Timer::end(Timer::TRSM_L, false, 1ll * p.b * (p.nprow - rowstart) * p.b * p.b);
}

template <typename F>
inline void update_left_panels_wGPU(const F *u, int ldu, Panels<F> &p, int rowstart,
                             int col)
{
    if (rowstart == p.nprow)
        return;

    Timer::beg(Timer::TRSM_L);
    F alf = 1.0f;
    const F *alpha = &alf;
    if ( !CHECK_BIT ( p.skip, 2 ) )
    {
       checkGPUblas(GPUBLAS_STRSM( p.cuHandle,
		    GPUBLAS_SIDE_RIGHT, GPUBLAS_FILL_MODE_UPPER, GPUBLAS_OP_N, GPUBLAS_DIAG_NON_UNIT,
		    (p.nprow - rowstart) * p.b, p.b, alpha, u, ldu,
                    p(rowstart, col, 'd'), p.d_lda));
//     GPU_DEVICE_SYNCHRONIZE();
    }
    Timer::end(Timer::TRSM_L, false, 1ll * p.b * (p.nprow - rowstart) * p.b * p.b);
}

template <typename F>
void update_right_panels(F const *l, int ldl, Panels<F> &p, int row,
                         int colstart) {
    if (colstart == p.npcol)
        return;
    Timer::beg(Timer::TRSM_R);
    if (p.is_tile) {
        for (int j = colstart; j < p.npcol; ++j) {
            trsmL(p.b, p.b, l, ldl, p(row, j), p.lda);
        }
    } else {
        trsmL(p.b, p.b * (p.npcol - colstart), l, ldl, p(row, colstart), p.lda);
    }
    Timer::end(Timer::TRSM_R, false, 1ll * p.b * p.b * p.b * (p.npcol - colstart));
}

template <typename F>
inline void update_right_panels_wGPU(F const *l, int ldl, Panels<F> &p, int row,
                              int colstart)
{
    if (colstart == p.npcol)
        return;
//  float alf = 1.0;
    F alf = 1.0;
    Timer::beg(Timer::TRSM_R);
//  const float *alpha = &alf;
    const F *alpha = &alf;

    if ( !CHECK_BIT ( p.skip, 2 ) ) 
    {
        checkGPUblas(GPUBLAS_STRSM( p.cuHandle,
		     GPUBLAS_SIDE_LEFT, GPUBLAS_FILL_MODE_LOWER, GPUBLAS_OP_N, GPUBLAS_DIAG_UNIT,
		     p.b, (p.npcol - colstart) * p.b, alpha, l, ldl,
                     p(row, colstart, 'd'), p.d_lda));
//      GPU_DEVICE_SYNCHRONIZE();
    }
    Timer::end(Timer::TRSM_R, false, 1ll * p.b * p.b * p.b * (p.npcol - colstart));
}

template <typename FHigh, typename FLow>
void convert_panel_impl(int b, FHigh scale, FHigh const *__restrict__ a,
                        size_t lda, FLow *__restrict__ to, size_t ldb) {
// we may change the type of a and b
#pragma omp parallel for
    for (int j = 0; j < b; ++j)
        for (int i = 0; i < b; ++i)
            to[j * ldb + i] = static_cast<FLow>(scale * a[j * lda + i]);
}

template <typename FHigh, typename FLow>
inline void convert_left_panels_wGPU(Panels<FHigh> const &p, FHigh scale, int rowstart,
                              int col, LRPanels<FLow> &lp)
{
    Timer::beg(Timer::CONV_L);
    lp.set_start(rowstart);

    int b = p.b;
    int nprow = p.nprow;
    size_t plda = p.d_lda;
    size_t lplda = lp.get_lda();
    FHigh const *pdata = p(rowstart, col, 'd');
    FLow *lpdata = lp(rowstart, 'd');

    half_conversion_left(pdata, b, plda, lpdata, lplda);

    //GPU_DEVICE_SYNCHRONIZE();

    Timer::end(Timer::CONV_L, false, 1ll * p.b * p.b * (p.nprow - rowstart));
}
template <typename FHigh, typename FLow>
void convert_left_panels(Panels<FHigh> const &p, FHigh scale, int rowstart,
                         int col, LRPanels<FLow> &lp) {
    Timer::beg(Timer::CONV_L);
    lp.set_start(rowstart);
    if (p.is_tile || lp.is_tile) {
        for (int i = rowstart; i < p.nprow; ++i) {
            convert_panel_impl(p.b, scale, p(i, col), p.lda, lp(i), lp.get_lda());
        }
    } else {
        int b = p.b;
        int nprow = p.nprow;
        size_t plda = p.lda;
        size_t lplda = lp.get_lda();
        FHigh const *pdata = p(rowstart, col);
        FLow *lpdata = lp(rowstart);
#pragma omp parallel for
        for (int j = 0; j < b; ++j) {
            for (int i = 0; i < b * (nprow - rowstart); ++i) {
                lpdata[j * lplda + i] =
                    static_cast<FLow>(scale * pdata[j * plda + i]);
            }
        }
    }
    Timer::end(Timer::CONV_L, false, 1ll * p.b * p.b * (p.nprow - rowstart));
}

template <typename FHigh, typename FLow>
inline void convert_right_panels_wGPU(Panels<FHigh> const &p, FHigh scale, int row,
                               int colstart, LRPanels<FLow> &rp)
{
    Timer::beg(Timer::CONV_R);
    rp.set_start(colstart);
     
    int b = p.b;
    int npcol = p.npcol;
    size_t plda = p.lda;
    size_t rplda = rp.get_lda();
    FHigh const *pdata = p(row, colstart, 'd');
    FLow *rpdata = rp(colstart, 'd');

    half_conversion_right_trans(pdata, p.b, p.lda, rpdata, rplda);
    // half_conversion_right_trans2(pdata, 0, 0, p.b, p.lda, rpdata, rplda, p.npcol*b);

    //GPU_DEVICE_SYNCHRONIZE();

    Timer::end(Timer::CONV_R, false, 1ll * p.b * (p.npcol - colstart) * p.b);
}

template <typename FHigh, typename FLow>
void convert_right_panels(Panels<FHigh> const &p, FHigh scale, int row,
                          int colstart, LRPanels<FLow> &rp) {
    Timer::beg(Timer::CONV_R);
    rp.set_start(colstart);
    if (p.is_tile) {
        for (int j = colstart; j < p.npcol; ++j) {
            convert_panel_impl(p.b, scale, p(row, j), p.lda, rp(j), rp.get_lda());
        }
    } else {
        int b = p.b;
        int npcol = p.npcol;
        size_t plda = p.lda;
        size_t rplda = rp.get_lda();
        FHigh const *pdata = p(row, colstart);
        FLow *rpdata = rp(colstart);
#pragma omp parallel for
        for (int j = 0; j < b * (npcol - colstart); ++j) {
            for (int i = 0; i < b; ++i) {
                rpdata[j * rplda + i] =
                    static_cast<FLow>(scale * pdata[j * plda + i]);
            }
        }
    }
    Timer::end(Timer::CONV_R, false, 1ll * p.b * (p.npcol - colstart) * p.b);
}







/***** Use for referrence ******
template <typename FHigh, typename FLow, template<class> class Matgen, int du,
bool dd> void panel_lu(Panels<FHigh> &p, LRPanels<FLow> lrpanels[4],
Matgen<FHigh>& mg, FHigh *piv, size_t ldpiv, Grid &grid, bool warmup=false)
{
        // easy implemnetation of LU decomp. for description purpose
        typedef LazyInitializer<FLow, du, dd> LI;
        typedef SchurUpdator<FHigh, FLow, dd> SU;
        int const nb = p.nblocks;
        int const b = p.b;
        size_t const lda = p.lda;
        int const nprow = p.nprow;
        int const npcol = p.npcol;
        LRPanels<FLow>& lp = lrpanels[0];
        LRPanels<FLow>& rp = lrpanels[1];
        RequestStack req(8);
        FHigh scalea = static_cast<FHigh>(1);
        FHigh scaleb = static_cast<FHigh>(1);
        FHigh downscale = static_cast<FHigh>(1./((double)scalea*scaleb));
        FHigh* buf = piv + ldpiv * b;
        int kend = warmup? 20 : nb;
        if(kend > nb) kend = nb;

        for (int k = 0; k < kend; ++k) {
                // position of the panels to decomp in process grid
                int const rootrow = k % grid.nrow;
                int const rootcol = k % grid.ncol;
                // position of the panels to decomp in local matrix
                int i = k / grid.nrow + (rootrow > grid.row ? 1 : 0);
                int j = k / grid.ncol + (rootcol > grid.col ? 1 : 0);
                if (rootrow == grid.row && rootcol == grid.col) {
                        // I have a pivot panel.
                        // 1) add the initial values to the partial sum (lazy
init) FHigh *lu = p(i, j); LI::update_diag(mg, p, downscale, i, j, buf, ldpiv);
                        // 2) lu decomp of the diagonal panel
                        if(p.is_tile){
                                getrf_nopiv(b, p(i, j), lda);
                        }else{
                                getrf_nopiv(b, p(i, j), lda, piv, ldpiv);
                        }
                        if (k == nb - 1) return;
                        // 3) broadcast it
                        broadcast_pivlu(p.is_tile, b, lu, lda, piv, ldpiv, grid,
req);

                        // 4) trsm row and column panels
                        // 4.1) lazy init
                        LI::update(mg, p, downscale, i+1, nprow, j, j+1, buf,
ldpiv);
                        // 4.2) trsm
                        update_left_panels(lu, p.lda, p, i + 1, j);
                        // 4.3) downcast from FHigh to FLow
                        convert_left_panels(p, scalea, i + 1, j, lp);
                        // 4.4) broadcast
                        broadcast_left_panels(b, lp, i + 1, nprow, grid, req);

                        // 5) same as 4
                        LI::update(mg, p, downscale, i, i+1, j+1, npcol, buf,
ldpiv); update_right_panels(lu, p.lda, p, i, j + 1); convert_right_panels(p,
scaleb, i, j + 1, rp); broadcast_right_panels(b, rp, j + 1, npcol, grid, req);
                        ++i;
                        ++j;
                }
                else if (rootrow == grid.row) {
                        if (k == nb - 1) return;
                        // I have a right panel.
                        // 1) lazy-init
                        LI::update(mg, p, downscale, i, i+1, j, npcol, buf,
ldpiv);
                        // 2) get the LU factors of the diagonal panel
                        receive_pivl(b, piv, ldpiv, rootcol, grid, req);
                        req.wait_all();

                        // 3) trsm U
                        update_right_panels(piv, ldpiv, p, i, j);
                        // 4) downcast from FHigh to FLow
                        convert_right_panels(p, scaleb, i, j, rp);
                        // 5) broadcast (send) U
                        broadcast_right_panels(b, rp, j, npcol, grid, req);
                        ++i;
                        // 6) broadast (receive) L
                        receive_left_panels(b, lp, i, nprow, rootcol, grid,
req);
                }
                else if (rootcol == grid.col) {
                        if (k == nb - 1) return;
                        // I have a left panel.
                        LI::update(mg, p, downscale, i, nprow, j, j+1, buf,
ldpiv); receive_pivu(b, piv, ldpiv, rootrow, grid, req); req.wait_all();

                        update_left_panels(piv, ldpiv, p, i, j);
                        convert_left_panels(p, scalea, i, j, lp);
                        broadcast_left_panels(b, lp, i, nprow, grid, req);
                        ++j;
                        receive_right_panels(b, rp, j, npcol, rootrow, grid,
req);
                }
                else {
                        if (k == nb - 1) return;
                        // broadcast (receive) L and U panels
                        receive_left_panels(b, lp, i, nprow, rootcol, grid,
req); receive_right_panels(b, rp, j, npcol, rootrow, grid, req);
                }
                req.wait_all();
                // GEMM
                SU::update(p, lp, rp, i, p.nprow, j, p.npcol);
        }
}
****/

template <typename FHigh, typename FLow, template <class> class Matgen, int du,
          bool dd>
void panel_lu_async_wGPU(Panels<FHigh> &p, LRPanels<FLow> lrpanels[4],
                         Matgen<FHigh> &mg, FHigh *piv, FHigh *d_piv,
                         size_t ldpiv, Grid &grid, bool warmup = false) {

    // same as panel_lu but with look-ahead computation for better communication
    // hiding
    // typedef LazyInitializer<FLow, du, dd> LI;
    // typedef SchurUpdator<FHigh, FLow, dd> SU;
    int test;
    int const nb = p.nblocks;
    int const b = p.b;
    int const n = nb * b;
    size_t const lda = p.lda;
    int const nprow = p.nprow;
    int const npcol = p.npcol;
    FHigh scalea = du ? static_cast<FHigh>(mg.scalea) : static_cast<FHigh>(1);
    FHigh scaleb = du ? static_cast<FHigh>(mg.scaleb) : static_cast<FHigh>(1);
    FHigh downscale = static_cast<FHigh>(1. / ((double)scalea * scaleb));
    FHigh *buf = piv + ldpiv * b;

    const float alf = 1.0f;
    const float nalf = -1.0f;
    const float bet = 1.0f;
    const float *alpha = &alf;
    const float *beta = &bet;

    LRPanels<FLow> *lprev = &lrpanels[0], *rprev = &lrpanels[1],
                   *lnext = &lrpanels[2], *rnext = &lrpanels[3];

    RequestStack lrreq(8);
    RequestStack pivreq(8);

    int schur_row = 0;
    int schur_col = 0;

    int64_t diag_lu_comp = b * (b * (4ll * b - 3ll) + 5ll) / 6ll;
    int kend = warmup ? 20 : nb;
    if (kend > nb)
        kend = nb;

    //
    static int once = 1;
    int lwork;
    static FHigh *d_work;

    GPUSOLVER_STATUS_T solvStat;

    if (once && p.solv == 1) {
        once = 0;

#ifdef CUDA_OLCF_PLATFORM
//      solvStat = GPUSOLVER_SGETRF_BUFFERSIZE(p.solvHandle, b, b, p(0, 0, 'd'),
        GPUSOLVER_SGETRF_BUFFERSIZE(p.solvHandle, b, b, p(0, 0, 'd'),
                                               p.d_lda, &lwork);
//      assert(solvStat == GPUSOLVER_STATUS_SUCCESS);
        // overallocation by 2x
        checkGPU( GPU_MALLOC( (void **)&d_work, 2 * lwork * sizeof( FHigh ) ),
	         "<FATAL%s> %s\n", hostRank, "Allocating device[gpu] cusolver work space" );

#endif
    }

    {
        // look ahead
        // same as above, but leaves gemm to next iteration
        if (0 == grid.row && 0 == grid.col) {
            // I have the pivot panel.
            FHigh *lu = p(0, 0);
            FHigh *d_lu = p(0, 0, 'd');
            Timer::beg(Timer::DIAG_LU);
            if (p.is_tile) {
                // getrf_nopiv(b, p(0, 0), lda);
            } else {
                // getrf_nopiv(b, p(0, 0), lda, piv, ldpiv);
		//
                if (p.solv == 1) {
//                  solvStat = GPUSOLVER_SGETRF(p.solvHandle, b, b, p(0, 0, 'd'),
                    GPUSOLVER_SGETRF(p.solvHandle, b, b, p(0, 0, 'd'),
                                         p.d_lda, d_work, NULL, p.d_infoArray );
//                  assert(solvStat == GPUSOLVER_STATUS_SUCCESS);

                } else {

#ifdef CUDA_OLCF_PLATFORM
                    float *diag_address = p(0, 0, 'd');
                    checkGPUblas(GPUBLAS_SGETRF_BATCHED(p.cuHandle, b,
                                                    &diag_address, p.d_lda,
                                                    NULL, p.d_infoArray, 1));
#else // ROCM_OLCF_PLATFORM
                    throw std::runtime_error("ROCBLAS does not support sgetrf, please use ROCSOLVE (-solv 1)");
#endif

                }

                GPU_DEVICE_SYNCHRONIZE();

            }
            // Copy piv back to cpu
            GPU_MEMCPY_2D(piv, ldpiv * sizeof(FHigh), p(0, 0, 'd'),
                    p.d_lda * sizeof(FHigh), b * sizeof(FHigh), b, GPU_MEMCPY_DEVICE_TO_HOST);

            Timer::end(Timer::DIAG_LU, false, diag_lu_comp);
            // GPU_DEVICE_SYNCHRONIZE();

            broadcast_pivlu_wGPU(p.is_tile, b, lu, lda, piv, ldpiv, grid,
                                 pivreq);

            update_left_panels_wGPU(p(0, 0, 'd'), p.d_lda, p, 1, 0);
            convert_left_panels_wGPU(p, scalea, 1, 0, *lprev);
            GPU_MEMCPY((*lprev)(1), (*lprev)(1, 'd'),
                       p.b * (*lprev).get_lda() * sizeof(__half),
                       GPU_MEMCPY_DEVICE_TO_HOST);
            broadcast_left_panels(b, *lprev, 1, nprow, grid, lrreq);

            update_right_panels_wGPU(d_lu, lda, p, 0, 1);
            convert_right_panels_wGPU(p, scaleb, 0, 1, *rprev);
            GPU_MEMCPY((*rprev)(1), (*rprev)(1, 'd'),
                       p.b * (*rprev).get_lda() * sizeof(__half),
                       GPU_MEMCPY_DEVICE_TO_HOST);
            broadcast_right_panels(b, *rprev, 1, npcol, grid, lrreq);
            schur_row = 1;
            schur_col = 1;
        } else if (0 == grid.row) {
            // I have the right panel.
            receive_pivl(b, piv, ldpiv, 0, grid, pivreq);
            pivreq.wait_all(Timer::DIAG_BCAST);
            GPU_MEMCPY(d_piv, piv, ldpiv * ldpiv * sizeof(FHigh),
                       GPU_MEMCPY_HOST_TO_DEVICE);

            receive_left_panels(b, *lprev, 1, nprow, 0, grid, lrreq);
            update_right_panels_wGPU(d_piv, ldpiv, p, 0, 0);
            convert_right_panels_wGPU(p, scaleb, 0, 0, *rprev);
            GPU_MEMCPY((*rprev)(0), (*rprev)(0, 'd'),
                       p.b * (*rprev).get_lda() * sizeof(__half),
                       GPU_MEMCPY_DEVICE_TO_HOST);
            broadcast_right_panels(b, *rprev, 0, npcol, grid, lrreq);
            schur_row = 1;
        } else if (0 == grid.col) {
            // I have the left panel.
            receive_pivu(b, piv, ldpiv, 0, grid, pivreq);
            pivreq.wait_all(Timer::DIAG_BCAST);
            GPU_MEMCPY(d_piv, piv, ldpiv * ldpiv * sizeof(FHigh),
                       GPU_MEMCPY_HOST_TO_DEVICE);

            receive_right_panels(b, *rprev, 1, npcol, 0, grid, lrreq);
            update_left_panels_wGPU(d_piv, ldpiv, p, 0, 0);
            convert_left_panels_wGPU(p, scalea, 0, 0, *lprev);
            GPU_MEMCPY((*lprev)(0), (*lprev)(0, 'd'),
                       p.b * (*lprev).get_lda() * sizeof(__half),
                       GPU_MEMCPY_DEVICE_TO_HOST);
            broadcast_left_panels(b, *lprev, 0, nprow, grid, lrreq);
            schur_col = 1;
        } else {
            receive_left_panels(b, *lprev, 0, nprow, 0, grid, lrreq);
            receive_right_panels(b, *rprev, 0, npcol, 0, grid, lrreq);
        }
        if (0 == grid.row && 0 == grid.col)
            pivreq.wait_all(Timer::DIAG_BCAST);
        lrreq.wait_all(Timer::WAIT);

        if (0 != grid.col && 0 == grid.row) {
            GPU_MEMCPY( (*lprev)(1, 'd'), (*lprev)(1),
                    p.b * (*lprev).get_lda() * sizeof(__half), GPU_MEMCPY_HOST_TO_DEVICE );
        } else if (0 != grid.row && 0 == grid.col) {
            GPU_MEMCPY( (*rprev)(1, 'd'), (*rprev)(1),
                    p.b * (*rprev).get_lda() * sizeof(__half), GPU_MEMCPY_HOST_TO_DEVICE );
        } else if (0 != grid.row && 0 != grid.col) {
            GPU_MEMCPY( (*lprev)(0, 'd'), (*lprev)(0),
                    p.b * (*lprev).get_lda() * sizeof(__half), GPU_MEMCPY_HOST_TO_DEVICE );
            GPU_MEMCPY( (*rprev)(0, 'd'), (*rprev)(0),
                    p.b * (*rprev).get_lda() * sizeof(__half), GPU_MEMCPY_HOST_TO_DEVICE);
        }
    }

//  LOG->info("Inside panel_lu_async_wGPU 4-loop");

    // Main loop
    for (int k = 1; k < kend; ++k) {
        // printf("process=%d %d, k=%d, wtime=%f\n", grid.row, grid.col, k,
        // Timer::put(Timer::MISC));
        int const rootrow = k % grid.nrow;
        int const rootcol = k % grid.ncol;
        int i = k / grid.nrow + (rootrow > grid.row ? 1 : 0);
        int j = k / grid.ncol + (rootcol > grid.col ? 1 : 0);
        if (rootrow == grid.row && rootcol == grid.col) {
            // I have the pivot panel.
            // do GEMM for the diagonal block
            Timer::beg(Timer::GEMM_UPDATE);

            checkGPUblas( GPUBLAS_SGEMM_EX(
                p.cuHandle, GPUBLAS_OP_N, GPUBLAS_OP_T, b, b, b, &nalf,
                (void *)(*lprev)(i, 'd'), GPU_R_16F, (*lprev).get_lda(),
                (void *)(*rprev)(j, 'd'), GPU_R_16F, (*rprev).get_lda(), beta,
                p(i, j, 'd'), GPU_R_32F, p.lda) );
            GPU_DEVICE_SYNCHRONIZE();

            Timer::end(Timer::GEMM_UPDATE, false, 2ull * p.b * p.b * p.b);

            FHigh *lu = p(i, j);
            FHigh *d_lu = p(i, j, 'd');

            Timer::beg(Timer::DIAG_LU);
            if (p.is_tile) {
                // getrf_nopiv(b, p(i, j), lda);
            } else {
                // getrf_nopiv(b, p(i, j), lda, piv, ldpiv);
                if (p.solv == 1) {
//                  solvStat =
                        GPUSOLVER_SGETRF(p.solvHandle, b, b, p(i, j, 'd'),
                                         p.d_lda, d_work, NULL, p.d_infoArray );
//                  assert(solvStat == GPUSOLVER_STATUS_SUCCESS);
                } else {

#ifdef CUDA_OLCF_PLATFORM
                    float *diag_address = p(i, j, 'd');
                    checkGPUblas(GPUBLAS_SGETRF_BATCHED(p.cuHandle, b,
                                                    &diag_address, p.d_lda,
                                                    NULL, p.d_infoArray, 1));
#else // ROCM_OLCF_PLATFORM
                    throw std::runtime_error("ROCBLAS does not support sgetrf, please use ROCSOLVE (-solv 1)");
#endif
                }

                GPU_DEVICE_SYNCHRONIZE();

            }
            GPU_MEMCPY_2D(piv, ldpiv * sizeof(FHigh), p(i, j, 'd'),
                         p.d_lda * sizeof(FHigh), b * sizeof(FHigh), b,
                         GPU_MEMCPY_DEVICE_TO_HOST);
            Timer::end(Timer::DIAG_LU, false, diag_lu_comp);

            broadcast_pivlu_wGPU(p.is_tile, b, lu, lda, piv, ldpiv, grid,
                                 pivreq);

            // do GEMM of L
            Timer::beg(Timer::GEMM_UPDATE);
            if (p.nprow - i - 1 > 0) {

                checkGPUblas( GPUBLAS_SGEMM_EX(
                    p.cuHandle, GPUBLAS_OP_N, GPUBLAS_OP_T, b * (p.nprow - i - 1),
                    b, b, &nalf, (void *)(*lprev)(i + 1, 'd'), GPU_R_16F,
                    (*lprev).get_lda(), (void *)(*rprev)(j, 'd'), GPU_R_16F,
                    (*rprev).get_lda(), beta, p(i + 1, j, 'd'), GPU_R_32F,
                    p.lda ) );

            }

            GPU_DEVICE_SYNCHRONIZE();

            Timer::end(Timer::GEMM_UPDATE, false, 2ull * p.b * p.b * p.b * (p.nprow - i - 1));

            update_left_panels_wGPU(d_lu, lda, p, i + 1, j);
            convert_left_panels_wGPU(p, scalea, i + 1, j, *lnext);
            GPU_MEMCPY( (*lnext)(i + 1), (*lnext)(i + 1, 'd'),
                    p.b * (*lnext).get_lda() * sizeof(__half), GPU_MEMCPY_DEVICE_TO_HOST );

            broadcast_left_panels(b, *lnext, i + 1, nprow, grid, lrreq);

            // do GEMM of U
            Timer::beg(Timer::GEMM_UPDATE);
            if (p.npcol - j - 1 > 0) {

                checkGPUblas( GPUBLAS_SGEMM_EX(p.cuHandle, GPUBLAS_OP_N, GPUBLAS_OP_T,
                                          b, p.b * (p.npcol - j - 1), b, &nalf,
                                          (void *)(*lprev)(i, 'd'), GPU_R_16F,
                                          (*lprev).get_lda(),
                                          (void *)(*rprev)(j + 1, 'd'),
                                          GPU_R_16F, (*rprev).get_lda(), beta,
                                          p(i, j + 1, 'd'), GPU_R_32F, p.lda) );
            }

            GPU_DEVICE_SYNCHRONIZE();

            Timer::end(Timer::GEMM_UPDATE, false, 2ull * p.b * p.b * p.b * (npcol - j - 1));

            update_right_panels_wGPU(d_lu, lda, p, i, j + 1);
            convert_right_panels_wGPU(p, scaleb, i, j + 1, *rnext);
            GPU_MEMCPY((*rnext)(j + 1), (*rnext)(j + 1, 'd'),
                    p.b * (*rnext).get_lda() * sizeof(__half), GPU_MEMCPY_DEVICE_TO_HOST );

            broadcast_right_panels(b, *rnext, j + 1, npcol, grid, lrreq);

            ++schur_row;
            ++schur_col;
            ++i;
            ++j;
        } else if (rootrow == grid.row) {
            // receive LU factors of the diagonal block
            receive_pivl(b, piv, ldpiv, rootcol, grid, pivreq);
            // GEMM
            Timer::beg(Timer::GEMM_UPDATE);
            if (p.npcol - j > 0) {

                checkGPUblas( GPUBLAS_SGEMM_EX(
                    p.cuHandle, GPUBLAS_OP_N, GPUBLAS_OP_T, b,
                    p.b * (p.npcol - j), b, &nalf, (void *)(*lprev)(i, 'd'),
                    GPU_R_16F, (*lprev).get_lda(), (void *)(*rprev)(j, 'd'),
                    GPU_R_16F, (*rprev).get_lda(), beta, p(i, j, 'd'),
                    GPU_R_32F, p.lda) );
            }

            GPU_DEVICE_SYNCHRONIZE();

            Timer::end(Timer::GEMM_UPDATE, false, 2ull * p.b * p.b * p.b * (p.npcol - j));

            ++schur_row;
            while (!pivreq.test_all(Timer::DIAG_BCAST)) {
                pivreq.wait_all(Timer::DIAG_BCAST);
                break;
            }
            GPU_MEMCPY(d_piv, piv, ldpiv * ldpiv * sizeof(FHigh), GPU_MEMCPY_HOST_TO_DEVICE);
            receive_left_panels(b, *lnext, i + 1, nprow, rootcol, grid, lrreq);

            update_right_panels_wGPU(d_piv, ldpiv, p, i, j);
            convert_right_panels_wGPU(p, scaleb, i, j, *rnext);
            GPU_MEMCPY( (*rnext)(j), (*rnext)(j, 'd'),
                    p.b * (*rnext).get_lda() * sizeof(__half), GPU_MEMCPY_DEVICE_TO_HOST );
            broadcast_right_panels(b, *rnext, j, npcol, grid, lrreq);
            ++i;
        } else if (rootcol == grid.col) {
            receive_pivu(b, piv, ldpiv, rootrow, grid, pivreq);
            Timer::beg(Timer::GEMM_UPDATE);
            if (p.nprow - i > 0) {

                checkGPUblas( GPUBLAS_SGEMM_EX(
                    p.cuHandle, GPUBLAS_OP_N, GPUBLAS_OP_T, b * (p.nprow - i), b,
                    b, &nalf, (void *)(*lprev)(i, 'd'), GPU_R_16F,
                    (*lprev).get_lda(), (void *)(*rprev)(j, 'd'), GPU_R_16F,
                    (*rprev).get_lda(), beta, p(i, j, 'd'), GPU_R_32F, p.lda) );

            }
            GPU_DEVICE_SYNCHRONIZE();

            Timer::end(Timer::GEMM_UPDATE, false, 2ull * p.b * p.b * p.b * (p.nprow - i));

            ++schur_col;
            while (!pivreq.test_all(Timer::DIAG_BCAST)) {
                pivreq.wait_all(Timer::DIAG_BCAST);
                break;
            }
            GPU_MEMCPY(d_piv, piv, ldpiv * ldpiv * sizeof(FHigh), GPU_MEMCPY_HOST_TO_DEVICE);
            receive_right_panels(b, *rnext, j + 1, npcol, rootrow, grid, lrreq);
            update_left_panels_wGPU(d_piv, ldpiv, p, i, j);
            convert_left_panels_wGPU(p, scalea, i, j, *lnext);
            GPU_MEMCPY((*lnext)(i), (*lnext)(i, 'd'),
                    p.b * (*lnext).get_lda() * sizeof(__half), GPU_MEMCPY_DEVICE_TO_HOST);
            broadcast_left_panels(b, *lnext, i, nprow, grid, lrreq);
            ++j;
        } else {
            receive_left_panels(b, *lnext, i, nprow, rootcol, grid, lrreq);
            receive_right_panels(b, *rnext, j, npcol, rootrow, grid, lrreq);
        }

        Timer::beg(Timer::GEMM_UPDATE);
        if (p.nprow - schur_row > 0 && p.npcol - schur_col > 0) {

            checkGPUblas( GPUBLAS_SGEMM_EX(
                p.cuHandle, GPUBLAS_OP_N, GPUBLAS_OP_T,
                p.b * (p.nprow - schur_row), p.b * (p.npcol - schur_col), b,
                &nalf, (void *)(*lprev)(schur_row, 'd'), GPU_R_16F,
                (*lprev).get_lda(), (void *)(*rprev)(schur_col, 'd'),
                GPU_R_16F, (*rprev).get_lda(), beta,
                p(schur_row, schur_col, 'd'), GPU_R_32F, p.lda) );
        }

        GPU_DEVICE_SYNCHRONIZE();

        Timer::end(Timer::GEMM_UPDATE, false,
                   2ull * p.b * p.b * p.b * (p.nprow - schur_row) * (p.npcol - schur_col));

        LRPanels<FLow> *t;
        t = lprev;
        lprev = lnext;
        lnext = t;
        t = rprev;
        rprev = rnext;
        rnext = t;
        schur_row = i;
        schur_col = j;

        if (rootrow == grid.row && rootcol == grid.col)
            pivreq.wait_all(Timer::DIAG_BCAST);
        lrreq.wait_all(Timer::WAIT);

        if (rootcol != grid.col)
            GPU_MEMCPY( (*lprev)(i, 'd'), (*lprev)(i),
                    p.b * (*lprev).get_lda() * sizeof(__half), GPU_MEMCPY_HOST_TO_DEVICE );
        if (rootrow != grid.row)
            GPU_MEMCPY( (*rprev)(j, 'd'), (*rprev)(j),
                    p.b * (*rprev).get_lda() * sizeof(__half), GPU_MEMCPY_HOST_TO_DEVICE );
    }
    if (warmup) {
        pivreq.wait_all();
        lrreq.wait_all();
    }
}

#endif
