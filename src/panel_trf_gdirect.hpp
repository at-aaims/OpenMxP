#ifndef PANEL_TRF_G_HPP
#define PANEL_TRF_G_HPP
#include "panel_trf_block.hpp"
#include "device_macros.h"

// #include <rocm_smi/rocm_smi.h>

#define SLEEP      0
#define NUM_THREAD 3

extern int   gcdOrder[ ];
extern float power0;

template <typename F>
//inline
void broadcast_left_panels_gdirect(int b, const LRPanels<F> &lp, int rowstart, int nrow, Grid &grid) {
//  Timer::beg(Timer::LCOL_BCAST);
    MPI_Bcast(const_cast<F *>(lp(rowstart,'d')), lp.get_lda() * b, T2MPI<F>::type, grid.col, grid.hcomm);
//  Timer::end(Timer::LCOL_BCAST);
}
template <typename F>
//inline
void receive_left_panels_gdirect(int b, LRPanels<F> &lp, int rowstart, int nrow, int root, Grid &grid) {
//  Timer::beg(Timer::LCOL_BCAST);
    lp.set_start(rowstart);
    MPI_Bcast(const_cast<F *>(lp(rowstart,'d')), lp.get_lda() * b, T2MPI<F>::type, root, grid.hcomm);
//  Timer::end(Timer::LCOL_BCAST);
}
template <typename F> 
//inline
void broadcast_right_panels_gdirect(int b, LRPanels<F> &rp, int colstart, int ncol, Grid &grid) {
//  Timer::beg(Timer::RROW_BCAST);
    MPI_Bcast(const_cast<F *>(rp(colstart,'d')), rp.get_lda() * b, T2MPI<F>::type, grid.row, grid.vcomm);
//  Timer::end(Timer::RROW_BCAST);
}
template <typename F>
//inline
void receive_right_panels_gdirect(int b, LRPanels<F> &rp, int colstart, int ncol, int root, Grid &grid) {
//  Timer::beg(Timer::RROW_BCAST);
    rp.set_start(colstart);
    //	T2MPI<F>::type, root, grid.vcomm, req.get_request());
    MPI_Bcast(const_cast<F *>(rp(colstart,'d')), rp.get_lda() * b, T2MPI<F>::type, root, grid.vcomm);
//  Timer::end(Timer::RROW_BCAST);
}

template <typename FHigh, typename FLow>
void update_left_panels_wGPU_alt_simple(const FHigh *u, int ldu, Panels<FHigh> &p, int rowstart,
                             int col,  LRPanels<FLow> &lp, Workspace<FHigh,FLow> &wks) {
	FHigh scale;
        lp.set_start(rowstart);

	const FHigh zlf = 0;
	const FHigh olf = 1;

	if (rowstart == p.nprow)
		return;

	// Cast it down to fp16
    downCast_copy_general(lp(rowstart,'d'),(long) lp.get_lda(),
							(long)(p.nprow - rowstart) * p.b,(long)p.b,
							p(rowstart,col,'d'), (long)p.d_lda);
    //fp16 inverse gemm
	checkGPUblas( GPUBLAS_SGEMM_EX(p.cuHandle, GPUBLAS_OP_N, GPUBLAS_OP_N,
                                          (p.nprow-rowstart)*p.b, p.b, p.b, &olf,
                                          (void *)lp(rowstart,'d'), GPU_R_16F, lp.get_lda(),
                                          (void *)wks.Imatrix_low,GPU_R_16F, wks.Imatrix_ld,
                                          &zlf, p(rowstart, col, 'd'), GPU_R_32F, p.d_lda) );

	GPU_DEVICE_SYNCHRONIZE();
        Timer::end(Timer::TRSM_L, false, 1ll * p.b * (p.nprow - rowstart) * p.b * p.b);
}

template <typename FHigh, typename FLow>
void update_left_panels_wGPU_alt(const FHigh *u, int ldu, Panels<FHigh> &p, int rowstart,
                             int col,  LRPanels<FLow> &lp, Workspace<FHigh,FLow> &wks) {
	FHigh scale;
        lp.set_start(rowstart);

	const FHigh zlf = 0;
	const FHigh olf = 1;

	if (rowstart == p.nprow)
		return;


	//generate a Identity matirx
	gen_identity_mat(wks.Imatrix_high, (long)p.b,(long)p.b);
	// Find inverse from U;
    checkGPUblas(GPUBLAS_STRSM(
            p.cuHandle, GPUBLAS_SIDE_RIGHT, GPUBLAS_FILL_MODE_UPPER, GPUBLAS_OP_N,
            GPUBLAS_DIAG_NON_UNIT, p.b, p.b, &olf, u,
            ldu, wks.Imatrix_high, wks.Imatrix_ld));
    downCast_copy_general(lp(rowstart,'d'),(long) lp.get_lda(),
							(long)(p.nprow - rowstart) * p.b,(long)p.b,
							p(rowstart,col,'d'), (long)p.d_lda);
	downCast_copy_general(wks.Imatrix_low, (long)wks.Imatrix_ld,
							(long)p.b, (long)p.b,
							wks.Imatrix_high, (long)wks.Imatrix_ld);
    //fp16 inverse gemm
	checkGPUblas( GPUBLAS_SGEMM_EX(p.cuHandle, GPUBLAS_OP_N, GPUBLAS_OP_N,
                                          (p.nprow-rowstart)*p.b, p.b, p.b, &olf,
                                          (void *)lp(rowstart,'d'), GPU_R_16F, lp.get_lda(),
                                          (void *)wks.Imatrix_low,GPU_R_16F, wks.Imatrix_ld,
                                          &zlf, p(rowstart, col, 'd'), GPU_R_32F, p.d_lda) );

	GPU_DEVICE_SYNCHRONIZE();

}

template <typename FHigh, typename FLow>
void update_right_panels_wGPU_alt(const FHigh *l, int ldl, Panels<FHigh> &p, int row,
                              int colstart, LRPanels<FLow> &rp, Workspace<FHigh,FLow> &wks) {
        FHigh scale;
	rp.set_start(colstart);
	const FHigh zlf = 0;
	const FHigh olf = 1;

	if (colstart == p.npcol) return;

	//generate a Identity matirx
	gen_identity_mat(wks.Imatrix_high, (long)p.b,(long)p.b);
	checkGPUblas(GPUBLAS_STRSM(
            p.cuHandle, GPUBLAS_SIDE_RIGHT, GPUBLAS_FILL_MODE_LOWER, GPUBLAS_OP_N,
            GPUBLAS_DIAG_UNIT, p.b, p.b, &olf, l, ldl,
            wks.Imatrix_high, wks.Imatrix_ld));

	downCast_trans_general(rp(colstart,'d'), (long)rp.get_lda(),
							(long)p.b, (long)(p.npcol - colstart) * p.b,
							p(row,colstart,'d'), p.d_lda);
	
    downCast_copy_general(wks.Imatrix_low, (long)wks.Imatrix_ld,
							(long)p.b, (long)p.b,
							wks.Imatrix_high, (long)wks.Imatrix_ld);

    //fp16 inverse gemm
	checkGPUblas( GPUBLAS_SGEMM_EX(p.cuHandle, GPUBLAS_OP_N, GPUBLAS_OP_T,
                                          p.b, (p.npcol-colstart)*p.b, p.b, &olf,
                                          (void *)wks.Imatrix_low,GPU_R_16F, wks.Imatrix_ld,
                                          (void *)rp(colstart,'d'), GPU_R_16F, rp.get_lda(),//p.b,
                                          &zlf, p(row, colstart, 'd'), GPU_R_32F, p.d_lda) );
	GPU_DEVICE_SYNCHRONIZE();
}

template <typename FHigh, typename FLow>
void update_left_panels_wGPU_alt2(FHigh *u, int ldu, Panels<FHigh> &p, int rowstart,
                             int col,  LRPanels<FLow> &lp, Workspace<FHigh,FLow> &wks) {
	FHigh scale;
        lp.set_start(rowstart);
	const FHigh zlf = 0;
	const FHigh olf = 1;
    GPUSOLVER_STATUS_T cus_status = GPUSOLVER_STATUS_SUCCESS;
	if (rowstart == p.nprow)
		return;  
	downCast_copy_upper(wks.Imatrix_low, wks.Imatrix_ld,
							p.b, p.b,
							u, p.b);
    //fp16 inverse gemm
	checkGPUblas( GPUBLAS_SGEMM_EX(p.cuHandle, GPUBLAS_OP_N, GPUBLAS_OP_N,
                                          (p.nprow-rowstart)*p.b, p.b, p.b, &olf,
                                          (void *)lp(rowstart,'d'), GPU_R_16F, lp.get_lda(),
                                          (void *)wks.Imatrix_low,GPU_R_16F, wks.Imatrix_ld,
                                          &zlf, p(rowstart, col, 'd'), GPU_R_32F, p.d_lda) );
}

template <typename FHigh, typename FLow>
void update_right_panels_wGPU_alt2(FHigh *l, int ldl, Panels<FHigh> &p, int row,
                              int colstart, LRPanels<FLow> &rp, Workspace<FHigh,FLow>& wks) {
    FHigh scale;
	rp.set_start(colstart);
	const FHigh zlf = 0;
	const FHigh olf = 1;
    GPUSOLVER_STATUS_T cus_status = GPUSOLVER_STATUS_SUCCESS;
	if (colstart == p.npcol) return;
    downCast_copy_lower(wks.Imatrix_low, wks.Imatrix_ld,
							p.b, p.b,
							l, wks.Imatrix_ld);
    
	checkGPUblas( GPUBLAS_SGEMM_EX(p.cuHandle, GPUBLAS_OP_N, GPUBLAS_OP_T,
                                          p.b, (p.npcol-colstart)*p.b, p.b, &olf,
                                          (void *)wks.Imatrix_low,GPU_R_16F, wks.Imatrix_ld,
                                          (void *)rp(colstart,'d'), GPU_R_16F, rp.get_lda(),//p.b,
                                          &zlf, p(row, colstart, 'd'), GPU_R_32F, p.d_lda) );
}


template <typename FHigh, typename FLow, template <class> class Matgen, int du, bool dd>
void panel_lu_async_wGPU_gdirect_alt1(Panels<FHigh> &p,
                                    LRPanels<FLow> lrpanels[4],
                                    Matgen<FHigh> &mg,
                                    FHigh *piv,
                                    FHigh *d_piv,
                                    size_t ldpiv,
                                    Grid &grid,
                                    bool warmup, Workspace<FHigh,FLow> &wks,Workspace<FHigh,FLow> &wks2) {
    // same as panel_lu but with look-ahead computation for better communication
    // hiding
    // typedef LazyInitializer<FLow, du, dd> LI;
    // typedef SchurUpdator<FHigh, FLow, dd> SU;
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

    // Alternative trsm
    int alternative = p.alt;
    /*Workspace<FHigh,FLow> wks;
    if ( alternative )
    {
       build_workspace( wks, b );
    }*/

    // Diagonal cusolver variables
    int lwork;

    GPUSOLVER_STATUS_T cus_status = GPUSOLVER_STATUS_SUCCESS;
    GPU_ERROR_T cudaStat1 = GPU_SUCCESS;


    float *d_work = NULL;

#ifdef CUDA_OLCF_PLATFORM
    GPUSOLVER_SGETRF_BUFFERSIZE(p.solvHandle, b, b, p(0, 0, 'd'),
                                             p.d_lda, &lwork);
    cudaStat1 = GPU_MALLOC((void **)&d_work, sizeof( FHigh ) * lwork);
    assert(GPU_SUCCESS == cudaStat1);
#endif

    int divider = ceil(nb*0.01);
    const FHigh alf = 1.0f;
    const FHigh nalf = -1.0f;
    const FHigh bet = 1.0f;
    const FHigh *alpha = &alf;
    const FHigh *beta = &bet;
	const FHigh zlf = 0.0;
	const FHigh olf = 1.0;

    LRPanels<FLow> *lprev = &lrpanels[0], *rprev = &lrpanels[1], *lnext = &lrpanels[2], *rnext = &lrpanels[3];

    int schur_row = 0;
    int schur_col = 0;

    int communication_type = 0;
    int64_t diag_lu_comp = b * (b * (4ll * b - 3ll) + 5ll) / 6ll;
    int kend = warmup ? 20 : nb;
    if(kend > nb)
        kend = nb;

    int haveGEMM, haveGETRF;
    hipEvent_t eBgn , eEnd ; // gemm
    hipEvent_t eBgn2, eEnd2; // getrf
    hipEvent_t eBgn3, eEnd3; // trsm
    hipEvent_t eBgn4, eEnd4; // convert
    hipEvent_t eBgn5, eEnd5; // convert

    float mSec    = 0.0; // gemm
    float t_mSec  = 0.0; // gemm total
    float mSec2   = 0.0; // getrf
    float t_mSec2 = 0.0; // getrf total
    float mSec3   = 0.0; // trsm 
    float t_mSec3 = 0.0; // trsm total
    float mSec4   = 0.0; // convert
    float t_mSec4 = 0.0; // convert total
    float mSec5   = 0.0; // memcpy
    float t_mSec5 = 0.0; // memcpy total

    PrintLogMsg( "Entering LU\n" );

    int          n_gemm = 0;
    static int   mnk   [ 10 ][ 3 ];  // m - n - k dimensions 
    static float tflops[ 10 ];

    double lu_time;

    {
        bgn_time2 = get_utime( );
        lu_time   = bgn_time2;

        // look ahead
        // same as above, but leaves gemm to next iteration
        if(0 == grid.row && 0 == grid.col) {
            // I have the pivot panel.
            FHigh *lu = p(0, 0);
            FHigh *d_lu = p(0, 0, 'd');

            // Solve Diagonal
            //Timer::beg(Timer::DIAG_LU);
    
            if ( !CHECK_BIT( p.skip, 1 ) )
            {
                cus_status = GPUSOLVER_SGETRF( p.solvHandle, b, b,
                            p(0, 0, 'd'), p.d_lda,
                            d_work, NULL, p.d_infoArray );
                assert(GPUSOLVER_STATUS_SUCCESS == cus_status);
            }
    
            GPU_MEMCPY_2D(d_piv,
                        ldpiv * sizeof(FHigh),
                        p(0, 0, 'd'),
                        p.d_lda * sizeof(FHigh),
                        b * sizeof(FHigh),
                        b,
                        GPU_MEMCPY_DEVICE_TO_DEVICE);
            broadcast_pivlu_wGPU_block(p.is_tile, b, lu, lda, d_piv, ldpiv, grid);

            // Update left panels
            update_left_panels_wGPU(d_lu, p.d_lda, p, 1, 0);
            convert_left_panels_wGPU(p, scalea, 1, 0, *lprev);
            broadcast_left_panels_gdirect(b, *lprev, 1, nprow, grid);

            // Update right panels
            update_right_panels_wGPU(d_lu, lda, p, 0, 1);
            convert_right_panels_wGPU(p, scaleb, 0, 1, *rprev);
            broadcast_right_panels_gdirect(b, *rprev, 1, npcol, grid);

            schur_row = 1;
            schur_col = 1;
        
        } else if(0 == grid.row) {
            receive_pivl_block(b, d_piv, ldpiv, 0, grid);
            update_right_panels_wGPU(d_piv, ldpiv, p, 0, 0);
            convert_right_panels_wGPU(p, scaleb, 0, 0, *rprev);
            broadcast_right_panels_gdirect(b, *rprev, 0, npcol, grid);
            receive_left_panels_gdirect(b, *lprev, 1, nprow, 0, grid);
            schur_row = 1;
        } else if(0 == grid.col) {
            receive_pivu_block(b, d_piv, ldpiv, 0, grid);
            update_left_panels_wGPU(d_piv, ldpiv, p, 0, 0);
            convert_left_panels_wGPU(p, scalea, 0, 0, *lprev);
            broadcast_left_panels_gdirect(b, *lprev, 0, nprow, grid);
            receive_right_panels_gdirect(b, *rprev, 1, npcol, 0, grid);
            schur_col = 1;
        } else {
            receive_left_panels_gdirect(b, *lprev, 0, nprow, 0, grid);
            receive_right_panels_gdirect(b, *rprev, 0, npcol, 0, grid);
        }
    }
	
	//GPU_DEVICE_SYNCHRONIZE();
	// gather times from all nodes ???
	// Define ring comm flags
    int rrpiv_flag, rcpiv_flag, rrp_flag, rlp_flag;
	MPI_Request lrequest, rrequest;

	// Start of Steps
	uint32_t dv_ind = gcdOrder[ grank % 8 ];
	uint32_t sensor_ind = 0;
    uint64_t power;
    int64_t  temp;
    float    counter_resolution;
    uint64_t timestamp;
	double   time_used = 0.0;
   	
    // Main Loop
    for(int k = 1; k < kend; ++k)
    {
        double bgn_time_k = get_utime( );
        int const rootrow = k % grid.nrow;
        int const rootcol = k % grid.ncol;
        int i = k / grid.nrow + (rootrow > grid.row ? 1 : 0);
        int j = k / grid.ncol + (rootcol > grid.col ? 1 : 0);
        if(rootrow == grid.row && rootcol == grid.col) {
            // I have the pivot panel.
            // do GEMM for the diagonal block
            GPUBLAS_SGEMM_EX( p.cuHandle, GPUBLAS_OP_N, GPUBLAS_OP_T,
                            b, b, b, &nalf,
                            ( void * )(*lprev)(i, 'd'), GPU_R_16F, (*lprev).get_lda(),
                            ( void * )(*rprev)(j, 'd'), GPU_R_16F, (*rprev).get_lda(),
                            beta,       p(i, j, 'd'), GPU_R_32F, p.lda );
            cus_status = GPUSOLVER_SGETRF(p.solvHandle, b, b, p(i, j, 'd'),
                                                p.d_lda, d_work, NULL, p.d_infoArray );
            assert(GPUSOLVER_STATUS_SUCCESS == cus_status);
            GPU_MEMCPY_2D(d_piv,
                         ldpiv * sizeof(FHigh),
                         p(i, j, 'd'),
                         p.d_lda * sizeof(FHigh),
                         b * sizeof(FHigh),
                         b,
                         GPU_MEMCPY_DEVICE_TO_DEVICE);
          
            #pragma omp parallel for num_threads( NUM_THREAD )
	        for ( int kk = 0; kk < NUM_THREAD; kk++ )
	        {
	            if ( kk == 1 ){
                        gen_identity_mat(wks.Imatrix_high, (long)p.b,(long)p.b);
                        checkGPUblas(GPUBLAS_STRSM(
                        p.cuHandle, GPUBLAS_SIDE_RIGHT, GPUBLAS_FILL_MODE_UPPER, GPUBLAS_OP_N,
                        GPUBLAS_DIAG_NON_UNIT, p.b, p.b, &olf, d_piv,
                        b, wks.Imatrix_high, wks.Imatrix_ld));
                        downCast_copy_general(wks.Imatrix_low, (long)wks.Imatrix_ld,
							(long)p.b, (long)p.b,
							d_piv, (long)wks.Imatrix_ld);
                        GPU_DEVICE_SYNCHRONIZE();
                }
	            if ( kk == 2 ){
	                while ( myBcast(d_piv, ldpiv * b, grid.col, grid.hcomm, rootcol, k, grid.ncol, p.dcomm )
		                == myB_KEEPTEST ){}
                }
	        }
           
           
            #pragma omp parallel for num_threads( NUM_THREAD )
	        for ( int kk = 0; kk < NUM_THREAD; kk++ )
	        {
	            if ( kk == 1 ){
                    while ( myBcast(wks.Imatrix_low, wks.Imatrix_ld * b, grid.row, grid.vcomm, rootrow, k, grid.nrow, p.dcomm )
	                        == myB_KEEPTEST ){}
                }
	            if ( kk == 2 ){
	                  // do GEMM of L
                    if(p.nprow - i - 1 > 0) {
                        GPUBLAS_SGEMM_EX( p.cuHandle, GPUBLAS_OP_N, GPUBLAS_OP_T,
                                        b * (p.nprow - i - 1), b, b, &nalf,
                                        ( void * )(*lprev)(i + 1, 'd'), GPU_R_16F, (*lprev).get_lda(),
                                        ( void * )(*rprev)(j    , 'd'), GPU_R_16F, (*rprev).get_lda(),
                                        beta,       p(i + 1, j, 'd'), GPU_R_32F, p.lda );
                    }
                    // do GEMM of U
                    if(p.npcol - j - 1 > 0) {
                        GPUBLAS_SGEMM_EX( p.cuHandle, GPUBLAS_OP_N, GPUBLAS_OP_T,
                                                b, p.b * (p.npcol - j - 1), b, &nalf,
                                            ( void * )(*lprev)(i    , 'd'), GPU_R_16F, (*lprev).get_lda(),
                                            ( void * )(*rprev)(j + 1, 'd'), GPU_R_16F, (*rprev).get_lda(),
                                                beta, p(i, j + 1, 'd'), GPU_R_32F, p.lda );
                    }
                
                    update_left_panels_wGPU_alt_simple(d_piv, ldpiv, p, i + 1, j, *lnext, wks );
                    convert_left_panels_wGPU(p, scalea, i + 1, j, *lnext);
                }
            }
        

            update_right_panels_wGPU_alt( d_piv, ldpiv, p, i, j + 1, *rnext, wks2 );
            convert_right_panels_wGPU(p, scaleb, i, j + 1, *rnext);
	        ++schur_row;
            ++schur_col;

            GPUBLAS_SGEMM_EX( p.cuHandle, GPUBLAS_OP_N, GPUBLAS_OP_T,
                                        p.b * (p.nprow - schur_row), p.b * (p.npcol - schur_col), b, &nalf,
                                    ( void * )(*lprev)(schur_row, 'd'), GPU_R_16F, (*lprev).get_lda(),
                                    ( void * )(*rprev)(schur_col, 'd'), GPU_R_16F, (*rprev).get_lda(),
                                    beta, p(schur_row, schur_col, 'd'), GPU_R_32F, p.lda );

            #pragma omp parallel for num_threads( NUM_THREAD )
	        for ( int kk = 0; kk < NUM_THREAD; kk++ )
	        {
	            if ( kk == 1 )
                    while ( myBcast( (*lnext)(i + 1, 'd'), lnext->get_lda() * b, grid.col, grid.hcomm, rootcol, k, grid.ncol, p.comm)
  	                    == myB_KEEPTEST ){}
	            if ( kk == 2 )
	                while ( myBcast( (*rnext)(j + 1, 'd'), rnext->get_lda() * b, grid.row, grid.vcomm, rootrow, k, grid.nrow, p.comm)
		                == myB_KEEPTEST ){}
            }

            ++i;
            ++j;
        } 
        else if(rootrow == grid.row)
        {
            if(p.npcol - j > 0) {
		        GPUBLAS_SGEMM_EX( p.cuHandle, GPUBLAS_OP_N, GPUBLAS_OP_T,
                                     b, p.b * (p.npcol - j), b, &nalf,
                                   ( void * )(*lprev)(i, 'd'), GPU_R_16F, (*lprev).get_lda(),
                                   ( void * )(*rprev)(j, 'd'), GPU_R_16F, (*rprev).get_lda(),
                                     beta,       p(i, j, 'd'), GPU_R_32F, p.lda );
            }
            ++schur_row;
            while ( myBcast(d_piv, ldpiv * b, grid.col, grid.hcomm, rootcol, k, grid.ncol, p.dcomm)
	            == myB_KEEPTEST ){}

            // Update U
            update_right_panels_wGPU_alt( d_piv, ldpiv, p, i, j, *rnext, wks2 );
	        convert_right_panels_wGPU(p, scaleb, i, j, *rnext);
	        // Lauching Trailing update
	        if(p.nprow - schur_row > 0 && p.npcol - schur_col > 0) {
		        GPUBLAS_SGEMM_EX( p.cuHandle, GPUBLAS_OP_N, GPUBLAS_OP_T,
                                     p.b * (p.nprow - schur_row), p.b * (p.npcol - schur_col), b, &nalf,
                                   ( void * )(*lprev)(schur_row, 'd'), GPU_R_16F, (*lprev).get_lda(),
                                   ( void * )(*rprev)(schur_col, 'd'), GPU_R_16F, (*rprev).get_lda(),
                                   beta, p(schur_row, schur_col, 'd'), GPU_R_32F, p.lda);
            }
            lnext->set_start(i + 1);
            #pragma omp parallel for num_threads( NUM_THREAD )
	        for ( int kk = 0; kk < NUM_THREAD; kk++ )
	        {
	            if ( kk == 1 )
                    while ( myBcast( (*lnext)(i + 1,'d'), lnext->get_lda() * b, grid.col, grid.hcomm, rootcol, k, grid.ncol, p.comm)
                        == myB_KEEPTEST ){}
                if ( kk == 2 )
                    while ( myBcast( (*rnext)(j,'d'), rnext->get_lda() * b, grid.row, grid.vcomm, rootrow, k, grid.nrow, p.comm)
                        == myB_KEEPTEST ){}
            }
            ++i;

        }
        else if(rootcol == grid.col) {
            if(p.nprow - i > 0) {
		        GPUBLAS_SGEMM_EX( p.cuHandle, GPUBLAS_OP_N, GPUBLAS_OP_T,
                                     b * (p.nprow - i), b, b, &nalf,
                                   ( void * )(*lprev)(i, 'd'), GPU_R_16F, (*lprev).get_lda(),
                                   ( void * )(*rprev)(j, 'd'), GPU_R_16F, (*rprev).get_lda(),
                                     beta,       p(i, j, 'd'), GPU_R_32F, p.lda );
  	        
            }
            ++schur_col;
            while ( myBcast(wks.Imatrix_low, wks.Imatrix_ld * b, grid.row, grid.vcomm, rootrow, k, grid.nrow, p.dcomm )
	            == myB_KEEPTEST ){}
	        update_left_panels_wGPU_alt_simple( d_piv, ldpiv, p, i, j, *lnext, wks ); 
            convert_left_panels_wGPU(p, scalea, i, j, *lnext);
	        if(p.nprow - schur_row > 0 && p.npcol - schur_col > 0) {
         	    GPUBLAS_SGEMM_EX( p.cuHandle, GPUBLAS_OP_N, GPUBLAS_OP_T,
                                      p.b * (p.nprow - schur_row), p.b * (p.npcol - schur_col), b, &nalf,
                                    ( void * )(*lprev)(schur_row, 'd'), GPU_R_16F, (*lprev).get_lda(),
                                    ( void * )(*rprev)(schur_col, 'd'), GPU_R_16F, (*rprev).get_lda(),
                                    beta, p(schur_row, schur_col, 'd'), GPU_R_32F, p.lda );
  	        }
            rnext->set_start(j + 1);

            #pragma omp parallel for num_threads( NUM_THREAD )
            for ( int kk = 0; kk < NUM_THREAD; kk++ )
            {
                if ( kk == 1 )
                    while ( myBcast( (*lnext)(i,'d'), lnext->get_lda() * b, grid.col, grid.hcomm, rootcol, k, grid.ncol, p.comm)
                        == myB_KEEPTEST ){}
                if ( kk == 2 )
                    while ( myBcast( (*rnext)(j + 1,'d'), rnext->get_lda() * b, grid.row, grid.vcomm, rootrow, k, grid.nrow, p.comm)
                        == myB_KEEPTEST ){}
                
	        }
            ++j;
        } 
        else 
        {
            if(p.nprow - schur_row > 0 && p.npcol - schur_col > 0) {
         	    GPUBLAS_SGEMM_EX( p.cuHandle, GPUBLAS_OP_N, GPUBLAS_OP_T,
                                      p.b * (p.nprow - schur_row), p.b * (p.npcol - schur_col), b, &nalf,
                                    ( void * )(*lprev)(schur_row, 'd'), GPU_R_16F, (*lprev).get_lda(),
                                    ( void * )(*rprev)(schur_col, 'd'), GPU_R_16F, (*rprev).get_lda(),
                                    beta, p(schur_row, schur_col, 'd'), GPU_R_32F, p.lda );
            }
            rnext->set_start(j);
            lnext->set_start(i);
            #pragma omp parallel for num_threads( NUM_THREAD )
	        for ( int kk = 0; kk < NUM_THREAD; kk++ )
            {
                if ( kk == 1 )
                    while ( myBcast( (*lnext)(i,'d'), lnext->get_lda() * b, grid.col, grid.hcomm, rootcol, k, grid.ncol, p.comm)
                        == myB_KEEPTEST ){}
                if ( kk == 2 )
                    while ( myBcast( (*rnext)(j,'d'), rnext->get_lda() * b, grid.row, grid.vcomm, rootrow, k, grid.nrow, p.comm)
                        == myB_KEEPTEST ){}
            }
        }
	    if ( grank == 0  &&  p.progress > 0  &&  ( k % p.progress == 0 ) )
        {
           double d_n  = ( double )n;
           double fact = 1.0 - ( double )( k * b ) / d_n;
           double fraction_left = 100.0 * fact * fact * fact;
           end_time2 = get_utime( );
           time_used += ( ( end_time2 - bgn_time2 ) * 1.0e-09 );
           double proj_t_l = time_used/(100.0-fraction_left)*fraction_left;
           double proj_t_t = time_used + proj_t_l;
           double proj_gf  = 2.0 / 3.0 * d_n * d_n * d_n / proj_t_t * 1.0e-09;
	        double proj_tf = proj_gf * 0.001 / ( double )gsize;
            printf( "Step %d, Rank %d, Elapsed %.2lf, Work left %.2lf%%, Time left: %.2lf, Est Total: %.2lf, GFlops: %.2lf, TFlops/GPU: %.2lf\n", k, grank, time_used, fraction_left, proj_t_l, proj_t_t, proj_gf, proj_tf );
            bgn_time2 = end_time2;
        }
        GPU_DEVICE_SYNCHRONIZE();
        LRPanels<FLow> *t;
        t = lprev;
        lprev = lnext;
        lnext = t;
        t = rprev;
        rprev = rnext;
        rnext = t;
        schur_row = i;
        schur_col = j;
    } 

    lu_time = get_utime( ) - lu_time;
    if ( grank == 0 )
    {
       printf( "LU factorization %.2f sec\n", lu_time * 1.0e-09 );
       fflush( stdout );
    }
}

template <typename FHigh, typename FLow, template <class> class Matgen, int du, bool dd>
void panel_lu_async_wGPU_gdirect_alt2(Panels<FHigh> &p,
                                    LRPanels<FLow> lrpanels[4],
                                    Matgen<FHigh> &mg,
                                    FHigh *piv,
                                    FHigh *d_piv,
                                    size_t ldpiv,
                                    Grid &grid,
                                    bool warmup,Workspace<FHigh,FLow> &wks,Workspace<FHigh,FLow> &wks2) {
    // same as panel_lu but with look-ahead computation for better communication
    // hiding
    // typedef LazyInitializer<FLow, du, dd> LI;
    // typedef SchurUpdator<FHigh, FLow, dd> SU;
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

    // Alternative trsm
    int alternative = p.alt;

    // Diagonal cusolver variables
    int lwork;
    GPUSOLVER_STATUS_T cus_status = GPUSOLVER_STATUS_SUCCESS;
    GPU_ERROR_T cudaStat1 = GPU_SUCCESS;
    float *d_work = NULL;

#ifdef CUDA_OLCF_PLATFORM
    GPUSOLVER_SGETRF_BUFFERSIZE(p.solvHandle, b, b, p(0, 0, 'd'),
                                             p.d_lda, &lwork);
    cudaStat1 = GPU_MALLOC((void **)&d_work, sizeof( FHigh ) * lwork);
    assert(GPU_SUCCESS == cudaStat1);
#endif

    int divider = ceil(nb*0.01);
    const FHigh alf = 1.0f;
    const FHigh nalf = -1.0f;
    const FHigh bet = 1.0f;
    const FHigh *alpha = &alf;
    const FHigh *beta = &bet;

    LRPanels<FLow> *lprev = &lrpanels[0], *rprev = &lrpanels[1], *lnext = &lrpanels[2], *rnext = &lrpanels[3];

    int schur_row = 0;
    int schur_col = 0;

    int communication_type = 0;
    int64_t diag_lu_comp = b * (b * (4ll * b - 3ll) + 5ll) / 6ll;
    int kend = warmup ? 20 : nb;
    if(kend > nb)
        kend = nb;

    int haveGEMM, haveGETRF;
    hipEvent_t eBgn , eEnd ; // gemm
    hipEvent_t eBgn2, eEnd2; // getrf
    hipEvent_t eBgn3, eEnd3; // trsm
    hipEvent_t eBgn4, eEnd4; // convert
    hipEvent_t eBgn5, eEnd5; // convert

    float mSec    = 0.0; // gemm
    float t_mSec  = 0.0; // gemm total
    float mSec2   = 0.0; // getrf
    float t_mSec2 = 0.0; // getrf total
    float mSec3   = 0.0; // trsm 
    float t_mSec3 = 0.0; // trsm total
    float mSec4   = 0.0; // convert
    float t_mSec4 = 0.0; // convert total
    float mSec5   = 0.0; // memcpy
    float t_mSec5 = 0.0; // memcpy total

    PrintLogMsg( "Entering LU\n" );

    hipEvent_t diagonalupdate, lnextupdate,rnextupdate;
    hipEventCreate( &diagonalupdate);
    hipEventCreate( &lnextupdate);
    hipEventCreate( &rnextupdate);

    GPU_EVENTCREATE( &eBgn  );
    GPU_EVENTCREATE( &eEnd  );
    GPU_EVENTCREATE( &eBgn2 );
    GPU_EVENTCREATE( &eEnd2 );
    GPU_EVENTCREATE( &eBgn3 );
    GPU_EVENTCREATE( &eEnd3 );
    GPU_EVENTCREATE( &eBgn4 );
    GPU_EVENTCREATE( &eEnd4 );
    GPU_EVENTCREATE( &eBgn5 );
    GPU_EVENTCREATE( &eEnd5 );

    int          n_gemm = 0;
    static int   mnk   [ 10 ][ 3 ];  // m - n - k dimensions 
    static float tflops[ 10 ];

    double lu_time;

    {
        bgn_time2 = get_utime( );
        lu_time   = bgn_time2;

        // look ahead
        // same as above, but leaves gemm to next iteration
        if(0 == grid.row && 0 == grid.col) {
            // I have the pivot panel.
            FHigh *lu = p(0, 0);
            FHigh *d_lu = p(0, 0, 'd');

            // Solve Diagonal
            //Timer::beg(Timer::DIAG_LU);
    
            if ( !CHECK_BIT( p.skip, 1 ) )
            {
                cus_status = GPUSOLVER_SGETRF( p.solvHandle, b, b,
                            p(0, 0, 'd'), p.d_lda,
                            d_work, NULL, p.d_infoArray );
                assert(GPUSOLVER_STATUS_SUCCESS == cus_status);
            }
    
            GPU_MEMCPY_2D(d_piv,
                        ldpiv * sizeof(FHigh),
                        p(0, 0, 'd'),
                        p.d_lda * sizeof(FHigh),
                        b * sizeof(FHigh),
                        b,
                        GPU_MEMCPY_DEVICE_TO_DEVICE);
            broadcast_pivlu_wGPU_block(p.is_tile, b, lu, lda, d_piv, ldpiv, grid);

            // Update left panels
            update_left_panels_wGPU(d_lu, p.d_lda, p, 1, 0);
            convert_left_panels_wGPU(p, scalea, 1, 0, *lprev);
            GPU_DEVICE_SYNCHRONIZE();
            broadcast_left_panels_gdirect(b, *lprev, 1, nprow, grid);

            // Update right panels
            update_right_panels_wGPU(d_lu, lda, p, 0, 1);
            convert_right_panels_wGPU(p, scaleb, 0, 1, *rprev);
            GPU_DEVICE_SYNCHRONIZE();
            broadcast_right_panels_gdirect(b, *rprev, 1, npcol, grid);

            schur_row = 1;
            schur_col = 1;
        
        } else if(0 == grid.row) {
            receive_pivl_block(b, d_piv, ldpiv, 0, grid);
            update_right_panels_wGPU(d_piv, ldpiv, p, 0, 0);
            convert_right_panels_wGPU(p, scaleb, 0, 0, *rprev);
            GPU_DEVICE_SYNCHRONIZE();
            broadcast_right_panels_gdirect(b, *rprev, 0, npcol, grid);
            receive_left_panels_gdirect(b, *lprev, 1, nprow, 0, grid);
            schur_row = 1;
        } else if(0 == grid.col) {
            receive_pivu_block(b, d_piv, ldpiv, 0, grid);
            update_left_panels_wGPU(d_piv, ldpiv, p, 0, 0);
            convert_left_panels_wGPU(p, scalea, 0, 0, *lprev);
            GPU_DEVICE_SYNCHRONIZE();
            broadcast_left_panels_gdirect(b, *lprev, 0, nprow, grid);
            receive_right_panels_gdirect(b, *rprev, 1, npcol, 0, grid);
            schur_col = 1;
        } else {
            receive_left_panels_gdirect(b, *lprev, 0, nprow, 0, grid);
            receive_right_panels_gdirect(b, *rprev, 0, npcol, 0, grid);
        }
    }
	
	GPU_DEVICE_SYNCHRONIZE();
	// gather times from all nodes ???
	// Define ring comm flags
    int rrpiv_flag, rcpiv_flag, rrp_flag, rlp_flag;
	MPI_Request lrequest, rrequest;

	// Start of Steps
	uint32_t dv_ind = gcdOrder[ grank % 8 ];
	uint32_t sensor_ind = 0;
    uint64_t power;
    int64_t  temp;
    float    counter_resolution;
    uint64_t timestamp;
	double   time_used = 0.0;
   	
    // Main Loop
    for(int k = 1; k < kend; ++k)
    {
        double bgn_time_k = get_utime( );
        int const rootrow = k % grid.nrow;
        int const rootcol = k % grid.ncol;
        int i = k / grid.nrow + (rootrow > grid.row ? 1 : 0);
        int j = k / grid.ncol + (rootcol > grid.col ? 1 : 0);
        if(rootrow == grid.row && rootcol == grid.col) {
            // I have the pivot panel.
            // do GEMM for the diagonal block
	        haveGEMM = 0;
	        if ( !CHECK_BIT( p.skip, 1 ) )
            {
                haveGEMM = 1;
                GPUBLAS_SGEMM_EX( p.cuHandle, GPUBLAS_OP_N, GPUBLAS_OP_T,
                            b, b, b, &nalf,
                            ( void * )(*lprev)(i, 'd'), GPU_R_16F, (*lprev).get_lda(),
                            ( void * )(*rprev)(j, 'd'), GPU_R_16F, (*rprev).get_lda(),
                            beta,       p(i, j, 'd'), GPU_R_32F, p.lda );
            }
            
            if ( !CHECK_BIT( p.skip, 1 ) )
            {
                cus_status = GPUSOLVER_SGETRF(p.solvHandle, b, b, p(i, j, 'd'),
                                                p.d_lda, d_work, NULL, p.d_infoArray );
                assert(GPUSOLVER_STATUS_SUCCESS == cus_status);
            }
	        
            GPU_MEMCPY_2D(d_piv,
                         ldpiv * sizeof(FHigh),
                         p(i, j, 'd'),
                         p.d_lda * sizeof(FHigh),
                         b * sizeof(FHigh),
                         b,
                         GPU_MEMCPY_DEVICE_TO_DEVICE);
	        
            if ( alternative && p.alt == 2){
                cus_status=(rocsolver_strtri(p.solvHandle,
                                    GPUBLAS_FILL_MODE_UPPER,
                                    GPUBLAS_DIAG_NON_UNIT,
                                    p.b,
                                    d_piv,
                                    p.b,
                                    p.d_infoArray));
                                    
                cus_status=(rocsolver_strtri(p.solvHandle,
                                    GPUBLAS_FILL_MODE_LOWER,
                                    GPUBLAS_DIAG_UNIT,
                                    p.b,
                                    d_piv,
                                    p.b,
                                    p.d_infoArray));
            }
            hipEventRecord(diagonalupdate, NULL);
            
            if(p.nprow - i - 1 > 0) {
                if ( !CHECK_BIT( p.skip, 2 ) )
                {
                    haveGEMM = 1;
                    GPUBLAS_SGEMM_EX( p.cuHandle, GPUBLAS_OP_N, GPUBLAS_OP_T,
                                b * (p.nprow - i - 1), b, b, &nalf,
                                ( void * )(*lprev)(i + 1, 'd'), GPU_R_16F, (*lprev).get_lda(),
                                ( void * )(*rprev)(j    , 'd'), GPU_R_16F, (*rprev).get_lda(),
                                beta,       p(i + 1, j, 'd'), GPU_R_32F, p.lda );
                }
            }
            
            if ( alternative)
            {
                if(p.alt == 2){
                    lnext->set_start(i+1);
                    downCast_copy_general((*lnext)(i+1,'d'),(long) lnext->get_lda(),
							(long)(p.nprow - i-1) * p.b,(long)p.b,
							p(i+1,j,'d'), (long)p.d_lda);
                    update_left_panels_wGPU_alt2(d_piv, ldpiv, p, i + 1, j, *lnext, wks );
                }
                if(p.alt == 1)
                    update_left_panels_wGPU_alt(d_piv, ldpiv, p, i + 1, j, *lnext, wks );
            }
            else
            {
                update_left_panels_wGPU(d_piv, ldpiv, p, i + 1, j);
            }
            convert_left_panels_wGPU(p, scalea, i + 1, j, *lnext);
            hipEventRecord(lnextupdate, NULL);
           
            // do GEMM of U
            if(p.npcol - j - 1 > 0) {
                if ( !CHECK_BIT( p.skip, 2 ) )
                {
                    haveGEMM = 1;
                    GPUBLAS_SGEMM_EX( p.cuHandle, GPUBLAS_OP_N, GPUBLAS_OP_T,
                                        b, p.b * (p.npcol - j - 1), b, &nalf,
                                    ( void * )(*lprev)(i    , 'd'), GPU_R_16F, (*lprev).get_lda(),
                                    ( void * )(*rprev)(j + 1, 'd'), GPU_R_16F, (*rprev).get_lda(),
                                        beta, p(i, j + 1, 'd'), GPU_R_32F, p.lda );
                }
            }
            
            if ( alternative )
            {
                if(p.alt == 2){
                    rnext->set_start(j+1);
                    downCast_trans_general((*rnext)(j+1,'d'), (long)rnext->get_lda(),
							(long)p.b, (long)(p.npcol - j-1) * p.b,
							p(i,j+1,'d'), p.d_lda);
                    update_right_panels_wGPU_alt2( d_piv, ldpiv, p, i, j + 1, *rnext, wks2 );
                }
                if(p.alt == 1)
                    update_right_panels_wGPU_alt( d_piv, ldpiv, p, i, j + 1, *rnext, wks );
            }
            else
            {
                update_right_panels_wGPU(d_piv, ldpiv, p, i, j + 1);
            }
            convert_right_panels_wGPU(p, scaleb, i, j + 1, *rnext);
            hipEventRecord(rnextupdate, NULL);
	        
            ++schur_row;
            ++schur_col;
            
            // Launch Trialing update
	        haveGEMM = 0;
	        if(p.nprow - schur_row > 0 && p.npcol - schur_col > 0) {
                if ( !CHECK_BIT( p.skip, 3 ) )
                {
                    haveGEMM = 1;
                    GPU_EVENTRECORD( eBgn, NULL );
                    GPUBLAS_SGEMM_EX( p.cuHandle, GPUBLAS_OP_N, GPUBLAS_OP_T,
                                        p.b * (p.nprow - schur_row), p.b * (p.npcol - schur_col), b, &nalf,
                                    ( void * )(*lprev)(schur_row, 'd'), GPU_R_16F, (*lprev).get_lda(),
                                    ( void * )(*rprev)(schur_col, 'd'), GPU_R_16F, (*rprev).get_lda(),
                                    beta, p(schur_row, schur_col, 'd'), GPU_R_32F, p.lda );
                GPU_EVENTRECORD( eEnd, NULL );
                    mnk[ n_gemm ][ 0 ] = b * ( p.nprow - schur_row );
                    mnk[ n_gemm ][ 1 ] = b * ( p.npcol - schur_col );
                    mnk[ n_gemm ][ 2 ] = b;
                }
            }


            /*hipEventSynchronize(diagonalupdate);
            while ( myBcast(d_piv, ldpiv * b, grid.row, grid.vcomm, rootrow, k, grid.nrow, p.dcomm )
	                        == myB_KEEPTEST ){}
            while ( myBcast(d_piv, ldpiv * b, grid.col, grid.hcomm, rootcol, k, grid.ncol, p.dcomm )
		                == myB_KEEPTEST ){}
            */
            #pragma omp parallel for num_threads( NUM_THREAD )
	        for ( int kk = 0; kk < NUM_THREAD; kk++ )
	        {
	            if ( kk == 1 )
                    while ( myBcast(d_piv, ldpiv * b, grid.row, grid.vcomm, rootrow, k, grid.nrow, p.dcomm )
	                        == myB_KEEPTEST ){}

	            if ( kk == 2 )
	                while ( myBcast(d_piv, ldpiv * b, grid.col, grid.hcomm, rootcol, k, grid.ncol, p.dcomm )
		                == myB_KEEPTEST ){}
	        }
        
            /*hipEventSynchronize(lnextupdate);
            while ( myBcast( (*lnext)(i + 1, 'd'), lnext->get_lda() * b, grid.col, grid.hcomm, rootcol, k, grid.ncol, p.comm)
  	                    == myB_KEEPTEST ){}
            hipEventSynchronize(rnextupdate);
	        while ( myBcast( (*rnext)(j + 1, 'd'), rnext->get_lda() * b, grid.row, grid.vcomm, rootrow, k, grid.nrow, p.comm)
		                == myB_KEEPTEST ){}*/
            
            #pragma omp parallel for num_threads( NUM_THREAD )
	        for ( int kk = 0; kk < NUM_THREAD; kk++ )
	        {
	            if ( kk == 1 ){
                    hipEventSynchronize(lnextupdate);
                    while ( myBcast( (*lnext)(i + 1, 'd'), lnext->get_lda() * b, grid.col, grid.hcomm, rootcol, k, grid.ncol, p.comm)
  	                    == myB_KEEPTEST ){}
                }
	            if ( kk == 2 ){
                    hipEventSynchronize(rnextupdate);
	                while ( myBcast( (*rnext)(j + 1, 'd'), rnext->get_lda() * b, grid.row, grid.vcomm, rootrow, k, grid.nrow, p.comm)
		                == myB_KEEPTEST ){}
                }
            }

            ++i;
            ++j;
        } 
        else if(rootrow == grid.row)
        {
            haveGEMM = 0;
            if(p.npcol - j > 0) {
		        if ( !CHECK_BIT( p.skip, 2 ) )
		        {
	                GPUBLAS_SGEMM_EX( p.cuHandle, GPUBLAS_OP_N, GPUBLAS_OP_T,
                                     b, p.b * (p.npcol - j), b, &nalf,
                                   ( void * )(*lprev)(i, 'd'), GPU_R_16F, (*lprev).get_lda(),
                                   ( void * )(*rprev)(j, 'd'), GPU_R_16F, (*rprev).get_lda(),
                                     beta,       p(i, j, 'd'), GPU_R_32F, p.lda );
  	            }
            }

            if ( alternative )
	        {  
                if(p.alt == 2){
                    rnext->set_start(j);
                    downCast_trans_general((*rnext)(j,'d'), (long)rnext->get_lda(),
                                    (long)p.b, (long)(p.npcol - j) * p.b,
                                    p(i,j,'d'), p.d_lda);
                }
            }
            ++schur_row;
            while ( myBcast(d_piv, ldpiv * b, grid.col, grid.hcomm, rootcol, k, grid.ncol, p.dcomm)
	            == myB_KEEPTEST ){}

            // Update U
            if ( alternative )
	        {  
                if(p.alt == 2)
                    update_right_panels_wGPU_alt2( d_piv, ldpiv, p, i, j, *rnext, wks2);
                if(p.alt == 1)
                    update_right_panels_wGPU_alt( d_piv, ldpiv, p, i, j, *rnext, wks );
	        }
	        else
            {
                update_right_panels_wGPU(d_piv, ldpiv, p, i, j);
            }
	        convert_right_panels_wGPU(p, scaleb, i, j, *rnext);
	        hipEventRecord(rnextupdate, NULL);

            // Lauching Trailing update
	        if(p.nprow - schur_row > 0 && p.npcol - schur_col > 0) {
                if ( !CHECK_BIT( p.skip, 3 ) )
                {
                    GPUBLAS_SGEMM_EX( p.cuHandle, GPUBLAS_OP_N, GPUBLAS_OP_T,
                                                p.b * (p.nprow - schur_row), p.b * (p.npcol - schur_col), b, &nalf,
                                            ( void * )(*lprev)(schur_row, 'd'), GPU_R_16F, (*lprev).get_lda(),
                                            ( void * )(*rprev)(schur_col, 'd'), GPU_R_16F, (*rprev).get_lda(),
                                            beta, p(schur_row, schur_col, 'd'), GPU_R_32F, p.lda);
                }
            }

            lnext->set_start(i + 1);

            /*while ( myBcast( (*lnext)(i + 1,'d'), lnext->get_lda() * b, grid.col, grid.hcomm, rootcol, k, grid.ncol, p.comm)
		                == myB_KEEPTEST ){}

            hipEventSynchronize(rnextupdate);
            while ( myBcast( (*rnext)(j,'d'), rnext->get_lda() * b, grid.row, grid.vcomm, rootrow, k, grid.nrow, p.comm)
		                == myB_KEEPTEST ){}*/

            #pragma omp parallel for num_threads( NUM_THREAD )
	        for ( int kk = 0; kk < NUM_THREAD; kk++ )
	        {
	            if ( kk == 1 ){
	                while ( myBcast( (*lnext)(i + 1,'d'), lnext->get_lda() * b, grid.col, grid.hcomm, rootcol, k, grid.ncol, p.comm)
		                == myB_KEEPTEST ){}
                }
                if ( kk == 2 ){
                    hipEventSynchronize(rnextupdate);
                    while ( myBcast( (*rnext)(j,'d'), rnext->get_lda() * b, grid.row, grid.vcomm, rootrow, k, grid.nrow, p.comm)
		                == myB_KEEPTEST ){}
                }
    	    }
            ++i;

        } else if(rootcol == grid.col) {
	        if(p.nprow - i > 0) {
		        if ( !CHECK_BIT( p.skip, 2 ) )
		        {
                    haveGEMM = 1;
                    GPUBLAS_SGEMM_EX( p.cuHandle, GPUBLAS_OP_N, GPUBLAS_OP_T,
                                            b * (p.nprow - i), b, b, &nalf,
                                        ( void * )(*lprev)(i, 'd'), GPU_R_16F, (*lprev).get_lda(),
                                        ( void * )(*rprev)(j, 'd'), GPU_R_16F, (*rprev).get_lda(),
                                            beta,       p(i, j, 'd'), GPU_R_32F, p.lda );
                }
            }
            
            if ( alternative )
	        {  
                if(p.alt == 2){
                    lnext->set_start(i);
                    downCast_copy_general((*lnext)(i,'d'),(long) lnext->get_lda(),
                                    (long)(p.nprow - i) * p.b,(long)p.b,
                                    p(i,j,'d'), (long)p.d_lda);  
                }
            }        
            ++schur_col;
            while ( myBcast(d_piv, ldpiv * b, grid.row, grid.vcomm, rootrow, k, grid.nrow, p.dcomm )
	            == myB_KEEPTEST ){}

            if ( alternative )
            {
                if(p.alt == 2)
                    update_left_panels_wGPU_alt2( d_piv, ldpiv, p, i, j, *lnext, wks );
                if(p.alt == 1)
                    update_left_panels_wGPU_alt( d_piv, ldpiv, p, i, j, *lnext, wks );
            }
            else
            {
                update_left_panels_wGPU(d_piv, ldpiv, p, i, j);
            }
            convert_left_panels_wGPU(p, scalea, i, j, *lnext);
            hipEventRecord(lnextupdate, NULL);

            if(p.nprow - schur_row > 0 && p.npcol - schur_col > 0) {
         	    if ( !CHECK_BIT( p.skip, 3 ) )
                {
                    haveGEMM = 1;
                    GPUBLAS_SGEMM_EX( p.cuHandle, GPUBLAS_OP_N, GPUBLAS_OP_T,
                                        p.b * (p.nprow - schur_row), p.b * (p.npcol - schur_col), b, &nalf,
                                        ( void * )(*lprev)(schur_row, 'd'), GPU_R_16F, (*lprev).get_lda(),
                                        ( void * )(*rprev)(schur_col, 'd'), GPU_R_16F, (*rprev).get_lda(),
                                        beta, p(schur_row, schur_col, 'd'), GPU_R_32F, p.lda );
                }
            }

            rnext->set_start(j + 1);
            /*hipEventSynchronize(lnextupdate);
            while ( myBcast( (*lnext)(i,'d'), lnext->get_lda() * b, grid.col, grid.hcomm, rootcol, k, grid.ncol, p.comm)
                        == myB_KEEPTEST ){}
            while ( myBcast( (*rnext)(j + 1,'d'), rnext->get_lda() * b, grid.row, grid.vcomm, rootrow, k, grid.nrow, p.comm)
                        == myB_KEEPTEST ){}*/
                        
            #pragma omp parallel for num_threads( NUM_THREAD )
            for ( int kk = 0; kk < NUM_THREAD; kk++ )
            {
                if ( kk == 1 ){
                    hipEventSynchronize(lnextupdate);
                    while ( myBcast( (*lnext)(i,'d'), lnext->get_lda() * b, grid.col, grid.hcomm, rootcol, k, grid.ncol, p.comm)
                        == myB_KEEPTEST ){}
                }
                if ( kk == 2 ){
                    while ( myBcast( (*rnext)(j + 1,'d'), rnext->get_lda() * b, grid.row, grid.vcomm, rootrow, k, grid.nrow, p.comm)
                        == myB_KEEPTEST ){}
                }
	        }			
            ++j;

        } else {
            haveGEMM = 0;
            if(p.nprow - schur_row > 0 && p.npcol - schur_col > 0) {
         	    if ( !CHECK_BIT( p.skip, 3 ) )
		        {
                    haveGEMM = 1;
                    GPUBLAS_SGEMM_EX( p.cuHandle, GPUBLAS_OP_N, GPUBLAS_OP_T,
                                        p.b * (p.nprow - schur_row), p.b * (p.npcol - schur_col), b, &nalf,
                                        ( void * )(*lprev)(schur_row, 'd'), GPU_R_16F, (*lprev).get_lda(),
                                        ( void * )(*rprev)(schur_col, 'd'), GPU_R_16F, (*rprev).get_lda(),
                                        beta, p(schur_row, schur_col, 'd'), GPU_R_32F, p.lda );
                }
            }

            rnext->set_start(j);
            lnext->set_start(i);
            /*while ( myBcast( (*lnext)(i,'d'), lnext->get_lda() * b, grid.col, grid.hcomm, rootcol, k, grid.ncol, p.comm)
                        == myB_KEEPTEST ){}
            while ( myBcast( (*rnext)(j,'d'), rnext->get_lda() * b, grid.row, grid.vcomm, rootrow, k, grid.nrow, p.comm)
                        == myB_KEEPTEST ){}*/

            #pragma omp parallel for num_threads( NUM_THREAD )
            for ( int kk = 0; kk < NUM_THREAD; kk++ )
            {
                if ( kk == 1 )
                    while ( myBcast( (*lnext)(i,'d'), lnext->get_lda() * b, grid.col, grid.hcomm, rootcol, k, grid.ncol, p.comm)
                        == myB_KEEPTEST ){}
                if ( kk == 2 )
                    while ( myBcast( (*rnext)(j,'d'), rnext->get_lda() * b, grid.row, grid.vcomm, rootrow, k, grid.nrow, p.comm)
                        == myB_KEEPTEST ){}
	        }
        }

        GPU_DEVICE_SYNCHRONIZE();

        if ( grank == 0  &&  p.progress > 0  &&  ( k % p.progress == 0 ) )
        {
            double d_n  = ( double )n;
            double fact = 1.0 - ( double )( k * b ) / d_n;
            double fraction_left = 100.0 * fact * fact * fact;
            end_time2 = get_utime( );
            time_used += ( ( end_time2 - bgn_time2 ) * 1.0e-09 );
            double proj_t_l = time_used/(100.0-fraction_left)*fraction_left;
            double proj_t_t = time_used + proj_t_l;
            double proj_gf  = 2.0 / 3.0 * d_n * d_n * d_n / proj_t_t * 1.0e-09;
            double proj_tf = proj_gf * 0.001 / ( double )gsize;
            printf( "Step %d, Rank %d, Elapsed %.2lf, Work left %.2lf%%, Time left: %.2lf, Est Total: %.2lf, GFlops: %.2lf, TFlops/GPU: %.2lf\n", k, grank, time_used, fraction_left, proj_t_l, proj_t_t, proj_gf, proj_tf );
	        bgn_time2 = end_time2;
        }

        LRPanels<FLow> *t;
        t = lprev;
        lprev = lnext;
        lnext = t;
        t = rprev;
        rprev = rnext;
        rnext = t;
        schur_row = i;
        schur_col = j;
    } 

    lu_time = get_utime( ) - lu_time;
    if ( grank == 0 )
    {
       printf( "LU factorization %.2f sec\n", lu_time * 1.0e-09 );
       fflush( stdout );
    }

    GPU_EVENTDESTROY( eBgn  );
    GPU_EVENTDESTROY( eEnd  );
    GPU_EVENTDESTROY( eBgn2 );
    GPU_EVENTDESTROY( eEnd2 );
    GPU_EVENTDESTROY( eBgn3 );
    GPU_EVENTDESTROY( eEnd3 );
    GPU_EVENTDESTROY( eBgn4 );
    GPU_EVENTDESTROY( eEnd4 );
    GPU_EVENTDESTROY( eBgn5 );
    GPU_EVENTDESTROY( eEnd5 );

}

#endif