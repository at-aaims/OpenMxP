#ifndef PANEL_TRF_B_HPP
#define PANEL_TRF_B_HPP
#include "panel_trf.hpp"

#include "device_macros.h"

int64_t bgn_time2;
int64_t end_time2;

template <typename F>
int myBcast_irecv(F *buffer, int count, int my_rank, MPI_Comm comm, int root, int k, int p_size, int type, MPI_Request* myRequest, int firsttime) {
    int next_guy, prev_guy;
    int go = 0;
    int ierr;
    MPI_Status myStatus;
    MPI_Datatype datatype = T2MPI<F>::type;

    if(type == RB_1R_IR) {
        if(my_rank == root) {
            next_guy = (my_rank + 1) % p_size;
            MPI_Send(buffer, count, datatype, next_guy, k, comm);
        } else {
            prev_guy = (my_rank - 1) % p_size;
            if(prev_guy < 0) {
                prev_guy += p_size;
            }
	    if( firsttime ) MPI_Irecv(buffer, count, datatype, prev_guy, k, comm, myRequest);
            
			ierr = MPI_Test(myRequest, &go, &myStatus);
            if(ierr == MPI_SUCCESS) {
                if(go != 0) {
                    //MPI_Recv(buffer, count, datatype, prev_guy, k, comm, &myStatus);
                    next_guy = (my_rank + 1) % p_size;
                    if(next_guy != root) {
                        MPI_Send(buffer, count, datatype, next_guy, k, comm);
                    }
                } else {
                    return myB_KEEPTEST;
                }
            } else {
                printf("Something wrong in ring bcast\n");
                return myB_KEEPTEST;
            }
        }
        return myB_SUCCESS;
    } else if(type == RB_2RM_IR) {
        int next_next_guy = (root + 2) % p_size;
        int mid_guy = (root + p_size / 2) % p_size;
        if(my_rank == root) {
            next_guy = (my_rank + 1) % p_size;
            MPI_Send(buffer, count, datatype, next_guy, k, comm);
            MPI_Send(buffer, count, datatype, next_next_guy, k, comm);
            MPI_Send(buffer, count, datatype, mid_guy, k, comm);
        } else if(my_rank == (root + 1) % p_size)  // I am the root next guy, who dont send
        {
            prev_guy = root;
	    if ( firsttime ) MPI_Irecv(buffer, count, datatype, prev_guy, k, comm, myRequest);
            	//ierr = MPI_Iprobe(prev_guy, k, comm, &go, &myStatus);
           	ierr = MPI_Test(myRequest, &go, &myStatus);
  			if(ierr == MPI_SUCCESS) {
                if(go != 0) {
                } else {
                    return myB_KEEPTEST;
                }
            } else {
                printf("Something wrong in 2RM ring bcast\n");
                return myB_KEEPTEST;
            }
        } else {
            if(my_rank == next_next_guy || my_rank == mid_guy)
                prev_guy = root;
            else
                prev_guy = (my_rank - 1 + p_size) % p_size;

			if(firsttime)
				MPI_Irecv(buffer, count, datatype, prev_guy, k, comm, myRequest);
            ierr = MPI_Test(myRequest, &go, &myStatus);
            if(ierr == MPI_SUCCESS) {
                if(go != 0) {
                    next_guy = (my_rank + 1) % p_size;
                    if((next_guy != root) && (next_guy != mid_guy)) {
                        MPI_Send(buffer, count, datatype, next_guy, k, comm);
                    }
                } else {
                    return myB_KEEPTEST;
                }
            } else {
                printf("Something wrong in 2RM ring bcast\n");
                return myB_KEEPTEST;
            }
        }
        return myB_SUCCESS;
    } else if(type == RB_1RM_IR) {
        int next_next_guy = (root + 2) % p_size;
        if(my_rank == root) {
            next_guy = (my_rank + 1) % p_size;
            MPI_Send(buffer, count, datatype, next_guy, k, comm);
            MPI_Send(buffer, count, datatype, next_next_guy, k, comm);
        } 
		else if(my_rank == (root + 1) % p_size)  // I am the root next guy, who dont send
        {
			prev_guy = root;
           	if(firsttime)
				MPI_Irecv(buffer, count, datatype, prev_guy, k, comm, myRequest);
        	ierr = MPI_Test(myRequest, &go, &myStatus);
		 	if(ierr == MPI_SUCCESS) {
                if(go != 0) {
                } else {
                    return myB_KEEPTEST;
                }
            } else {
                // printf("Something wrong in 1RM ring bcast\n");
                return myB_KEEPTEST;
            }
        } else {
            if(my_rank == next_next_guy) {
                prev_guy = root;
            } else {
                prev_guy = (my_rank - 1 + p_size) % p_size;
            }
			
			if(firsttime)
				MPI_Irecv(buffer, count, datatype, prev_guy, k, comm, myRequest);
            	
            ierr = MPI_Test(myRequest, &go, &myStatus);
			if(ierr == MPI_SUCCESS) {
                if(go != 0) {
                    next_guy = (my_rank + 1) % p_size;
                    if((ierr == MPI_SUCCESS) && (next_guy != root)) {
                        ierr = MPI_Send(buffer, count, datatype, next_guy, k, comm);
                    }
                } else {
                    return myB_KEEPTEST;
                }
            } else {
                printf("Something wrong in 1RM ring bcast\n");
                return myB_KEEPTEST;
            }
        }
        return myB_SUCCESS;
    } //else if(type == RB_Double_1R_IR) {

	return 0; 

	//}
}


template <typename F>
inline int myBcast(F *buffer, int count, int my_rank, MPI_Comm comm, int root, int k, int p_size, int type) {
    int next_guy, prev_guy;
    int go = 0;
    int ierr;
    MPI_Status myStatus;
    MPI_Datatype datatype = T2MPI<F>::type;

    if(type == RB_1R) {
        if(my_rank == root) {
            next_guy = (my_rank + 1) % p_size;
            MPI_Send(buffer, count, datatype, next_guy, k, comm);
        } else {
            prev_guy = (my_rank - 1) % p_size;
            if(prev_guy < 0) {
                prev_guy += p_size;
            }
//          ierr = MPI_Iprobe(prev_guy, k, comm, &go, &myStatus);
            ierr = MPI_SUCCESS;
	        go = 1;
            if(ierr == MPI_SUCCESS) {
                if(go != 0) {
                    ierr = MPI_Recv(buffer, count, datatype, prev_guy, k, comm, &myStatus);
                    next_guy = (my_rank + 1) % p_size;
                    if((ierr == MPI_SUCCESS) && (next_guy != root)) {
                        ierr = MPI_Send(buffer, count, datatype, next_guy, k, comm);
                    }
                } else {
                    return myB_KEEPTEST;
                }
            } else {
                printf("Something wrong in ring bcast\n");
                return myB_KEEPTEST;
            }
        }
        return myB_SUCCESS;
    } else if(type == RB_2RM) {
        int next_next_guy = (root + 2) % p_size;
        int mid_guy = (root + p_size / 2) % p_size;
        if(my_rank == root) {
            next_guy = (my_rank + 1) % p_size;
            MPI_Send(buffer, count, datatype, next_guy, k, comm);
            MPI_Send(buffer, count, datatype, next_next_guy, k, comm);
            MPI_Send(buffer, count, datatype, mid_guy, k, comm);
        } else if(my_rank == (root + 1) % p_size)  // I am the root next guy, who dont send
        {
            prev_guy = root;
//          ierr = MPI_Iprobe(prev_guy, k, comm, &go, &myStatus);
            ierr = MPI_SUCCESS;
            go = 1;
            if(ierr == MPI_SUCCESS) {
                if(go != 0) {
                    ierr = MPI_Recv(buffer, count, datatype, prev_guy, k, comm, &myStatus);
                } else {
                    return myB_KEEPTEST;
                }
            } else {
                printf("Something wrong in 2RM ring bcast\n");
                return myB_KEEPTEST;
            }
        } else {
            if(my_rank == next_next_guy || my_rank == mid_guy)
                prev_guy = root;
            else
                prev_guy = (my_rank - 1 + p_size) % p_size;

//           ierr = MPI_Iprobe(prev_guy, k, comm, &go, &myStatus);
            ierr = MPI_SUCCESS;
	    go = 1;
            if(ierr == MPI_SUCCESS) {
                if(go != 0) {
                    ierr = MPI_Recv(buffer, count, datatype, prev_guy, k, comm, &myStatus);
                    next_guy = (my_rank + 1) % p_size;
                    if((ierr == MPI_SUCCESS) && (next_guy != root) && (next_guy != mid_guy)) {
                        ierr = MPI_Send(buffer, count, datatype, next_guy, k, comm);
                    }
                } else {
                    return myB_KEEPTEST;
                }
            } else {
                printf("Something wrong in 2RM ring bcast\n");
                return myB_KEEPTEST;
            }
        }
        return myB_SUCCESS;
    } else if(type == RB_1RM) {
        int next_next_guy = (root + 2) % p_size;
        if(my_rank == root) {
            next_guy = (my_rank + 1) % p_size;
            MPI_Send(buffer, count, datatype, next_guy, k, comm);
            MPI_Send(buffer, count, datatype, next_next_guy, k, comm);
        } else if(my_rank == (root + 1) % p_size)  // I am the root next guy, who dont send
        {

            prev_guy = root;
            ierr = MPI_Iprobe(prev_guy, k, comm, &go, &myStatus);
            if(ierr == MPI_SUCCESS) {
                if(go != 0) {
                    ierr = MPI_Recv(buffer, count, datatype, prev_guy, k, comm, &myStatus);
                } else {
                    return myB_KEEPTEST;
                }
            } else {
                // printf("Something wrong in 1RM ring bcast\n");
                return myB_KEEPTEST;
            }
        } else {
            if(my_rank == next_next_guy) {
                prev_guy = root;
            } else {
                prev_guy = (my_rank - 1 + p_size) % p_size;
            }

            ierr = MPI_Iprobe(prev_guy, k, comm, &go, &myStatus);
            if(ierr == MPI_SUCCESS) {
                if(go != 0) {
                    ierr = MPI_Recv(buffer, count, datatype, prev_guy, k, comm, &myStatus);
                    next_guy = (my_rank + 1) % p_size;
                    if((ierr == MPI_SUCCESS) && (next_guy != root)) {
                        ierr = MPI_Send(buffer, count, datatype, next_guy, k, comm);
                    }
                } else {
                    return myB_KEEPTEST;
                }
            } else {
                printf("Something wrong in 1RM ring bcast\n");
                return myB_KEEPTEST;
            }
        }
        return myB_SUCCESS;
    } else {
        printf("Wrong selection on Bcast method\n");
	exit( 10 );
		return myB_KEEPTEST;
    }
}

template <typename F>
inline void broadcast_pivlu_wGPU_block(bool is_tile, int b, F *lu, size_t lda, F *piv, size_t ldpiv, Grid &grid) {
    // broadcast the pivot block to row and column
    Timer::beg(Timer::DIAG_BCAST);
    MPI_Bcast(piv, ldpiv * b, T2MPI<F>::type, grid.row, grid.vcomm);
    MPI_Bcast(piv, ldpiv * b, T2MPI<F>::type, grid.col, grid.hcomm);
    Timer::end(Timer::DIAG_BCAST);
}

template <typename F>
inline void receive_pivu_block(int b, F *piv, int ldpiv, int root, Grid &grid) {
    Timer::beg(Timer::DIAG_BCAST);
    MPI_Bcast(piv, ldpiv * b, T2MPI<F>::type, root, grid.vcomm);
    Timer::end(Timer::DIAG_BCAST);
}

template <typename F>
inline void receive_pivl_block(int b, F *piv, int ldpiv, int root, Grid &grid) {
    Timer::beg(Timer::DIAG_BCAST);
    MPI_Bcast(piv, ldpiv * b, T2MPI<F>::type, root, grid.hcomm);
    Timer::end(Timer::DIAG_BCAST);
}

template <typename F>
void broadcast_left_panels_block(int b, const LRPanels<F> &lp, int rowstart, int nrow, Grid &grid) {
    Timer::beg(Timer::LCOL_BCAST);
    MPI_Bcast(const_cast<F *>(lp(rowstart)), lp.get_lda() * b, T2MPI<F>::type, grid.col, grid.hcomm);
    Timer::end(Timer::LCOL_BCAST);
}

template <typename F>
void receive_left_panels_block(int b, LRPanels<F> &lp, int rowstart, int nrow, int root, Grid &grid) {
    Timer::beg(Timer::LCOL_BCAST);
    lp.set_start(rowstart);
    MPI_Bcast(const_cast<F *>(lp(rowstart)), lp.get_lda() * b, T2MPI<F>::type, root, grid.hcomm);
    Timer::end(Timer::LCOL_BCAST);
}
template <typename F> void broadcast_right_panels_block(int b, LRPanels<F> &rp, int colstart, int ncol, Grid &grid) {
    Timer::beg(Timer::RROW_BCAST);
    MPI_Bcast(const_cast<F *>(rp(colstart)), rp.get_lda() * b, T2MPI<F>::type, grid.row, grid.vcomm);
    Timer::end(Timer::RROW_BCAST);
}
template <typename F>
void receive_right_panels_block(int b, LRPanels<F> &rp, int colstart, int ncol, int root, Grid &grid) {
    Timer::beg(Timer::RROW_BCAST);
    rp.set_start(colstart);
    //	T2MPI<F>::type, root, grid.vcomm, req.get_request());
    MPI_Bcast(const_cast<F *>(rp(colstart)), rp.get_lda() * b, T2MPI<F>::type, root, grid.vcomm);
    Timer::end(Timer::RROW_BCAST);
}

template <typename FHigh, typename FLow, template <class> class Matgen, int du, bool dd>
void panel_lu_async_wGPU_blockBcast(Panels<FHigh> &p,
                                    LRPanels<FLow> lrpanels[4],
                                    Matgen<FHigh> &mg,
                                    FHigh *piv,
                                    FHigh *d_piv,
                                    size_t ldpiv,
                                    Grid &grid,
                                    bool warmup = false) {

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

    // Diagonal cusolver variables
    int lwork;

    GPUSOLVER_STATUS_T cus_status = GPUSOLVER_STATUS_SUCCESS;
    GPU_ERROR_T cudaStat1 = GPU_SUCCESS;

    /* Dont need if we keep it on default stream
    cudaStat1 = GPU_STREAM_CREATE_WITH_FLAGS(&stream, GPU_STREAM_NON_BLOCKING);
    assert(GPU_SUCCESS == cudaStat1);
    cus_status = GPUSOLVER_SET_STREAM(cusolverH, stream);
    assert(GPUSOLVER_STATUS_SUCCESS == cus_status); */

    float *d_work = NULL;

#ifdef CUDA_OLCF_PLATFORM
//  cus_status = GPUSOLVER_SGETRF_BUFFERSIZE(p.solvHandle, b, b, p(0, 0, 'd'),
    GPUSOLVER_SGETRF_BUFFERSIZE(p.solvHandle, b, b, p(0, 0, 'd'),
                                             p.d_lda, &lwork);

//  assert(GPUSOLVER_STATUS_SUCCESS == cus_status);

    cudaStat1 = GPU_MALLOC((void **)&d_work, sizeof(float) * lwork);
    assert(GPU_SUCCESS == cudaStat1);
#endif

    int divider = ceil(nb*0.01);
    const float alf = 1.0f;
    const float nalf = -1.0f;
    const float bet = 1.0f;
    const float *alpha = &alf;
    const float *beta = &bet;

    LRPanels<FLow> *lprev = &lrpanels[0], *rprev = &lrpanels[1], *lnext = &lrpanels[2], *rnext = &lrpanels[3];

    int schur_row = 0;
    int schur_col = 0;

    int communication_type = 0;

    int64_t diag_lu_comp = b * (b * (4ll * b - 3ll) + 5ll) / 6ll;
    int kend = warmup ? 20 : nb;
    if(kend > nb)
        kend = nb;

    {
        
		if(grank == 0)
		   bgn_time2 = get_utime( );
      
		// look ahead
        // same as above, but leaves gemm to next iteration
        if(0 == grid.row && 0 == grid.col) {
            // I have the pivot panel.
            FHigh *lu = p(0, 0);
            FHigh *d_lu = p(0, 0, 'd');
//print_matrix(p(0,0),p.lda,p.lda*p.lda);
 
//print_matrix(piv,b,b*b);
 
            Timer::beg(Timer::DIAG_LU);

            // Solve Diagonal
            // float* diag_address= p(0,0,'d');
            // checkGPUblas(GPUBLAS_SGETRF_BATCHED(p.cu_handle, b, &diag_address,
            // p.d_lda, NULL, &inforArry, 1));

	    // Inserting solver log
	    // rocsolver_log_begin( );
	    // rocsolver_log_set_layer_mode( rocblas_layer_mode_log_trace );
	    // rocsolver_log_set_max_levels( 5 );
	    
//          cus_status = GPUSOLVER_SGETRF(p.solvHandle, b, b, p(0, 0, 'd'),
            GPUSOLVER_SGETRF(p.solvHandle, b, b, p(0, 0, 'd'),
                                          p.d_lda, d_work, NULL, p.d_infoArray );
	    // rocsolver_log_end( );
	      
            cudaStat1 = GPU_DEVICE_SYNCHRONIZE();

//          assert(GPUSOLVER_STATUS_SUCCESS == cus_status);
            assert(GPU_SUCCESS == cudaStat1);

            // Copy piv back to cpu
            Timer::beg(Timer::MEMCPY);

            GPU_MEMCPY_2D(piv,
                          ldpiv * sizeof(FHigh),
                          p(0, 0, 'd'),
                          p.d_lda * sizeof(FHigh),
                          b * sizeof(FHigh),
                          b,
                          GPU_MEMCPY_DEVICE_TO_HOST);

//print_matrix(piv,b,b*b);
 
            Timer::end(Timer::MEMCPY);

            Timer::end(Timer::DIAG_LU, false, diag_lu_comp);

            broadcast_pivlu_wGPU_block(p.is_tile, b, lu, lda, piv, ldpiv, grid);

            update_left_panels_wGPU(p(0, 0, 'd'), p.d_lda, p, 1, 0);
	    	convert_left_panels_wGPU(p, scalea, 1, 0, *lprev);
			
            Timer::beg(Timer::MEMCPY);
            GPU_MEMCPY( (*lprev)(1), (*lprev)(1, 'd'),
		    p.b * (*lprev).get_lda() * sizeof(__half), GPU_MEMCPY_DEVICE_TO_HOST );
            Timer::end(Timer::MEMCPY);

//print_matrix((*lprev)(1),lprev->get_lda(),lprev->get_lda()*b);

//return;
            broadcast_left_panels_block(b, *lprev, 1, nprow, grid);
	
            update_right_panels_wGPU(d_lu, lda, p, 0, 1);
            convert_right_panels_wGPU(p, scaleb, 0, 1, *rprev);

            Timer::beg(Timer::MEMCPY);
            GPU_MEMCPY( (*rprev)(1), (*rprev)(1, 'd'),
                    p.b * (*rprev).get_lda() * sizeof(__half), GPU_MEMCPY_DEVICE_TO_HOST );
            Timer::end(Timer::MEMCPY);

            broadcast_right_panels_block(b, *rprev, 1, npcol, grid);

            schur_row = 1;
            schur_col = 1;
        } else if(0 == grid.row) {
            receive_pivl_block(b, piv, ldpiv, 0, grid);

            Timer::beg(Timer::MEMCPY);
            GPU_MEMCPY( d_piv, piv, ldpiv * ldpiv * sizeof(FHigh),
                       GPU_MEMCPY_HOST_TO_DEVICE );
            Timer::end(Timer::MEMCPY);

            update_right_panels_wGPU(d_piv, ldpiv, p, 0, 0);
            convert_right_panels_wGPU(p, scaleb, 0, 0, *rprev);

            Timer::beg(Timer::MEMCPY);
            GPU_MEMCPY( (*rprev)(0), (*rprev)(0, 'd'),
                       p.b * (*rprev).get_lda() * sizeof(__half),
                       GPU_MEMCPY_DEVICE_TO_HOST );
            Timer::end(Timer::MEMCPY);

            receive_left_panels_block(b, *lprev, 1, nprow, 0, grid);
            broadcast_right_panels_block(b, *rprev, 0, npcol, grid);

            schur_row = 1;
        } else if(0 == grid.col) {
            receive_pivu_block(b, piv, ldpiv, 0, grid);

            Timer::beg(Timer::MEMCPY);
            GPU_MEMCPY( d_piv, piv, ldpiv * ldpiv * sizeof(FHigh),
                       GPU_MEMCPY_HOST_TO_DEVICE );
            Timer::end(Timer::MEMCPY);

            update_left_panels_wGPU(d_piv, ldpiv, p, 0, 0);
            convert_left_panels_wGPU(p, scalea, 0, 0, *lprev);

            Timer::beg(Timer::MEMCPY);
            GPU_MEMCPY( (*lprev)(0), (*lprev)(0, 'd'),
                       p.b * (*lprev).get_lda() * sizeof(__half),
                       GPU_MEMCPY_DEVICE_TO_HOST );
            Timer::end(Timer::MEMCPY);

            broadcast_left_panels_block(b, *lprev, 0, nprow, grid);
            receive_right_panels_block(b, *rprev, 1, npcol, 0, grid);

            schur_col = 1;
        } else {
            receive_left_panels_block(b, *lprev, 0, nprow, 0, grid);
            receive_right_panels_block(b, *rprev, 0, npcol, 0, grid);
        }

        if(0 != grid.col && 0 == grid.row) {
            Timer::beg(Timer::MEMCPY);
            GPU_MEMCPY( (*lprev)(1, 'd'), (*lprev)(1),
                    p.b * (*lprev).get_lda() * sizeof(__half), GPU_MEMCPY_HOST_TO_DEVICE );
            Timer::end(Timer::MEMCPY);
        } else if(0 != grid.row && 0 == grid.col) {
            Timer::beg(Timer::MEMCPY);
            GPU_MEMCPY( (*rprev)(1, 'd'), (*rprev)(1),
                    p.b * (*rprev).get_lda() * sizeof(__half), GPU_MEMCPY_HOST_TO_DEVICE );
            Timer::end(Timer::MEMCPY);
        } else if(0 != grid.row && 0 != grid.col) {
            Timer::beg(Timer::MEMCPY);
            GPU_MEMCPY( (*lprev)(0, 'd'), (*lprev)(0),
                    p.b * (*lprev).get_lda() * sizeof(__half), GPU_MEMCPY_HOST_TO_DEVICE );
            GPU_MEMCPY( (*rprev)(0, 'd'), (*rprev)(0),
                    p.b * (*rprev).get_lda() * sizeof(__half), GPU_MEMCPY_HOST_TO_DEVICE );
            Timer::end(Timer::MEMCPY);
        }
    }

    // Define ring comm flags
    int rrpiv_flag, rcpiv_flag, rrp_flag, rlp_flag;

	MPI_Request lrequest, rrequest;

    double time_used = 0.0;
    for(int k = 1; k < kend; ++k) {
        
		if( grank == 0 && k %(divider) == 0 && k < divider*60)
		{
		   end_time2 = get_utime( );
		   double d_n  = ( double )n;
		   double fact = 1.0 - ( double )( k * b ) / d_n;
		   double fraction_left = 100.0 * fact * fact * fact;
		   time_used += ( ( end_time2 - bgn_time2 ) * 1.0e-09 );
		   double proj_t_l = time_used/(100.0-fraction_left)*fraction_left;
		   double proj_t_t = time_used + proj_t_l; 
		   double proj_gf  = 2.0 / 3.0 * d_n * d_n * d_n / proj_t_t * 1.0e-09;
//                 INFO( "Elapsed {:.2f}, Work left {:.2f}%, Time left: {:.2f}, Est Total: {:.2f}, GFlops: {:.2f}", time_used, fraction_left, proj_t_l, proj_t_t, proj_gf );
		   bgn_time2 = get_utime( );
		}

        // Timer::put(Timer::MISC));
        int const rootrow = k % grid.nrow;
        int const rootcol = k % grid.ncol;
        int i = k / grid.nrow + (rootrow > grid.row ? 1 : 0);
        int j = k / grid.ncol + (rootcol > grid.col ? 1 : 0);
        if(rootrow == grid.row && rootcol == grid.col) {
            // I have the pivot panel.
            // do GEMM for the diagonal block
            Timer::beg(Timer::GEMM_UPDATE);
            GPUBLAS_SGEMM_EX( p.cuHandle, GPUBLAS_OP_N, GPUBLAS_OP_T, b, b, b, &nalf,
                          (void *)(*lprev)(i, 'd'), GPU_R_16F,
                          (*lprev).get_lda(), (void *)(*rprev)(j, 'd'),
                          GPU_R_16F, (*rprev).get_lda(), beta, p(i, j, 'd'),
                          GPU_R_32F, p.lda );
            Timer::end(Timer::GEMM_UPDATE, false, 2ull * p.b * p.b * p.b);

            FHigh *lu = p(i, j);
            FHigh *d_lu = p(i, j, 'd');

            cudaStat1 = GPU_DEVICE_SYNCHRONIZE();

            // Solve diag
            Timer::beg(Timer::DIAG_LU);

	    // inserting solver log
	    // rocsolver_log_begin( );
	    // rocsolver_log_set_layer_mode( rocblas_layer_mode_log_trace );
	    // rocsolver_log_set_max_levels( 4 );

//          cus_status = GPUSOLVER_SGETRF(p.solvHandle, b, b, p(i, j, 'd'),
            GPUSOLVER_SGETRF(p.solvHandle, b, b, p(i, j, 'd'),
                                          p.d_lda, d_work, NULL, p.d_infoArray );

	    // rocsolver_log_end( );

            cudaStat1 = GPU_DEVICE_SYNCHRONIZE();

//          assert(GPUSOLVER_STATUS_SUCCESS == cus_status);
            assert(GPU_SUCCESS == cudaStat1);
            Timer::beg(Timer::MEMCPY);

            GPU_MEMCPY_2D(piv,
                         ldpiv * sizeof(FHigh),
                         p(i, j, 'd'),
                         p.d_lda * sizeof(FHigh),
                         b * sizeof(FHigh),
                         b,
                         GPU_MEMCPY_DEVICE_TO_HOST);
            Timer::end(Timer::MEMCPY);
            Timer::end(Timer::DIAG_LU, false, diag_lu_comp);

            if(p.comm == 1) {
                broadcast_pivlu_wGPU_block(p.is_tile, b, lu, lda, piv, ldpiv, grid);
            } else if(p.comm<5) {
                rrpiv_flag = rcpiv_flag = myB_KEEPTEST;
                do {
                    if(rrpiv_flag == myB_KEEPTEST)
                        rrpiv_flag = myBcast(piv, ldpiv * b, grid.row, grid.vcomm, rootrow, k, grid.nrow, p.comm);
                    if(rcpiv_flag == myB_KEEPTEST)
                        rcpiv_flag = myBcast(piv, ldpiv * b, grid.col, grid.hcomm, rootcol, k, grid.ncol, p.comm);
                } while(rrpiv_flag == myB_KEEPTEST || rcpiv_flag == myB_KEEPTEST);
            }else{
				rrpiv_flag = rcpiv_flag = myB_KEEPTEST;
                int firsttime = 1;
				do {
                    if(rrpiv_flag == myB_KEEPTEST)
                        rrpiv_flag = myBcast_irecv(piv, ldpiv * b, grid.row, grid.vcomm, rootrow, k, grid.nrow, p.comm, &lrequest, firsttime);
                    if(rcpiv_flag == myB_KEEPTEST)
                        rcpiv_flag = myBcast_irecv(piv, ldpiv * b, grid.col, grid.hcomm, rootcol, k, grid.ncol, p.comm, &rrequest, firsttime);
            	    firsttime =0;
			    } while(rrpiv_flag == myB_KEEPTEST || rcpiv_flag == myB_KEEPTEST);
			}
            // do GEMM of L
            Timer::beg(Timer::GEMM_UPDATE);
            if(p.nprow - i - 1 > 0) {
                GPUBLAS_SGEMM_EX(p.cuHandle,
                              GPUBLAS_OP_N,
                              GPUBLAS_OP_T,
                              b * (p.nprow - i - 1),
                              b,
                              b,
                              &nalf,
                              (void *)(*lprev)(i + 1, 'd'),
                              GPU_R_16F,
                              (*lprev).get_lda(),
                              (void *)(*rprev)(j, 'd'),
                              GPU_R_16F,
                              (*rprev).get_lda(),
                              beta,
                              p(i + 1, j, 'd'),
                              GPU_R_32F,
                              p.lda);
            }
            if( p.sync > 0 ) GPU_DEVICE_SYNCHRONIZE();

            Timer::end(Timer::GEMM_UPDATE, false, 2ull * p.b * p.b * p.b * (p.nprow - i - 1));

            update_left_panels_wGPU(d_lu, lda, p, i + 1, j);
            convert_left_panels_wGPU(p, scalea, i + 1, j, *lnext);

            Timer::beg(Timer::MEMCPY);
            GPU_MEMCPY( (*lnext)(i + 1), (*lnext)(i + 1, 'd'),
                    p.b * (*lnext).get_lda() * sizeof(__half), GPU_MEMCPY_DEVICE_TO_HOST) ;
            Timer::end(Timer::MEMCPY);

            // broadcast_left_panels_block(b, *lnext, i+1, nprow, grid);

            // do GEMM of U
            Timer::beg(Timer::GEMM_UPDATE);
            if(p.npcol - j - 1 > 0) {
                GPUBLAS_SGEMM_EX(p.cuHandle,
                              GPUBLAS_OP_N,
                              GPUBLAS_OP_T,
                              b,
                              p.b * (p.npcol - j - 1),
                              b,
                              &nalf,
                              (void *)(*lprev)(i, 'd'),
                              GPU_R_16F,
                              (*lprev).get_lda(),
                              (void *)(*rprev)(j + 1, 'd'),
                              GPU_R_16F,
                              (*rprev).get_lda(),
                              beta,
                              p(i, j + 1, 'd'),
                              GPU_R_32F,
                              p.lda);
            }

            if( p.sync > 0 ) GPU_DEVICE_SYNCHRONIZE();

            Timer::end(Timer::GEMM_UPDATE, false, 2ull * p.b * p.b * p.b * (npcol - j - 1));

            update_right_panels_wGPU(d_lu, lda, p, i, j + 1);

            convert_right_panels_wGPU(p, scaleb, i, j + 1, *rnext);

            Timer::beg(Timer::MEMCPY);
            GPU_MEMCPY((*rnext)(j + 1),
                       (*rnext)(j + 1, 'd'),
                       p.b * (*rnext).get_lda() * sizeof(__half),
                       GPU_MEMCPY_DEVICE_TO_HOST);
            Timer::end(Timer::MEMCPY);

            ++schur_row;
            ++schur_col;

            Timer::beg(Timer::GEMM_UPDATE);

            if(p.nprow - schur_row > 0 && p.npcol - schur_col > 0) {
                GPUBLAS_SGEMM_EX(p.cuHandle,
                              GPUBLAS_OP_N,
                              GPUBLAS_OP_T,
                              p.b * (p.nprow - schur_row),
                              p.b * (p.npcol - schur_col),
                              b,
                              &nalf,
                              (void *)(*lprev)(schur_row, 'd'),
                              GPU_R_16F,
                              (*lprev).get_lda(),
                              (void *)(*rprev)(schur_col, 'd'),
                              GPU_R_16F,
                              (*rprev).get_lda(),
                              beta,
                              p(schur_row, schur_col, 'd'),
                              GPU_R_32F,
                              p.lda);
            }

            if( p.sync > 0 ) GPU_DEVICE_SYNCHRONIZE();

            Timer::end(
                Timer::GEMM_UPDATE, false, 2ull * p.b * p.b * p.b * (p.nprow - schur_row) * (p.npcol - schur_col));

            // MPI_Barrier(MPI_COMM_WORLD);
	    
            if(p.comm == 1) {
                broadcast_left_panels_block(b, *lnext, i + 1, nprow, grid);
                broadcast_right_panels_block(b, *rnext, j + 1, npcol, grid);
            } else if(p.comm<5) {
                rrp_flag = rlp_flag = myB_KEEPTEST;
                do {
                    if(rlp_flag == myB_KEEPTEST)
                        rlp_flag = myBcast(
                            (*lnext)(i + 1), lnext->get_lda() * b, grid.col, grid.hcomm, rootcol, k, grid.ncol, p.comm);
                    if(rrp_flag == myB_KEEPTEST)
                        rrp_flag = myBcast(
                            (*rnext)(j + 1), rnext->get_lda() * b, grid.row, grid.vcomm, rootrow, k, grid.nrow, p.comm);
                } while(rrp_flag == myB_KEEPTEST || rlp_flag == myB_KEEPTEST);
            } else{
	       		rrp_flag = rlp_flag = myB_KEEPTEST;
				int firsttime=1;
                do {
                    if(rlp_flag == myB_KEEPTEST){
                        rlp_flag = myBcast_irecv(
                            (*lnext)(i + 1), lnext->get_lda() * b, grid.col, grid.hcomm, rootcol, k, grid.ncol, p.comm, &lrequest,firsttime);
                    }
                    if(rrp_flag == myB_KEEPTEST){
                        rrp_flag = myBcast_irecv(
                            (*rnext)(j + 1), rnext->get_lda() * b, grid.row, grid.vcomm, rootrow, k, grid.nrow, p.comm, &rrequest,firsttime);
                    }
                    firsttime =0;
				} while(rrp_flag == myB_KEEPTEST || rlp_flag == myB_KEEPTEST);

			}

            ++i;
            ++j;
        } else if(rootrow == grid.row) {
            // GEMM
            Timer::beg(Timer::GEMM_UPDATE);
            if(p.npcol - j > 0) {
                GPUBLAS_SGEMM_EX(p.cuHandle,
                              GPUBLAS_OP_N,
                              GPUBLAS_OP_T,
                              b,
                              p.b * (p.npcol - j),
                              b,
                              &nalf,
                              (void *)(*lprev)(i, 'd'),
                              GPU_R_16F,
                              (*lprev).get_lda(),
                              (void *)(*rprev)(j, 'd'),
                              GPU_R_16F,
                              (*rprev).get_lda(),
                              beta,
                              p(i, j, 'd'),
                              GPU_R_32F,
                              p.lda);
            }
            ++schur_row;

            if( p.sync > 0 ) GPU_DEVICE_SYNCHRONIZE();

            Timer::end(Timer::GEMM_UPDATE, false, 2ull * p.b * p.b * p.b * (p.npcol - j));

            if(p.comm == 1) {
                receive_pivl_block(b, piv, ldpiv, rootcol, grid);
            } else if (p.comm <5){
                do {
                    rcpiv_flag = myBcast(piv, ldpiv * b, grid.col, grid.hcomm, rootcol, k, grid.ncol, p.comm);
                } while(rcpiv_flag == myB_KEEPTEST);
            } else {
			    int firsttime=1;
				do {
                    rcpiv_flag = myBcast_irecv(piv, ldpiv * b, grid.col, grid.hcomm, rootcol, k, grid.ncol, p.comm, &lrequest, firsttime);
                	firsttime = 0;
				} while(rcpiv_flag == myB_KEEPTEST);
			}

            Timer::beg(Timer::MEMCPY);
            GPU_MEMCPY(d_piv, piv, ldpiv * ldpiv * sizeof(FHigh), GPU_MEMCPY_HOST_TO_DEVICE);
            Timer::end(Timer::MEMCPY);

            update_right_panels_wGPU(d_piv, ldpiv, p, i, j);

            convert_right_panels_wGPU(p, scaleb, i, j, *rnext);

            Timer::beg(Timer::MEMCPY);
            GPU_MEMCPY(
                (*rnext)(j), (*rnext)(j, 'd'), p.b * (*rnext).get_lda() * sizeof(__half), GPU_MEMCPY_DEVICE_TO_HOST);
            Timer::end(Timer::MEMCPY);

            Timer::beg(Timer::GEMM_UPDATE);
            if(p.nprow - schur_row > 0 && p.npcol - schur_col > 0) {
                GPUBLAS_SGEMM_EX(p.cuHandle,
                              GPUBLAS_OP_N,
                              GPUBLAS_OP_T,
                              p.b * (p.nprow - schur_row),
                              p.b * (p.npcol - schur_col),
                              b,
                              &nalf,
                              (void *)(*lprev)(schur_row, 'd'),
                              GPU_R_16F,
                              (*lprev).get_lda(),
                              (void *)(*rprev)(schur_col, 'd'),
                              GPU_R_16F,
                              (*rprev).get_lda(),
                              beta,
                              p(schur_row, schur_col, 'd'),
                              GPU_R_32F,
                              p.lda);
            }

            if( p.sync > 0 ) GPU_DEVICE_SYNCHRONIZE();

            Timer::end(
                Timer::GEMM_UPDATE, false, 2ull * p.b * p.b * p.b * (p.nprow - schur_row) * (p.npcol - schur_col));

            // MPI_Barrier(MPI_COMM_WORLD);
            if(p.comm == 1) {
                receive_left_panels_block(b, *lnext, i + 1, nprow, rootcol, grid);
                broadcast_right_panels_block(b, *rnext, j, npcol, grid);
            } else if(p.comm <5) {
                lnext->set_start(i + 1);
                rrp_flag = rlp_flag = myB_KEEPTEST;
                do {
                    if(rlp_flag == myB_KEEPTEST)
                        rlp_flag = myBcast(
                            (*lnext)(i + 1), lnext->get_lda() * b, grid.col, grid.hcomm, rootcol, k, grid.ncol, p.comm);
                    if(rrp_flag == myB_KEEPTEST)
                        rrp_flag = myBcast(
                            (*rnext)(j), rnext->get_lda() * b, grid.row, grid.vcomm, rootrow, k, grid.nrow, p.comm);
                } while(rrp_flag == myB_KEEPTEST || rlp_flag == myB_KEEPTEST);
            } else{
				int firsttime = 1;
				lnext->set_start(i + 1);
                rrp_flag = rlp_flag = myB_KEEPTEST;
                do {
                    if(rlp_flag == myB_KEEPTEST){
                        rlp_flag = myBcast_irecv(
                            (*lnext)(i + 1), lnext->get_lda() * b, grid.col, grid.hcomm, rootcol, k, grid.ncol, p.comm, &lrequest, firsttime);
                    }
                    if(rrp_flag == myB_KEEPTEST){
                        rrp_flag = myBcast_irecv(
                            (*rnext)(j), rnext->get_lda() * b, grid.row, grid.vcomm, rootrow, k, grid.nrow, p.comm, &rrequest, firsttime);
                    }
                    firsttime=0;
				} while(rrp_flag == myB_KEEPTEST || rlp_flag == myB_KEEPTEST);
			}
            ++i;

        } else if(rootcol == grid.col) {
            Timer::beg(Timer::GEMM_UPDATE);
            if(p.nprow - i > 0) {
                GPUBLAS_SGEMM_EX(p.cuHandle,
                              GPUBLAS_OP_N,
                              GPUBLAS_OP_T,
                              b * (p.nprow - i),
                              b,
                              b,
                              &nalf,
                              (void *)(*lprev)(i, 'd'),
                              GPU_R_16F,
                              (*lprev).get_lda(),
                              (void *)(*rprev)(j, 'd'),
                              GPU_R_16F,
                              (*rprev).get_lda(),
                              beta,
                              p(i, j, 'd'),
                              GPU_R_32F,
                              p.lda);
            }
            ++schur_col;

            if( p.sync > 0 ) GPU_DEVICE_SYNCHRONIZE();

            Timer::end(Timer::GEMM_UPDATE, false, 2ull * p.b * p.b * p.b * (p.nprow - i));

            if(p.comm == 1) {
                receive_pivu_block(b, piv, ldpiv, rootrow, grid);
            } else if(p.comm <5){
                do {
                    rrpiv_flag = myBcast(piv, ldpiv * b, grid.row, grid.vcomm, rootrow, k, grid.nrow, p.comm);
                } while(rrpiv_flag == myB_KEEPTEST);
            }else{
				int firsttime = 1;
			    do {
                    rrpiv_flag = myBcast_irecv(piv, ldpiv * b, grid.row, grid.vcomm, rootrow, k, grid.nrow, p.comm, &lrequest, firsttime);
                	firsttime = 0;
				} while(rrpiv_flag == myB_KEEPTEST);
			}

            Timer::beg(Timer::MEMCPY);
            GPU_MEMCPY(d_piv, piv, ldpiv * ldpiv * sizeof(FHigh), GPU_MEMCPY_HOST_TO_DEVICE);
            Timer::end(Timer::MEMCPY);

            update_left_panels_wGPU(d_piv, ldpiv, p, i, j);

            convert_left_panels_wGPU(p, scalea, i, j, *lnext);

            Timer::beg(Timer::MEMCPY);
            GPU_MEMCPY(
                (*lnext)(i), (*lnext)(i, 'd'), p.b * (*lnext).get_lda() * sizeof(__half), GPU_MEMCPY_DEVICE_TO_HOST);
            Timer::end(Timer::MEMCPY);

            Timer::beg(Timer::GEMM_UPDATE);
            if(p.nprow - schur_row > 0 && p.npcol - schur_col > 0) {
                GPUBLAS_SGEMM_EX(p.cuHandle,
                              GPUBLAS_OP_N,
                              GPUBLAS_OP_T,
                              p.b * (p.nprow - schur_row),
                              p.b * (p.npcol - schur_col),
                              b,
                              &nalf,
                              (void *)(*lprev)(schur_row, 'd'),
                              GPU_R_16F,
                              (*lprev).get_lda(),
                              (void *)(*rprev)(schur_col, 'd'),
                              GPU_R_16F,
                              (*rprev).get_lda(),
                              beta,
                              p(schur_row, schur_col, 'd'),
                              GPU_R_32F,
                              p.lda);
            }

            if( p.sync > 0 ) GPU_DEVICE_SYNCHRONIZE();

            Timer::end(
                Timer::GEMM_UPDATE, false, 2ull * p.b * p.b * p.b * (p.nprow - schur_row) * (p.npcol - schur_col));

            if(p.comm == 1) {
                broadcast_left_panels_block(b, *lnext, i, nprow, grid);
                receive_right_panels_block(b, *rnext, j + 1, npcol, rootrow, grid);
            } else if(p.comm <5){
                rnext->set_start(j + 1);
                rrp_flag = rlp_flag = myB_KEEPTEST;
                do {
                    if(rlp_flag == myB_KEEPTEST)
                        rlp_flag = myBcast(
                            (*lnext)(i), lnext->get_lda() * b, grid.col, grid.hcomm, rootcol, k, grid.ncol, p.comm);
                    if(rrp_flag == myB_KEEPTEST)
                        rrp_flag = myBcast(
                            (*rnext)(j + 1), rnext->get_lda() * b, grid.row, grid.vcomm, rootrow, k, grid.nrow, p.comm);
                } while(rrp_flag == myB_KEEPTEST || rlp_flag == myB_KEEPTEST);
            }
			else{
				int firsttime = 1;
                rnext->set_start(j + 1);
                rrp_flag = rlp_flag = myB_KEEPTEST;
                do {
                    if(rlp_flag == myB_KEEPTEST)
                        rlp_flag = myBcast_irecv(
                            (*lnext)(i), lnext->get_lda() * b, grid.col, grid.hcomm, rootcol, k, grid.ncol, p.comm,&lrequest,firsttime);
                    if(rrp_flag == myB_KEEPTEST)
                        rrp_flag = myBcast_irecv(
                            (*rnext)(j + 1), rnext->get_lda() * b, grid.row, grid.vcomm, rootrow, k, grid.nrow, p.comm, &rrequest, firsttime);
                	firsttime =0;
				} while(rrp_flag == myB_KEEPTEST || rlp_flag == myB_KEEPTEST);
			}
			++j;

        } else {
            Timer::beg(Timer::GEMM_UPDATE);
            if(p.nprow - schur_row > 0 && p.npcol - schur_col > 0) {
                GPUBLAS_SGEMM_EX(p.cuHandle,
                              GPUBLAS_OP_N,
                              GPUBLAS_OP_T,
                              p.b * (p.nprow - schur_row),
                              p.b * (p.npcol - schur_col),
                              b,
                              &nalf,
                              (void *)(*lprev)(schur_row, 'd'),
                              GPU_R_16F,
                              (*lprev).get_lda(),
                              (void *)(*rprev)(schur_col, 'd'),
                              GPU_R_16F,
                              (*rprev).get_lda(),
                              beta,
                              p(schur_row, schur_col, 'd'),
                              GPU_R_32F,
                              p.lda);
            }

            if( p.sync > 0 ) GPU_DEVICE_SYNCHRONIZE();

            Timer::end(
                Timer::GEMM_UPDATE, false, 2ull * p.b * p.b * p.b * (p.nprow - schur_row) * (p.npcol - schur_col));

            // MPI_Barrier(MPI_COMM_WORLD);
	    
            if(p.comm == 1) {
                receive_left_panels_block(b, *lnext, i, nprow, rootcol, grid);
                receive_right_panels_block(b, *rnext, j, npcol, rootrow, grid);
            } else if (p.comm <5){
                rnext->set_start(j);
                lnext->set_start(i);
                rrp_flag = rlp_flag = myB_KEEPTEST;
                do {
                    if(rlp_flag == myB_KEEPTEST)
                        rlp_flag = myBcast(
                            (*lnext)(i), lnext->get_lda() * b, grid.col, grid.hcomm, rootcol, k, grid.ncol, p.comm);
                    if(rrp_flag == myB_KEEPTEST)
                        rrp_flag = myBcast(
                            (*rnext)(j), rnext->get_lda() * b, grid.row, grid.vcomm, rootrow, k, grid.nrow, p.comm);
                } while(rrp_flag == myB_KEEPTEST || rlp_flag == myB_KEEPTEST);
            }else{
				int firsttime = 1;
			    rnext->set_start(j);
                lnext->set_start(i);
                rrp_flag = rlp_flag = myB_KEEPTEST;
                do {
                    if(rlp_flag == myB_KEEPTEST)
                        rlp_flag = myBcast_irecv(
                            (*lnext)(i), lnext->get_lda() * b, grid.col, grid.hcomm, rootcol, k, grid.ncol, p.comm, &lrequest, firsttime);
                    if(rrp_flag == myB_KEEPTEST)
                        rrp_flag = myBcast_irecv(
                            (*rnext)(j), rnext->get_lda() * b, grid.row, grid.vcomm, rootrow, k, grid.nrow, p.comm, &rrequest, firsttime);
                	firsttime = 0;
				} while(rrp_flag == myB_KEEPTEST || rlp_flag == myB_KEEPTEST);
            }
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

        if(rootcol != grid.col) {
            Timer::beg(Timer::MEMCPY);
            GPU_MEMCPY( (*lprev)(i, 'd'), (*lprev)(i), p.b * (*lprev).get_lda() * sizeof(__half), GPU_MEMCPY_HOST_TO_DEVICE );
            Timer::end(Timer::MEMCPY);
        }
        if(rootrow != grid.row) {
            Timer::beg(Timer::MEMCPY);
            GPU_MEMCPY( (*rprev)(j, 'd'), (*rprev)(j), p.b * (*rprev).get_lda() * sizeof(__half), GPU_MEMCPY_HOST_TO_DEVICE );
            Timer::end(Timer::MEMCPY);
        }
    }
}

#endif
