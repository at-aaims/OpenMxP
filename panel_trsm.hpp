#ifndef PANEL_TRSM_HPP
#define PANEL_TRSM_HPP
#include "grid.hpp"
#include "panel.hpp"
#include "timer.hpp"
#include "device_macros.h"

#include <mpi.h>
#include <omp.h>
#include <cmath>


 
template <typename FHigh>
void panel_trsmL_lookahead(Panels<FHigh> &p,
                                Panels<FHigh>p2,
                                LRPanels<FHigh> lrpanels[4], 
                                FHigh *d_piv,
                                size_t ldpiv,
                                Grid &grid)
{
    int const nb = p.nblocks;
    int const b = p.b;
    int const n = nb * b;
    size_t const lda = p.lda;
    int const nprowA = p.nprow;
    int const npcolA = p.npcol;
    int const nprowB = p2.nprow;
    int const npcolB = p2.npcol;
    int kend =  nb;
    
    FHigh * cpu_piv, *cpu_nextB;
    checkGPU( GPU_MALLOC_HOST( ( void ** )&cpu_piv, b * b * sizeof( FHigh ) ),
             "<FATAL:%s> %s\n", hostRank, "Allocating host[cpu] piv" );
    checkGPU( GPU_MALLOC_HOST( ( void ** )&cpu_nextB, b * b * p2.npcol * sizeof( FHigh ) ),
             "<FATAL:%s> %s\n", hostRank, "Allocating host[cpu] piv" );
    
    GPU_ERROR_T cudaStat1 = GPU_SUCCESS;

    int divider = ceil(nb*0.01);
    const FHigh alf = 1.0f;
    const FHigh nalf = -1.0f;
    const FHigh bet = 1.0f;
    const FHigh *alpha = &alf;
    const FHigh *beta = &bet;
    int dogemm = 0;
    // Update Bprev parameters
    int schur_row = 0;
    int schur_col = 0;
    LRPanels<FHigh> *Aprev = &lrpanels[0], *Anext = &lrpanels[1], *Bprev = &lrpanels[2],*Bnext = &lrpanels[3];
    Bprev->lda = b * p2.npcol;
    Bprev->set_start(0);
    Bnext->lda = b * p2.npcol;
    Bnext->set_start(0);

    //FHigh *buf = d_piv + ldpiv * b;
    
    for(int k = 0; k < kend; ++k) {
        int const rootcol = k % grid.ncol;
        int const rootrow = k % grid.nrow;
        int i = k / grid.nrow + (rootrow > grid.row ? 1 : 0);
        int j = k / grid.ncol + (rootcol > grid.col ? 1 : 0);
        int solvedRow;
        int solvedColumn;

        // Sending out the dependency from left to right panel
        if(rootrow == grid.row && rootcol == grid.col){
            GPU_MEMCPY_2D(d_piv,
                         ldpiv * sizeof(FHigh),
                         p(i, j, 'd'),
                         p.d_lda * sizeof(FHigh),
                         b * sizeof(FHigh),
                         b,
                         GPU_MEMCPY_DEVICE_TO_DEVICE);
        }
       
        GPU_MEMCPY(cpu_piv, d_piv, b *b *sizeof(FHigh),GPU_MEMCPY_DEVICE_TO_HOST);
        printf("Pivot\n");
        print_matrix(cpu_piv,b,b*b);
        

        // Preparing 2 panels
        if(rootrow == grid.row){
            MPI_Bcast(d_piv, ldpiv * b, T2MPI<FHigh>::type, rootcol, grid.hcomm);
           
            printf("%d before\n",i);
            GPU_MEMCPY(p2(0, 0), p2(0, 0, 'd'), b * p2.npcol* p.nprow*b *sizeof(FHigh), GPU_MEMCPY_DEVICE_TO_HOST);
            print_matrix(p2(0,0),p.lda,p.lda*p.lda);
            fflush(stdout);
            
            if(dogemm == 1 &&  p.nprow - schur_row > 0)
            {
                GPUBLAS_SGEMM_EX(p.cuHandle,
                    GPUBLAS_OP_N,
                    GPUBLAS_OP_T,
                    p.b , //to prevent previous iteration and current iteration has different beginB
                    p.b * p2.npcol,
                    b,
                    &nalf,
                    (void *)(*Aprev)(schur_row, 'd'), GPU_R_32F, Aprev->get_lda(),
                    (void *)(*Bprev)(0, 'd'), GPU_R_32F, Bprev->get_lda(),
                    beta,
                    p2(schur_row, 0, 'd'),GPU_R_32F,p.lda);
                
                printf("previous B\n");
                GPU_MEMCPY(cpu_nextB, (*Bprev)(0, 'd'), b * Bprev->get_lda() *sizeof(FHigh), GPU_MEMCPY_DEVICE_TO_HOST);
                print_matrix(cpu_nextB,Bprev->get_lda(),b * Bprev->get_lda());
                
                printf("update\n");
                GPU_MEMCPY(p2(0, 0), p2(0, 0, 'd'), b * p2.npcol* p.nprow*b *sizeof(FHigh), GPU_MEMCPY_DEVICE_TO_HOST);
                print_matrix(p2(0,0),p.lda,p.lda*p.lda);
            }
            checkGPUblas(GPUBLAS_STRSM(p.cuHandle, GPUBLAS_SIDE_LEFT, GPUBLAS_FILL_MODE_LOWER, 
                                        GPUBLAS_OP_N, GPUBLAS_DIAG_UNIT, 
                                        p.b, p2.npcol* p.b, alpha,
                                        d_piv, ldpiv,
                                        p2(i,0,'d'), p2.d_lda));
           
            printf("%d???\n",i);
            GPU_MEMCPY(p2(0, 0), p2(0, 0, 'd'), b * p2.npcol* p.nprow*b *sizeof(FHigh), GPU_MEMCPY_DEVICE_TO_HOST);
            print_matrix(p2(0,0),p.lda,p.lda*p.lda);
            fflush(stdout);
            
            GPU_DEVICE_SYNCHRONIZE();

            Bnext->set_start(0);
            right_trans(p2(i,0,'d'), p.b, p.lda, (*Bnext)(0,'d'), Bnext->get_lda());
            
            ++i;
            ++schur_row;
            
            printf("nextB\n",i);
            GPU_MEMCPY(cpu_nextB, (*Bnext)(0,'d'), b * Bnext->get_lda() *sizeof(FHigh), GPU_MEMCPY_DEVICE_TO_HOST);
            print_matrix(cpu_nextB,  Bnext->get_lda() ,b * Bnext->get_lda());
            
        }

        if(rootcol == grid.col){
            Anext->set_start(i);
            GPU_MEMCPY_2D((*Anext)(i,'d'), Anext->get_lda()* sizeof(FHigh), p(i,j,'d'), p.lda* sizeof(FHigh), 
                            Anext->get_lda()* sizeof(FHigh), p.b, GPU_MEMCPY_DEVICE_TO_DEVICE);
        }

        // All panel ready to send
        GPU_DEVICE_SYNCHRONIZE();

        // Luanch gemm before the MPI
        if(dogemm == 1)
        {
            if(p.nprow - schur_row > 0)
            {
                GPUBLAS_SGEMM_EX(p.cuHandle,
                    GPUBLAS_OP_N,
                    GPUBLAS_OP_T,
                    p.b * (p.nprow-schur_row),
                    p.b * p2.npcol,
                    b,
                    &nalf,
                    (void *)(*Aprev)(schur_row, 'd'), GPU_R_32F, Aprev->get_lda(),
                    (void *)(*Bprev)(0, 'd'), GPU_R_32F, Bprev->get_lda(),
                    beta,
                    p2(schur_row, 0, 'd'),GPU_R_32F,p.lda);
                 printf("update\n");
                GPU_MEMCPY(p2(0, 0), p2(0, 0, 'd'), b * p2.npcol* p.nprow*b *sizeof(FHigh), GPU_MEMCPY_DEVICE_TO_HOST);
                print_matrix(p2(0,0),p.lda,p.lda*p.lda);
            }
           
        }

        Anext->set_start(i);
        Bnext->set_start(0);
        MPI_Bcast(const_cast<FHigh *> ((*Anext)(i,'d')), Anext->get_lda() * b, T2MPI<FHigh>::type, rootcol, grid.hcomm);
        MPI_Bcast(const_cast<FHigh *> ((*Bnext)(0,'d')), Bnext->get_lda() * b, T2MPI<FHigh>::type, rootrow, grid.vcomm);

        GPU_DEVICE_SYNCHRONIZE();
        
        dogemm = 1;
        LRPanels<FHigh> *t;
        t = Aprev;
        Aprev = Anext;
        Anext = t;
        t = Bprev;
        Bprev = Bnext;
        Bnext = t;
        schur_row = i;
        schur_col = j;
    }
}

template <typename FHigh>
void panel_trsmU_lookahead(Panels<FHigh> &p,
                                Panels<FHigh>&p2,
                                LRPanels<FHigh> lrpanels[4], 
                                FHigh *d_piv,
                                size_t ldpiv,
                                Grid &grid)
{
    int const nb = p.nblocks;
    int const b = p.b;
    int const n = nb * b;
    size_t const lda = p.lda;
    int const nprowA = p.nprow;
    int const npcolA = p.npcol;
    int const nprowB = p2.nprow;
    int const npcolB = p2.npcol;
    int kend =  nb;
    
    
       
    FHigh * cpu_piv, *cpu_nextB, * cpu_nextA;
    checkGPU( GPU_MALLOC_HOST( ( void ** )&cpu_piv, b * b * sizeof( FHigh ) ),
             "<FATAL:%s> %s\n", hostRank, "Allocating host[cpu] piv" );
    checkGPU( GPU_MALLOC_HOST( ( void ** )&cpu_nextB, b * b * p2.npcol * sizeof( FHigh ) ),
             "<FATAL:%s> %s\n", hostRank, "Allocating host[cpu] piv" );
    checkGPU( GPU_MALLOC_HOST( ( void ** )&cpu_nextA, b * b * p.nprow * sizeof( FHigh ) ),
             "<FATAL:%s> %s\n", hostRank, "Allocating host[cpu] piv" );
    
    
    
    GPU_ERROR_T cudaStat1 = GPU_SUCCESS;

    int divider = ceil(nb*0.01);
    const FHigh alf = 1.0f;
    const FHigh nalf = -1.0f;
    const FHigh bet = 1.0f;
    const FHigh *alpha = &alf;
    const FHigh *beta = &bet;
    int dogemm = 0;
    // Update Bprev parameters
    int schur_row = (kend-1) / grid.nrow;// + (((kend-1) % grid.nrow)> grid.row ? 1 : 0);
    int schur_col = (kend-1) / grid.ncol + (((kend-1) % grid.ncol) > grid.col ? 1 : 0);
    LRPanels<FHigh> *Aprev = &lrpanels[0], *Anext = &lrpanels[1], *Bprev = &lrpanels[2],*Bnext = &lrpanels[3];
    Bprev->lda = p2.npcol*b;
    Bprev->set_start(0);
    Bnext->lda = p2.npcol*b;
    Bnext->set_start(0);
    
    char filebuf[256];
    sprintf(filebuf, "nodeOutput_%d",grank);
    FILE* outfile = fopen(filebuf,"w");

    for(int k = kend-1; k >= 0; --k) {
        int const rootcol = k % grid.ncol;
        int const rootrow = k % grid.nrow;
        int i = k / grid.nrow  + (rootrow < grid.row ? -1 : 0);        
        int j = k / grid.ncol + (rootcol > grid.col ? 1 : 0);
        int offset = 0;

        fprintf(outfile," ROW COl : %d %d \n", i,j);
        // Sending out the dependency from left to right panel
        if(rootrow == grid.row && rootcol == grid.col){
            GPU_MEMCPY_2D(d_piv,
                         ldpiv * sizeof(FHigh),
                         p(i, j, 'd'),
                         p.d_lda * sizeof(FHigh),
                         b * sizeof(FHigh),
                         b,
                         GPU_MEMCPY_DEVICE_TO_DEVICE);
            GPU_MEMCPY(cpu_piv, d_piv, b *b *sizeof(FHigh),GPU_MEMCPY_DEVICE_TO_HOST);
            fprintf(outfile,"Pivot\n");
            print_matrix(cpu_piv,b,b*b,outfile);
            MPI_Bcast(d_piv, ldpiv * b, T2MPI<FHigh>::type, rootcol, grid.hcomm);
            
            if(dogemm == 1 &&  Aprev->get_lda() > 0)
            {
                GPUBLAS_SGEMM_EX(p.cuHandle,
                    GPUBLAS_OP_N,
                    GPUBLAS_OP_T,
                    p.b , //to prevent previous iteration and current iteration has different beginB
                    p.b * p2.npcol,
                    b,
                    &nalf,
                    (void *)(*Aprev)(schur_row, 'd'), GPU_R_32F, Aprev->get_lda(),
                    (void *)(*Bprev)(0, 'd'), GPU_R_32F, Bprev->get_lda(),
                    beta,
                    p2(schur_row, 0, 'd'),GPU_R_32F,p.lda);

                fprintf(outfile,"diagonal previous A, %d %d\n",k, Aprev->get_lda());
                GPU_MEMCPY(cpu_nextA, (*Aprev)(0,'d'), b * Aprev->get_lda() *sizeof(FHigh), GPU_MEMCPY_DEVICE_TO_HOST);
                print_matrix(cpu_nextA,  Aprev->get_lda() ,b * Aprev->get_lda(),outfile);
           
                
                fprintf(outfile,"diagonal previous B %d\n",k);
                GPU_MEMCPY(cpu_nextB, (*Bprev)(0, 'd'), b * Bprev->get_lda() *sizeof(FHigh), GPU_MEMCPY_DEVICE_TO_HOST);
                print_matrix(cpu_nextB,Bprev->get_lda(),b * Bprev->get_lda(),outfile);
            
                
                fprintf(outfile,"diagonal update, %d %d\n",k, schur_row);
                GPU_MEMCPY(p2(0, 0), p2(0, 0, 'd'), b * p2.npcol* p.nprow*b *sizeof(FHigh), GPU_MEMCPY_DEVICE_TO_HOST);
                print_matrix(p2(0,0),p.lda,p.lda*p.lda,outfile);
            
            }
            checkGPUblas(GPUBLAS_STRSM(p.cuHandle, GPUBLAS_SIDE_LEFT, GPUBLAS_FILL_MODE_UPPER, 
                                        GPUBLAS_OP_N, GPUBLAS_DIAG_NON_UNIT, 
                                        p.b, p2.npcol* p.b, alpha,
                                        d_piv, ldpiv,
                                        p2(i,0,'d'), p2.d_lda));
            
            fprintf(outfile, "diagonal after pivot solve %d???\n",k);
            GPU_MEMCPY(p2(0, 0), p2(0, 0, 'd'), b * p2.npcol* p.nprow*b *sizeof(FHigh), GPU_MEMCPY_DEVICE_TO_HOST);
            print_matrix(p2(0,0),p.lda,p.lda*p.lda,outfile);
            fflush(stdout);
            
            GPU_DEVICE_SYNCHRONIZE();
            Bnext->set_start(0);
            right_trans(p2(i,0,'d'), p.b, p.lda, (*Bnext)(0,'d'), Bnext->get_lda());
            
            fprintf(outfile,"diagonal nextB %d\n",k);
            GPU_MEMCPY(cpu_nextB, (*Bnext)(0,'d'), b * Bnext->get_lda() *sizeof(FHigh), GPU_MEMCPY_DEVICE_TO_HOST);
            print_matrix(cpu_nextB,  Bnext->get_lda() ,b * Bnext->get_lda(),outfile);
            
            --i;
            --schur_row;
            offset = b;
           
            Anext->set_start(0);
            Anext->lda = (i+1)*b;

            GPU_MEMCPY_2D((*Anext)(0,'d'), Anext->get_lda()* sizeof(FHigh), p(0,j,'d'), p.lda* sizeof(FHigh), 
                     Anext->get_lda()* sizeof(FHigh), p.b, GPU_MEMCPY_DEVICE_TO_DEVICE);
                
            if(Anext->get_lda() > 0)
            {
                fprintf(outfile, "diagonal nextA %d %d\n",k, Anext->get_lda());
                GPU_MEMCPY(cpu_nextA, (*Anext)(0,'d'), b * Anext->get_lda() *sizeof(FHigh), GPU_MEMCPY_DEVICE_TO_HOST);
                print_matrix(cpu_nextA,  Anext->get_lda() ,b * Anext->get_lda(),outfile);
            }
        }

        // Preparing 2 panels
        else if(rootrow == grid.row){
            MPI_Bcast(d_piv, ldpiv * b, T2MPI<FHigh>::type, rootcol, grid.hcomm);
            if(dogemm == 1 &&  p.nprow - schur_row > 0)
            {
                GPUBLAS_SGEMM_EX(p.cuHandle,
                    GPUBLAS_OP_N,
                    GPUBLAS_OP_T,
                    p.b , //to prevent previous iteration and current iteration has different beginB
                    p.b * p2.npcol,
                    b,
                    &nalf,
                    (void *)(*Aprev)(schur_row, 'd'), GPU_R_32F, Aprev->get_lda(),
                    (void *)(*Bprev)(0, 'd'), GPU_R_32F, Bprev->get_lda(),
                    beta,
                    p2(schur_row, 0, 'd'),GPU_R_32F,p.lda);

                fprintf(outfile, "row previous A, %d %d\n",k, Aprev->get_lda());
                GPU_MEMCPY(cpu_nextA, (*Aprev)(0,'d'), b * Aprev->get_lda() *sizeof(FHigh), GPU_MEMCPY_DEVICE_TO_HOST);
                print_matrix(cpu_nextA,  Aprev->get_lda() ,b * Aprev->get_lda(),outfile);
           
                fprintf(outfile, "row previous B, %d\n",k);
                GPU_MEMCPY(cpu_nextB, (*Bprev)(0, 'd'), b * Bprev->get_lda() *sizeof(FHigh), GPU_MEMCPY_DEVICE_TO_HOST);
                print_matrix(cpu_nextB,Bprev->get_lda(),b * Bprev->get_lda(),outfile);
            
                
                fprintf(outfile, "row update, %d %d\n",k, schur_row);
                GPU_MEMCPY(p2(0, 0), p2(0, 0, 'd'), b * p2.npcol* p.nprow*b *sizeof(FHigh), GPU_MEMCPY_DEVICE_TO_HOST);
                print_matrix(p2(0,0),p.lda,p.lda*p.lda,outfile);
            
            }
            checkGPUblas(GPUBLAS_STRSM(p.cuHandle, GPUBLAS_SIDE_LEFT, GPUBLAS_FILL_MODE_UPPER, 
                                        GPUBLAS_OP_N, GPUBLAS_DIAG_NON_UNIT, 
                                        p.b, p2.npcol* p.b, alpha,
                                        d_piv, ldpiv,
                                        p2(i,0,'d'), p2.d_lda));
            
            fprintf(outfile, "row after solve %d???\n",k);
            GPU_MEMCPY(p2(0, 0), p2(0, 0, 'd'), b * p2.npcol* p.nprow*b *sizeof(FHigh), GPU_MEMCPY_DEVICE_TO_HOST);
            print_matrix(p2(0,0),p.lda,p.lda*p.lda,outfile);
            fflush(stdout);
            
            GPU_DEVICE_SYNCHRONIZE();
            Bnext->set_start(0);
            right_trans(p2(i,0,'d'), p.b, p.lda, (*Bnext)(0,'d'), Bnext->get_lda());
            
            fprintf(outfile, "row nextB %d\n",k);
            GPU_MEMCPY(cpu_nextB, (*Bnext)(0,'d'), b * Bnext->get_lda() *sizeof(FHigh), GPU_MEMCPY_DEVICE_TO_HOST);
            print_matrix(cpu_nextB,  Bnext->get_lda() ,b * Bnext->get_lda(),outfile);
            
            --i;
            --schur_row;
            offset = b;

            Anext->set_start(0);
            Bnext->set_start(0);
            Anext->lda = (i+1)*b;
            
        }
        else if(rootcol == grid.col){
            Anext->set_start(0);
            Bnext->set_start(0);
            Anext->lda = (i+1)*b;
            GPU_MEMCPY_2D((*Anext)(0,'d'), Anext->get_lda()* sizeof(FHigh), p(0,j,'d'), p.lda* sizeof(FHigh), 
                    Anext->get_lda()* sizeof(FHigh), p.b, GPU_MEMCPY_DEVICE_TO_DEVICE);
            if(Anext->get_lda() >0)
            {
                fprintf(outfile, "column nextA %d %d\n",k, Anext->get_lda());
                GPU_MEMCPY(cpu_nextA, (*Anext)(0,'d'), b * Anext->get_lda() *sizeof(FHigh), GPU_MEMCPY_DEVICE_TO_HOST);
                print_matrix(cpu_nextA,  Anext->get_lda() ,b * Anext->get_lda(),outfile);
            }
        }
        else
        {
            Anext->set_start(0);
            Bnext->set_start(0);
            Anext->lda = (i+1)*b;
        }
        // All panel ready to send
        GPU_DEVICE_SYNCHRONIZE();
              // Luanch gemm before the MPI
        
        fprintf(outfile, "here\n");
        fflush(outfile);
        if(dogemm == 1 && Aprev->get_lda() > 0)
        {
            if(Aprev->get_lda()-offset > 0)
            {
                GPUBLAS_SGEMM_EX(p.cuHandle,
                    GPUBLAS_OP_N,
                    GPUBLAS_OP_T,
                    Aprev->get_lda()-offset,
                    p.b * p2.npcol,
                    b,
                    &nalf,
                    (void *)(*Aprev)(0, 'd'), GPU_R_32F, Aprev->get_lda(),
                    (void *)(*Bprev)(0, 'd'), GPU_R_32F, Bprev->get_lda(),
                    beta,
                    p2(0, 0, 'd'),GPU_R_32F,p.lda);
            }
        }
      

        MPI_Bcast(const_cast<FHigh *> ((*Anext)(0,'d')), Anext->get_lda() * b, T2MPI<FHigh>::type, rootcol, grid.hcomm);
        MPI_Bcast(const_cast<FHigh *> ((*Bnext)(0,'d')), Bnext->get_lda() * b, T2MPI<FHigh>::type, rootrow, grid.vcomm);

        fprintf(outfile,"end iteration %d ???\n",k);
        GPU_MEMCPY(p2(0, 0), p2(0, 0, 'd'), b * p2.npcol* p.nprow*b *sizeof(FHigh), GPU_MEMCPY_DEVICE_TO_HOST);
        print_matrix(p2(0,0),p.lda,p.lda*p.lda,outfile);
        fflush(stdout);
        
        GPU_DEVICE_SYNCHRONIZE();
        
        dogemm = 1;
        LRPanels<FHigh> *t;
        t = Aprev;
        Aprev = Anext;
        Anext = t;
        t = Bprev;
        Bprev = Bnext;
        Bnext = t;
        schur_row = i;
        schur_col = j;
    }
    fprintf(outfile,"exit\n");
    GPU_MEMCPY(p2(0, 0), p2(0, 0, 'd'), b * p2.npcol* p.nprow*b *sizeof(FHigh), GPU_MEMCPY_DEVICE_TO_HOST);
    print_matrix(p2(0,0),p.lda,p.lda*p.lda,outfile);
    fflush(stdout);
    
    
}


template <typename FHigh>
void panel_trsmU_no_lookahead(Panels<FHigh> &p,
                                Panels<FHigh>&p2,
                                LRPanels<FHigh> lrpanels[4], 
                                FHigh *d_piv,
                                size_t ldpiv,
                                Grid &grid)
{
    int const nb = p.nblocks;
    int const b = p.b;
    int const n = nb * b;
    size_t const lda = p.lda;
    int const nprowA = p.nprow;
    int const npcolA = p.npcol;
    int const nprowB = p2.nprow;
    int const npcolB = p2.npcol;
    int kend =  nb;
    
    char filebuf[256];
    sprintf(filebuf, "nodeOutput_%d",grank);
    FILE* outfile = fopen(filebuf,"w");

       
    FHigh * cpu_piv, *cpu_nextB, * cpu_nextA;
    checkGPU( GPU_MALLOC_HOST( ( void ** )&cpu_piv, b * b * sizeof( FHigh ) ),
             "<FATAL:%s> %s\n", hostRank, "Allocating host[cpu] piv" );
    checkGPU( GPU_MALLOC_HOST( ( void ** )&cpu_nextB, b * b * p2.npcol * sizeof( FHigh ) ),
             "<FATAL:%s> %s\n", hostRank, "Allocating host[cpu] piv" );
    checkGPU( GPU_MALLOC_HOST( ( void ** )&cpu_nextA, b * b * p.nprow * sizeof( FHigh ) ),
             "<FATAL:%s> %s\n", hostRank, "Allocating host[cpu] piv" );
    
    
    
    GPU_ERROR_T cudaStat1 = GPU_SUCCESS;

    int divider = ceil(nb*0.01);
    const FHigh alf = 1.0f;
    const FHigh nalf = -1.0f;
    const FHigh bet = 1.0f;
    const FHigh *alpha = &alf;
    const FHigh *beta = &bet;
    int dogemm = 0;
    // Update Bprev parameters
    int schur_row = (kend-1) / grid.nrow;// + (((kend-1) % grid.nrow)> grid.row ? 1 : 0);
    int schur_col = (kend-1) / grid.ncol + (((kend-1) % grid.ncol) > grid.col ? 1 : 0);
    LRPanels<FHigh> *Aprev = &lrpanels[0], *Anext = &lrpanels[1], *Bprev = &lrpanels[2],*Bnext = &lrpanels[3];
    Bprev->lda = p2.npcol*b;
    Bprev->set_start(0);
    Bnext->lda = p2.npcol*b;
    Bnext->set_start(0);

    for(int k = kend-1; k >= 0; --k) {
        int const rootcol = k % grid.ncol;
        int const rootrow = k % grid.nrow;
        int i = k / grid.nrow  + (rootrow < grid.row ? -1 : 0);        
        int j = k / grid.ncol + (rootcol > grid.col ? 1 : 0);
        int solvedRow;
        int solvedColumn;
        int offset = 0;

        fprintf(outfile, " ROW COl : %d %d \n", i,j);
        
        // Sending out the dependency from left to right panel
        if(rootrow == grid.row && rootcol == grid.col){
            GPU_MEMCPY_2D(d_piv,
                         ldpiv * sizeof(FHigh),
                         p(i, j, 'd'),
                         p.d_lda * sizeof(FHigh),
                         b * sizeof(FHigh),
                         b,
                         GPU_MEMCPY_DEVICE_TO_DEVICE);
            GPU_MEMCPY(cpu_piv, d_piv, b *b *sizeof(FHigh),GPU_MEMCPY_DEVICE_TO_HOST);
            fprintf(outfile, "Pivot\n");
            print_matrix(cpu_piv,b,b*b);
            MPI_Bcast(d_piv, ldpiv * b, T2MPI<FHigh>::type, rootcol, grid.hcomm);
            
            checkGPUblas(GPUBLAS_STRSM(p.cuHandle, GPUBLAS_SIDE_LEFT, GPUBLAS_FILL_MODE_UPPER, 
                                        GPUBLAS_OP_N, GPUBLAS_DIAG_NON_UNIT, 
                                        p.b, p2.npcol* p.b, alpha,
                                        d_piv, ldpiv,
                                        p2(i,0,'d'), p2.d_lda));
            
            
            fprintf(outfile, "diagonal %d???\n",i);
            GPU_MEMCPY(p2(0, 0), p2(0, 0, 'd'), b * p2.npcol* p.nprow*b *sizeof(FHigh), GPU_MEMCPY_DEVICE_TO_HOST);
            print_matrix(p2(0,0),p.lda,p.lda*p.lda,outfile);
            fflush(stdout);
            
            GPU_DEVICE_SYNCHRONIZE();
            Bnext->set_start(0);
            right_trans(p2(i,0,'d'), p.b, p.lda, (*Bnext)(0,'d'), Bnext->get_lda());
            
            fprintf(outfile,"diagonal nextB\n",i);
            GPU_MEMCPY(cpu_nextB, (*Bnext)(0,'d'), b * Bnext->get_lda() *sizeof(FHigh), GPU_MEMCPY_DEVICE_TO_HOST);
            print_matrix(cpu_nextB,  Bnext->get_lda() ,b * Bnext->get_lda(),outfile);
            
            --i;
            --schur_row;
            //offset = b;
           
            Anext->set_start(0);
            Anext->lda = (i+1)*b;
            GPU_MEMCPY_2D((*Anext)(0,'d'), Anext->get_lda()* sizeof(FHigh), p(0,j,'d'), p.lda* sizeof(FHigh), 
                    Anext->get_lda()* sizeof(FHigh), p.b, GPU_MEMCPY_DEVICE_TO_DEVICE);
            
            if(Anext->get_lda() >0){
                 fprintf(outfile,"diagonal nextA %d %d\n",i, Anext->get_lda());
                GPU_MEMCPY(cpu_nextA, (*Anext)(0,'d'), b * Anext->get_lda() *sizeof(FHigh), GPU_MEMCPY_DEVICE_TO_HOST);
                print_matrix(cpu_nextA,  Anext->get_lda() ,b * Anext->get_lda(),outfile);
            }
        }

        // Preparing 2 panels
        else if(rootrow == grid.row){
            MPI_Bcast(d_piv, ldpiv * b, T2MPI<FHigh>::type, rootcol, grid.hcomm);
        
            checkGPUblas(GPUBLAS_STRSM(p.cuHandle, GPUBLAS_SIDE_LEFT, GPUBLAS_FILL_MODE_UPPER, 
                                        GPUBLAS_OP_N, GPUBLAS_DIAG_NON_UNIT, 
                                        p.b, p2.npcol* p.b, alpha,
                                        d_piv, ldpiv,
                                        p2(i,0,'d'), p2.d_lda));
            
            fprintf(outfile,"row %d???\n",i);
            GPU_MEMCPY(p2(0, 0), p2(0, 0, 'd'), b * p2.npcol* p.nprow*b *sizeof(FHigh), GPU_MEMCPY_DEVICE_TO_HOST);
            print_matrix(p2(0,0),p.lda,p.lda*p.lda,outfile);
            fflush(stdout);
            
            GPU_DEVICE_SYNCHRONIZE();
            Bnext->set_start(0);
            right_trans(p2(i,0,'d'), p.b, p.lda, (*Bnext)(0,'d'), Bnext->get_lda());
            
             fprintf(outfile,"row nextB\n",i);
            GPU_MEMCPY(cpu_nextB, (*Bnext)(0,'d'), b * Bnext->get_lda() *sizeof(FHigh), GPU_MEMCPY_DEVICE_TO_HOST);
            print_matrix(cpu_nextB,  Bnext->get_lda() ,b * Bnext->get_lda(),outfile);
            
            --i;
            --schur_row;
             Anext->set_start(0);
            Bnext->set_start(0);
            Anext->lda = (i+1)*b;
            //offset = b;
            
        }
        else if(rootcol == grid.col){
            Anext->set_start(0);
            Anext->lda = (i+1)*b;
            GPU_MEMCPY_2D((*Anext)(0,'d'), Anext->get_lda()* sizeof(FHigh), p(0,j,'d'), p.lda* sizeof(FHigh), 
                    Anext->get_lda()* sizeof(FHigh), p.b, GPU_MEMCPY_DEVICE_TO_DEVICE);
            if(Anext->get_lda() >0){
                fprintf(outfile,"column nextA\n",i);
                GPU_MEMCPY(cpu_nextA, (*Anext)(0,'d'), b * Anext->get_lda() *sizeof(FHigh), GPU_MEMCPY_DEVICE_TO_HOST);
                print_matrix(cpu_nextA,  Anext->get_lda() ,b * Anext->get_lda(),outfile);
            }
        }
        else
        {
            Anext->set_start(0);
            Bnext->set_start(0);
            Anext->lda = (i+1)*b;
        }
        fprintf(outfile,"send/recive %d %d\n", Anext->get_lda() * b, Bnext->get_lda() * b);
        fflush(outfile);
        MPI_Bcast(const_cast<FHigh *> ((*Anext)(0,'d')), Anext->get_lda() * b, T2MPI<FHigh>::type, rootcol, grid.hcomm);
        MPI_Bcast(const_cast<FHigh *> ((*Bnext)(0,'d')), Bnext->get_lda() * b, T2MPI<FHigh>::type, rootrow, grid.vcomm);
        // All panel ready to send
        GPU_DEVICE_SYNCHRONIZE();
              // Luanch gemm before the MPI
        if(Anext->get_lda() > 0)
        {
                GPUBLAS_SGEMM_EX(p.cuHandle,
                    GPUBLAS_OP_N,
                    GPUBLAS_OP_T,
                    Anext->get_lda()-offset,
                    p.b * p2.npcol,
                    b,
                    &nalf,
                    (void *)(*Anext)(0, 'd'), GPU_R_32F, Anext->get_lda(),
                    (void *)(*Bnext)(0, 'd'), GPU_R_32F, Bnext->get_lda(),
                    beta,
                    p2(0, 0, 'd'),GPU_R_32F,p.lda);
        }
        
         fprintf(outfile,"end iteration???\n",i);
            GPU_MEMCPY(p2(0, 0), p2(0, 0, 'd'), b * p2.npcol* p.nprow*b *sizeof(FHigh), GPU_MEMCPY_DEVICE_TO_HOST);
            print_matrix(p2(0,0),p.lda,p.lda*p.lda,outfile);
            fflush(stdout);
        
        GPU_DEVICE_SYNCHRONIZE();
        
        dogemm = 1;
        LRPanels<FHigh> *t;
        t = Aprev;
        Aprev = Anext;
        Anext = t;
        t = Bprev;
        Bprev = Bnext;
        Bnext = t;
        schur_row = i;
        schur_col = j;
    }
     fprintf(outfile,"exit\n");
     GPU_MEMCPY(p2(0, 0), p2(0, 0, 'd'), b * p2.npcol* p.nprow*b *sizeof(FHigh), GPU_MEMCPY_DEVICE_TO_HOST);
     print_matrix(p2(0,0),p.lda,p.lda*p.lda,outfile);
      fflush(stdout);
}



template <typename FHigh>
void panel_trsvU_ORNL(Panels<FHigh> &p,
                                Panels<FHigh>&p2,
                                LRPanels<FHigh> lrpanels[4], 
                                FHigh *d_piv,
                                size_t ldpiv,
                                Grid &grid)
{
    int const nb = p.nblocks;
    int const b = p.b;
    int const n = nb * b;
    size_t const lda = p.lda;
    int const nprowA = p.nprow;
    int const npcolA = p.npcol;
    int const nprowB = p2.nprow;
    int const npcolB = p2.npcol;
    int kend =  nb;
    
    char filebuf[256];
    sprintf(filebuf, "nodeOutput_%d",grank);
    FILE* outfile = fopen(filebuf,"w");

       
    FHigh * cpu_piv, *cpu_nextB, * cpu_nextA;
    checkGPU( GPU_MALLOC_HOST( ( void ** )&cpu_piv, b * b * sizeof( FHigh ) ),
             "<FATAL:%s> %s\n", hostRank, "Allocating host[cpu] piv" );
    checkGPU( GPU_MALLOC_HOST( ( void ** )&cpu_nextB, b * b * p2.npcol * sizeof( FHigh ) ),
             "<FATAL:%s> %s\n", hostRank, "Allocating host[cpu] piv" );
    checkGPU( GPU_MALLOC_HOST( ( void ** )&cpu_nextA, b * b * p.nprow * sizeof( FHigh ) ),
             "<FATAL:%s> %s\n", hostRank, "Allocating host[cpu] piv" );
    
    
    
    GPU_ERROR_T cudaStat1 = GPU_SUCCESS;

    int divider = ceil(nb*0.01);
    const FHigh alf = 1.0f;
    const FHigh nalf = -1.0f;
    const FHigh bet = 1.0f;
    const FHigh *alpha = &alf;
    const FHigh *beta = &bet;
    int dogemm = 0;
    // Update Bprev parameters
    int schur_row = (kend-1) / grid.nrow;// + (((kend-1) % grid.nrow)> grid.row ? 1 : 0);
    int schur_col = (kend-1) / grid.ncol + (((kend-1) % grid.ncol) > grid.col ? 1 : 0);
    LRPanels<FHigh> *Aprev = &lrpanels[0], *Anext = &lrpanels[1], *Bprev = &lrpanels[2],*Bnext = &lrpanels[3];
    Bprev->lda = p2.npcol*b;
    Bprev->set_start(0);
    Bnext->lda = p2.npcol*b;
    Bnext->set_start(0);

    for(int k = kend-1; k >= 0; --k) {
        int const rootcol = k % grid.ncol;
        int const rootrow = k % grid.nrow;
        int i = k / grid.nrow  + (rootrow < grid.row ? -1 : 0);        
        int j = k / grid.ncol + (rootcol > grid.col ? 1 : 0);
        int solvedRow;
        int solvedColumn;
        int offset = 0;

        fprintf(outfile, " ROW COl : %d %d \n", i,j);
        
        // Sending out the dependency from left to right panel
        if(rootrow == grid.row && rootcol == grid.col){
            GPU_MEMCPY_2D(d_piv,
                         ldpiv * sizeof(FHigh),
                         p(i, j, 'd'),
                         p.d_lda * sizeof(FHigh),
                         b * sizeof(FHigh),
                         b,
                         GPU_MEMCPY_DEVICE_TO_DEVICE);
            GPU_MEMCPY(cpu_piv, d_piv, b *b *sizeof(FHigh),GPU_MEMCPY_DEVICE_TO_HOST);
            fprintf(outfile, "Pivot\n");
            print_matrix(cpu_piv,b,b*b);
            MPI_Bcast(d_piv, ldpiv * b, T2MPI<FHigh>::type, rootcol, grid.hcomm);
            
            checkGPUblas(GPUBLAS_STRSM(p.cuHandle, GPUBLAS_SIDE_LEFT, GPUBLAS_FILL_MODE_UPPER, 
                                        GPUBLAS_OP_N, GPUBLAS_DIAG_NON_UNIT, 
                                        p.b, p2.npcol* p.b, alpha,
                                        d_piv, ldpiv,
                                        p2(i,0,'d'), p2.d_lda));
            
            
            fprintf(outfile, "diagonal %d???\n",i);
            GPU_MEMCPY(p2(0, 0), p2(0, 0, 'd'), b * p2.npcol* p.nprow*b *sizeof(FHigh), GPU_MEMCPY_DEVICE_TO_HOST);
            print_matrix(p2(0,0),p.lda,p.lda*p.lda,outfile);
            fflush(stdout);
            
            GPU_DEVICE_SYNCHRONIZE();
            Bnext->set_start(0);
            right_trans(p2(i,0,'d'), p.b, p.lda, (*Bnext)(0,'d'), Bnext->get_lda());
            
            fprintf(outfile,"diagonal nextB\n",i);
            GPU_MEMCPY(cpu_nextB, (*Bnext)(0,'d'), b * Bnext->get_lda() *sizeof(FHigh), GPU_MEMCPY_DEVICE_TO_HOST);
            print_matrix(cpu_nextB,  Bnext->get_lda() ,b * Bnext->get_lda(),outfile);
            
            --i;
            --schur_row;
            //offset = b;
           
            Anext->set_start(0);
            Anext->lda = (i+1)*b;
            GPU_MEMCPY_2D((*Anext)(0,'d'), Anext->get_lda()* sizeof(FHigh), p(0,j,'d'), p.lda* sizeof(FHigh), 
                    Anext->get_lda()* sizeof(FHigh), p.b, GPU_MEMCPY_DEVICE_TO_DEVICE);
            
            if(Anext->get_lda() >0){
                 fprintf(outfile,"diagonal nextA %d %d\n",i, Anext->get_lda());
                GPU_MEMCPY(cpu_nextA, (*Anext)(0,'d'), b * Anext->get_lda() *sizeof(FHigh), GPU_MEMCPY_DEVICE_TO_HOST);
                print_matrix(cpu_nextA,  Anext->get_lda() ,b * Anext->get_lda(),outfile);
            }
        }

        // Preparing 2 panels
        else if(rootrow == grid.row){
            MPI_Bcast(d_piv, ldpiv * b, T2MPI<FHigh>::type, rootcol, grid.hcomm);
        
            checkGPUblas(GPUBLAS_STRSM(p.cuHandle, GPUBLAS_SIDE_LEFT, GPUBLAS_FILL_MODE_UPPER, 
                                        GPUBLAS_OP_N, GPUBLAS_DIAG_NON_UNIT, 
                                        p.b, p2.npcol* p.b, alpha,
                                        d_piv, ldpiv,
                                        p2(i,0,'d'), p2.d_lda));
            
            fprintf(outfile,"row %d???\n",i);
            GPU_MEMCPY(p2(0, 0), p2(0, 0, 'd'), b * p2.npcol* p.nprow*b *sizeof(FHigh), GPU_MEMCPY_DEVICE_TO_HOST);
            print_matrix(p2(0,0),p.lda,p.lda*p.lda,outfile);
            fflush(stdout);
            
            GPU_DEVICE_SYNCHRONIZE();
            Bnext->set_start(0);
            right_trans(p2(i,0,'d'), p.b, p.lda, (*Bnext)(0,'d'), Bnext->get_lda());
            
             fprintf(outfile,"row nextB\n",i);
            GPU_MEMCPY(cpu_nextB, (*Bnext)(0,'d'), b * Bnext->get_lda() *sizeof(FHigh), GPU_MEMCPY_DEVICE_TO_HOST);
            print_matrix(cpu_nextB,  Bnext->get_lda() ,b * Bnext->get_lda(),outfile);
            
            --i;
            --schur_row;
             Anext->set_start(0);
            Bnext->set_start(0);
            Anext->lda = (i+1)*b;
            //offset = b;
            
        }
        else if(rootcol == grid.col){
            Anext->set_start(0);
            Anext->lda = (i+1)*b;
            GPU_MEMCPY_2D((*Anext)(0,'d'), Anext->get_lda()* sizeof(FHigh), p(0,j,'d'), p.lda* sizeof(FHigh), 
                    Anext->get_lda()* sizeof(FHigh), p.b, GPU_MEMCPY_DEVICE_TO_DEVICE);
            if(Anext->get_lda() >0){
                fprintf(outfile,"column nextA\n",i);
                GPU_MEMCPY(cpu_nextA, (*Anext)(0,'d'), b * Anext->get_lda() *sizeof(FHigh), GPU_MEMCPY_DEVICE_TO_HOST);
                print_matrix(cpu_nextA,  Anext->get_lda() ,b * Anext->get_lda(),outfile);
            }
        }
        else
        {
            Anext->set_start(0);
            Bnext->set_start(0);
            Anext->lda = (i+1)*b;
        }
        fprintf(outfile,"send/recive %d %d\n", Anext->get_lda() * b, Bnext->get_lda() * b);
        fflush(outfile);
        MPI_Bcast(const_cast<FHigh *> ((*Anext)(0,'d')), Anext->get_lda() * b, T2MPI<FHigh>::type, rootcol, grid.hcomm);
        MPI_Bcast(const_cast<FHigh *> ((*Bnext)(0,'d')), Bnext->get_lda() * b, T2MPI<FHigh>::type, rootrow, grid.vcomm);
        // All panel ready to send
        GPU_DEVICE_SYNCHRONIZE();
              // Luanch gemm before the MPI
        if(Anext->get_lda() > 0)
        {
                GPUBLAS_SGEMM_EX(p.cuHandle,
                    GPUBLAS_OP_N,
                    GPUBLAS_OP_T,
                    Anext->get_lda()-offset,
                    p.b * p2.npcol,
                    b,
                    &nalf,
                    (void *)(*Anext)(0, 'd'), GPU_R_32F, Anext->get_lda(),
                    (void *)(*Bnext)(0, 'd'), GPU_R_32F, Bnext->get_lda(),
                    beta,
                    p2(0, 0, 'd'),GPU_R_32F,p.lda);
        }
        
         fprintf(outfile,"end iteration???\n",i);
            GPU_MEMCPY(p2(0, 0), p2(0, 0, 'd'), b * p2.npcol* p.nprow*b *sizeof(FHigh), GPU_MEMCPY_DEVICE_TO_HOST);
            print_matrix(p2(0,0),p.lda,p.lda*p.lda,outfile);
            fflush(stdout);
        
        GPU_DEVICE_SYNCHRONIZE();
        
        dogemm = 1;
        LRPanels<FHigh> *t;
        t = Aprev;
        Aprev = Anext;
        Anext = t;
        t = Bprev;
        Bprev = Bnext;
        Bnext = t;
        schur_row = i;
        schur_col = j;
    }
     fprintf(outfile,"exit\n");
     GPU_MEMCPY(p2(0, 0), p2(0, 0, 'd'), b * p2.npcol* p.nprow*b *sizeof(FHigh), GPU_MEMCPY_DEVICE_TO_HOST);
     print_matrix(p2(0,0),p.lda,p.lda*p.lda,outfile);
      fflush(stdout);
}
#endif