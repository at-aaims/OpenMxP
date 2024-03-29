#ifndef ITERATIVE_REFINEMENT_HPP
#define ITERATIVE_REFINEMENT_HPP
#include "grid.hpp"
#include "hpl_rand.hpp"
#include "panel.hpp"
#include "panel_gemv.hpp"
#include "panel_norm.hpp"
#include "panel_trsv.hpp"
#include "timer.hpp"
#include <cfloat>
#include <cstdio>

struct IRErrors {
    double residual;
    double hpl_harness;
};

template <typename FPanel, template <class> class Matgen>
void iterative_tester(int my_init, Panels<FPanel> &p, Matgen<double> &mg,
                              double *x, double *w, size_t ldv, double *rhs,
                              double norma, double normb, int maxit,
                              Grid &grid) {
    int const nb = p.nblocks;
    int const n = nb * p.b;
    size_t wlen = ldv * 10 + 2 * p.b * p.b;
    double *r = NULL;
    double *v = NULL;
    double *vw1_dev;
    FPanel *vw2_dev;
    double *dx_dev;
    double *testing_r = static_cast<double*>(std::malloc(sizeof(double) * ldv));
    r = w;
    v = w + ldv;
    double normr=0;

    size_t bytes = (size_t)(p.b * p.nprow) * (size_t)(p.b * p.npcol) * (size_t)sizeof(float);
    GPU_MEMCPY(p(0, 0), p(0, 0, 'd'), bytes, GPU_MEMCPY_DEVICE_TO_HOST);
    GPU_MEMCPY(testing_r, r, ldv*sizeof(double), GPU_MEMCPY_DEVICE_TO_HOST);
    printf("A from rank %d %d\n",grank,p.i1);
	print_matrix(p(0, 0),p.lda, p.lda*p.lda);
    fflush(stdout);

    GPU_MEMCPY(testing_r, rhs, ldv*sizeof(double), GPU_MEMCPY_DEVICE_TO_HOST);
    printf("rhs from rank %d\n",grank);
	print_matrix(testing_r,ldv/2,ldv/2);
    fflush(stdout);

    //trsvL_h(p, rhs, vw1_dev, ldv, grid);
    GPU_MEMCPY(testing_r, rhs, ldv*sizeof(double), GPU_MEMCPY_DEVICE_TO_HOST);
    printf("solve from rank %d\n",grank);
	print_matrix(testing_r,ldv/2,ldv/2);
    fflush(stdout);
    return;

    

    if(my_init == 1 || my_init == 4)
    {
        //panel_copycolv_dev(p, rhs, x);
        copycolv_h(p, rhs, x);
        divcolv_h(p, mg.diag_dev, x);
        copycolv_h(p, rhs, r);
        colv2rowv_h(p, x, v );
        if(my_init == 1){
            panel_gemv_h(-1., p, mg, false, v, 1., r, grid );
        }
       
        normr = colv_infnorm_h(p, r, grid, vw1_dev);
        trsvU_h(p, r, vw1_dev, ldv, grid);
        //panel_trsvL(p, r, vw1_dev, vw2_dev, ldv, grid);
        /*GPU_MEMCPY(testing_r, r, ldv*sizeof(double), GPU_MEMCPY_DEVICE_TO_HOST);
        if(p.i1 == p.j1 && grank == 10){
    	    printf("MV from rank %d %f\n",grank, normr);
		    print_matrix(testing_r,ldv/2,ldv/2);
            fflush(stdout);
        }*/
    }
    else
    {
        copycolv(p, rhs, x);
        divcolv(p, mg.diag, x);
        copycolv(p, rhs, r);
        colv2rowv(p, x, v);
        panel_gemv(my_init, -1., p, mg, false, v, 1., r, grid);
        normr = colv_infnorm(p, r, grid);
        if(p.i1 == p.j1 && grank == 10){
    	    printf("MV from rank %d %f\n",grank, normr);
		   // print_matrix(r,ldv/2,ldv/2);
            fflush(stdout);
        }
    }

   

}

#if 1
template <typename FPanel, template <class> class Matgen>
IRErrors iterative_refinement2(int my_init, Panels<FPanel> const &p, Matgen<double> &mg,
                              double *x, double *w, size_t ldv, double *rhs,
                              double norma, double normb, int maxit,
                              Grid &grid) {
    // do IR with approximated LU factors in p and the accurate initial matrix
    // which is generated by mg. x is solution. rhs is the right-hand-side
    // vector. w is working vectors. ldv is the leading dimention of w. set good
    // ldv for better performance. norma is inf-norm of the initial matrix.
    // normb is the inf-norm of rhs.
    int const nb = p.nblocks;
    int const n = nb * p.b;
    double *r = w;
    double *v = w + ldv;

    //double tmisc=0, time_trsvL=0, time_trsvU=0, time_pgv = 0, begintr_time, endtr_time;
    if(my_init != 1){
        copycolv(p, rhs, x);
        divcolv(p, mg.diag, x);        
        for (int iter = 0; iter < maxit; ++iter) {
            copycolv(p, rhs, r);
            double normx = colv_infnorm(p, x, grid);
            colv2rowv(p, x, v);
            panel_gemv(my_init, -1., p, mg, false, v, 1., r, grid);
            double normr = colv_infnorm(p, r, grid);
            MPI_Barrier(MPI_COMM_WORLD);
            double hplerror =
                normr / (norma * normx + normb) * 1. / (n * DBL_EPSILON / 2);
            if (grid.row == 0 && grid.col == 0) {
                printf("# iterative refinement: step=%d, residual=%e, "
                            "hpl-harness=%f\n",
                            iter, normr, hplerror);
            }
            if (hplerror < 16.) {
                return {normr, hplerror};
            }
            fflush(stdout);

            // x_1 = x_0 + (LU)^{-1} r
            panel_trsvL(p, r, v, ldv, grid);
            panel_trsvU(p, r, v, ldv, grid);
            addcolv(p, r, x);
        }
    }else{
        copycolv_h(p, rhs, x);
        divcolv_h(p, mg.diag_dev, x);
        for (int iter = 0; iter < maxit; ++iter) {
            copycolv_h(p, rhs, r);
            double normx = colv_infnorm_h(p, x, grid,v);
            colv2rowv_h(p, x, v);
            panel_gemv_h(-1., p, mg, false, v, 1., r, grid);
            
            GPU_THREAD_SYNCHRONIZE(0); 
            MPI_Barrier(MPI_COMM_WORLD);
            
            double normr = colv_infnorm_h(p, r, grid,v);
            // hplerror := \|b-Ax\|_\infty / (\|A\|_\infty \|x\|_\infty +
            // \|b\|_\infty) * (n * \epsilon)^{-1}
            double hplerror =
                normr / (norma * normx + normb) * 1. / (n * DBL_EPSILON / 2);
            if (grid.row == 0 && grid.col == 0) {
                printf("# iterative refinement: step=%d, residual=%e, "
                            "hpl-harness=%f\n",
                            iter, normr, hplerror);
            }
            if (hplerror < 16.) {
                return {normr, hplerror};
            }
            // x_1 = x_0 + (LU)^{-1} r
            trsvL_h(p, r, v, ldv, grid);
            trsvU_h(p, r, v, ldv, grid);
            addcolv_h(p, r, x);
        }
    }
    // OMG!
    return {-1., -1.};
}
#endif

/*template<typename FPanel>
IRErrors iterative_refinement(Panels<FPanel>const&p, HMGen<double>& mg,
        double* x, double* w, size_t ldv, double* rhs, double norma, double
normb, int maxit, Grid&grid)
{
        int const nb = p.nblocks;
        int const n = nb * p.b;
        double*r = w;
        double*v = w + ldv;
        // initial approximation
        // trsv
        copycolv(p, rhs, x);
        panel_trsvL(p, x, v, ldv, grid);
        panel_trsvU(p, x, v, ldv, grid);

        for(int iter=0; iter<maxit; ++iter){
                copycolv(p, rhs, r);
                double normx = colv_infnorm(p, x, grid);
                colv2rowv(p, x, v);
                // compute residual
                panel_gemv(-1., p, mg, false, v, 1., r, grid);
                double normr = colv_infnorm(p, r, grid);
                double hplerror = normr / (norma*normx + normb) * 1./(n *
DBL_EPSILON/2); if(grid.row==0 && grid.col==0){ std::printf("# iterative
refinement: step=%3d, residual=%20.16e hpl-harness=%f\n", iter, normr,
hplerror); fflush(stdout);
                }
                if(hplerror < 16.) return {normr, hplerror};

                panel_trsvL(p, r, v, ldv, grid);
                panel_trsvU(p, r, v, ldv, grid);
                addcolv(p, r, x);
        }
        // OMG!
        return {-1., -1.};
}*/

#endif
