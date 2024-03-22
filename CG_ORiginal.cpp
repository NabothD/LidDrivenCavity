#include <iostream>
#include <algorithm>
#include <cstring>
#include <mpi.h>
#include <omp.h>
using namespace std;

#include <cblas.h>

#include "SolverCG.h"

#define IDX(I,J) ((J)*Nx + (I))

/**
 * @brief This class solves for vorticity and stream function.
 * @param dx  The grid spacing in the x direction 
 * @param dy  The grid spacing in the y direction 
 * @param Nx  The number of grid in the x direction 
 * @param Ny  The number of grid in the y direction 
 */


SolverCG::SolverCG(int pNx, int pNy, double pdx, double pdy)
{
    dx = pdx;
    dy = pdy;
    Nx = pNx;
    Ny = pNy;
    int n = Nx*Ny;
    r = new double[n];
    p = new double[n];
    z = new double[n];
    t = new double[n]; //temp
}


SolverCG::~SolverCG()
{
    delete[] r;
    delete[] p;
    delete[] z;
    delete[] t;
}

/**
 * @brief Solve function takes in vorticity as b and stream function as x and updates them both
 * @param b  this is equivelant to vorticity
 * @param x  this is equivelant to stream function
 */ 

void SolverCG::Solve(double* bnew, double* xnew) {
    unsigned int n = Nx*Ny;
    int k;
    double alpha;
    double beta;
    double eps, eps_Global;
    double tol = 0.001;

    eps = cblas_dnrm2(n, b, 1);
    if (eps < tol*tol) {
        std::fill(x, x+n, 0.0);
        cout << "Norm is " << eps << endl;
        return;
    }

    ApplyOperator(x, t);
    cblas_dcopy(n, b, 1, r, 1);        // r_0 = b (i.e. b)
    ImposeBC(r);

    cblas_daxpy(n, -1.0, t, 1, r, 1);
    Precondition(r, z);
    cblas_dcopy(n, z, 1, p, 1);        // p_0 = r_0

    k = 0;
    do {
        k++;
        // Perform action of Nabla^2 * p
        ApplyOperator(p, t);

        alpha = cblas_ddot(n, t, 1, p, 1);  // alpha = p_k^T A p_k
        alpha = cblas_ddot(n, r, 1, z, 1) / alpha; // compute alpha_k
        beta  = cblas_ddot(n, r, 1, z, 1);  // z_k^T r_k

        cblas_daxpy(n,  alpha, p, 1, x, 1);  // x_{k+1} = x_k + alpha_k p_k
        cblas_daxpy(n, -alpha, t, 1, r, 1); // r_{k+1} = r_k - alpha_k A p_k

        eps = cblas_dnrm2(n, r, 1);
        // PI_Allreduce(&alpha_local, &alpha_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        // MPI_Allreduce(&eps, &eps_Global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        if (eps < tol*tol) {
            break;
        }
        Precondition(r, z);
        beta = cblas_ddot(n, r, 1, z, 1) / beta;

        cblas_dcopy(n, z, 1, t, 1);
        cblas_daxpy(n, beta, p, 1, t, 1);
        cblas_dcopy(n, t, 1, p, 1);

    } while (k < 5000); // Set a maximum number of iterations

    if (k == 5000) {
        cout << "FAILED TO CONVERGE" << endl;
        exit(-1);
    }

    cout << "Converged in " << k << " iterations. eps = " << eps_Global << endl;

}


/**
 * @brief The ApplyOperator function calculates and updates the stream function for all interior grid points for  a given time
 * @param out   The stream function of the interior for all grid points
 * @param in  The interior vorticity at time t for all the interior grid points
 */ 

void SolverCG::ApplyOperator(double* in, double* out) {
    // Assume ordered with y-direction fastest (column-by-column)
   
    double dx2i = 1.0/dx/dx;
    double dy2i = 1.0/dy/dy;
    int jm1 = 0, jp1 = 2;

    for (int j = 1; j < Ny - 1; ++j) {
        for (int i = 1; i < Nx - 1; ++i) {
            out[IDX(i,j)] = ( -     in[IDX(i-1, j)]
                              + 2.0*in[IDX(i,   j)]
                              -     in[IDX(i+1, j)])*dx2i
                          + ( -     in[IDX(i, jm1)]
                              + 2.0*in[IDX(i,   j)]
                              -     in[IDX(i, jp1)])*dy2i;
        }
        jm1++;
        jp1++;
    }
}


void SolverCG::Precondition(double* in, double* out) {
    int i, j;
    double dx2i = 1.0/dx/dx;
    double dy2i = 1.0/dy/dy;
    double factor = 2.0*(dx2i + dy2i);
    for (i = 1; i < Nx - 1; ++i) {
        for (j = 1; j < Ny - 1; ++j) {
            out[IDX(i,j)] = in[IDX(i,j)]/factor; 
        }
    }
    // Boundaries
    
    for (i = 0; i < Nx; ++i) {
        out[IDX(i, 0)] = in[IDX(i,0)]; // Bottom wall vorticity
        out[IDX(i, Ny-1)] = in[IDX(i, Ny-1)]; // Top wall or lid vorticity
    }

    for (j = 0; j < Ny; ++j) {
        out[IDX(0, j)] = in[IDX(0, j)]; // Left wall vorticity
        out[IDX(Nx - 1, j)] = in[IDX(Nx - 1, j)]; // Right wall vorticity
    }
}

void SolverCG::ImposeBC(double* inout) {
        // Boundaries 
    for (int i = 0; i < Nx; ++i) {
        inout[IDX(i, 0)] = 0.0;  // Bottom wall boundary
        inout[IDX(i, Ny-1)] = 0.0; // Top wall boundary 
    }
    for (int j = 0; j < Ny; ++j) {
        inout[IDX(0, j)] = 0.0; // Left wall boundary
        inout[IDX(Nx - 1, j)] = 0.0; // Right wall boundary
    }

}