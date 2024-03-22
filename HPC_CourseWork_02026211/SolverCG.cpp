#pragma once
#include <iostream>
#include <algorithm>
#include <cstring>
#include <mpi.h>
#include <omp.h>
#include <cmath>
using namespace std;

#include <cblas.h>

#include "SolverCG.h"

#define IDX_mini(I,J) ((J)*loc_endx + (I))

/**
 * @class SolverCG 
 * @brief This class solves for vorticity and updates the stream function.
 * @param dx  The grid spacing in the x direction 
 * @param dy  The grid spacing in the y direction 
 * @param Nx  The number of grid in the x direction 
 * @param Ny  The number of grid in the y direction 
 */


SolverCG::SolverCG(int pNx, int pNy, double pdx, double pdy, int g_Nx, int g_Ny)
{
    dx = pdx;
    dy = pdy;
    Nx = pNx;
    Ny = pNy;
    G_Nx = g_Nx; // where each grid begins globally in the x dierction 
    G_Ny = g_Ny; // where each grid begins globally in the y dierction 
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
 * @param Bx, By are set to zero or one depending wether the starting point of tyhe rank grid is at the boundary or not
 * @param Ex, Ey theses are set to be the end of each rank depending on wether the rank is at the boundary or nor
*/
void SolverCG::RecieveCG(int bx, int by, int ex, int ey,int l_rank ,int r_rank, int d_rank, int u_rank ,
                        int g_ex, int g_ey, int g_bx, int g_by , MPI_Comm mygrid)
{

    Bx = bx; 
    By = by;
    Ex = ex;
    Ey = ey;
    Lrank = l_rank;
    Rrank = r_rank;
    Drank = d_rank;
    Urank = u_rank;
    G_bx = g_bx;
    G_by = g_by;
    G_ex = g_ex;
    G_ey = g_ey;
    Mygrid = mygrid;
    loc_endx = G_ex - G_bx;
    loc_endy = G_ey - G_by;

}
/**
 * @brief Solve function takes in vorticity as b and stream function as x and updates them both using the conjeguate gradient method
 * @param b  this is equivelant to vorticity
 * @param x  this is equivelant to stream function
 */ 


void SolverCG::Solve(double* b, double* x) {  
    unsigned int n = Nx*Ny;
    int k;
    double alpha, alpha_Global, Alpha_Global, tempa;
    double beta, beta_Global, Beta_Global, tempb;
    double eps, eps_Global;
    double tol = 0.001;

    eps = cblas_dnrm2(n, b, 1);

    eps = eps*eps;
    MPI_Allreduce(&eps, &eps_Global, 1, MPI_DOUBLE, MPI_SUM, Mygrid); /**This is u*/
    eps_Global = sqrt(eps_Global);

    if (eps_Global < tol*tol) {
        std::fill(x, x+n, 0.0);
        cout << "Norm is " << eps_Global << endl;
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
        MPI_Allreduce(&alpha, &alpha_Global, 1, MPI_DOUBLE, MPI_SUM, Mygrid); //Calculating global value ie summing up 
        tempa = cblas_ddot(n, r, 1, z, 1);
        alpha_Global =  tempa* pow(alpha_Global,-1); // compute alpha_k
        MPI_Allreduce(&alpha_Global, &Alpha_Global, 1, MPI_DOUBLE, MPI_SUM, Mygrid); //Calculating final Global alpha value

        beta  = cblas_ddot(n, r, 1, z, 1);  // z_k^T r_k
        MPI_Allreduce(&beta, &beta_Global, 1, MPI_DOUBLE, MPI_SUM, Mygrid); //Calculating global value ie summing up 

        cblas_daxpy(n,  Alpha_Global, p, 1, x, 1);  // x_{k+1} = x_k + alpha_k p_k
        cblas_daxpy(n, -Alpha_Global, t, 1, r, 1); // r_{k+1} = r_k - alpha_k A p_k

        eps = cblas_dnrm2(n, r, 1);
        eps = eps*eps;
        MPI_Allreduce(&eps, &eps_Global, 1, MPI_DOUBLE, MPI_SUM, Mygrid);
        eps_Global = sqrt(eps_Global); /// Calculation of global variable 
        
        if (eps_Global < tol*tol) {
            break;
        }
        Precondition(r, z);
        tempb = cblas_ddot(n, r, 1, z, 1);
        beta_Global =  tempb* pow(beta_Global,-1);
        MPI_Allreduce(&beta_Global, &Beta_Global, 1, MPI_DOUBLE, MPI_SUM, Mygrid); //Calculating global value ie summing up 


        cblas_dcopy(n, z, 1, t, 1);
        cblas_daxpy(n, Beta_Global, p, 1, t, 1);
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

    // cout << "It is Apply operator" << endl;
    double LColSend_in[loc_endy], LColRecv_in[loc_endy], RColSend_in[loc_endy], RColRecv_in[loc_endy];
    double BRowSend_in[loc_endx], BRowRecv_in[loc_endx], TRowSend_in[loc_endx], TRowRecv_in[loc_endx];
    // --------------------------------------------------------------------------------------------------------------------
    if(Lrank != MPI_PROC_NULL){
        for (int j =0; j < loc_endy; ++j) {
            LColSend_in[j] = in[IDX_mini(0, j)];
        }
        MPI_Sendrecv(&LColSend_in, loc_endy, MPI_DOUBLE, Lrank, 0, &LColRecv_in, loc_endy, MPI_DOUBLE, Lrank, 0, Mygrid, MPI_STATUS_IGNORE);
    }  // This Sends the left coloumn to the left rank and recieves from the left rank

    if(Rrank != MPI_PROC_NULL){
        for (int j =0; j < loc_endy; ++j) {
            RColSend_in[j] = in[IDX_mini(loc_endx-1, j)];
        }
        MPI_Sendrecv(&RColSend_in, loc_endy, MPI_DOUBLE, Rrank, 0, &RColRecv_in, loc_endy, MPI_DOUBLE, Rrank, 0, Mygrid, MPI_STATUS_IGNORE);
    }  // This Sends the right coloumn to the right rank and recieves from the right rank

   // --------------------------------------------------------------------------------------------------------------------

    if (Urank != MPI_PROC_NULL){
        for (int i =0; i < loc_endx; ++i){
            TRowSend_in[i] = in[IDX_mini(i, loc_endy-1)];
        }
        MPI_Sendrecv(&TRowSend_in, loc_endx, MPI_DOUBLE, Urank, 0, &TRowRecv_in, loc_endx, MPI_DOUBLE, Urank, 0, Mygrid, MPI_STATUS_IGNORE);
    }

    if (Drank != MPI_PROC_NULL){
        for (int i =0; i < loc_endx; ++i){
            BRowSend_in[i] = in[IDX_mini(i, 0)];
        }
        MPI_Sendrecv(&BRowSend_in, loc_endx, MPI_DOUBLE, Drank, 0, &BRowRecv_in, loc_endx, MPI_DOUBLE, Drank, 0, Mygrid, MPI_STATUS_IGNORE);
    }   

    /**
     * @param inLim, inRmax, inBmin, inTmax these are the temporary variables that are used to assess the send and recieve variables
    */
   
    double dx2i = pow(dx,-2);
    double dy2i = pow(dy,-2);
    double inLmin, inRmax, inBmin, inTmax;

    for (int j = By; j < Ey; ++j) {
        for (int i = Bx; i < Ex; ++i) {
            inLmin = (i == 0)? LColRecv_in[j] : in[IDX_mini(i-1,j)];
            inRmax = (i == loc_endx-1 )? RColRecv_in[j] : in[IDX_mini(i+1,j)];
            

            inBmin = (j == 0)? BRowRecv_in[i] : in[IDX_mini(i,j-1)] ;
            inTmax = (j == loc_endy-1)? TRowRecv_in[i] : in[IDX_mini(i,j+1)] ;

            out[IDX_mini(i,j)] = ( -     inLmin
                              + 2.0*in[IDX_mini(i,   j)]
                              -     inRmax)*dx2i
                          + ( -     inBmin
                              + 2.0*in[IDX_mini(i,   j)]
                              -     inTmax)*dy2i;
        }
    }   
}


void SolverCG::Precondition(double* in, double* out) {
    // Assume ordered with y-direction fastest (column-by-column)

    int i, j;
    double dx2i = pow(dx,-2);
    double dy2i = pow(dy,-2);
    double factor = pow(2.0*(dx2i + dy2i),-1);
    for (i = Bx; i < Ex; ++i) {
        for ( j = By; j < Ey; ++j) {
            out[IDX_mini(i,j)] = in[IDX_mini(i,j)]*factor; 
        }
    }
    // Boundaries
    for (i = 0; i < loc_endx; ++i) {
        if(Drank == MPI_PROC_NULL){
        out[IDX_mini(i, 0)] = in[IDX_mini(i,0)]; // Bottom wall vorticity
        }
        if(Urank == MPI_PROC_NULL){
            out[IDX_mini(i, loc_endy-1)] = in[IDX_mini(i, loc_endy-1)]; // Top wall or lid vorticity
        }
    }
    // Continue from here
    for (j = 0; j < loc_endy; ++j) {
        if(Lrank == MPI_PROC_NULL){
            out[IDX_mini(0, j)] = in[IDX_mini(0, j)]; // Left wall vorticity
        }
        if(Rrank == MPI_PROC_NULL){
             out[IDX_mini(loc_endx-1, j)] = in[IDX_mini(loc_endx-1, j)]; // Right wall vorticity
        }      
    }
}

void SolverCG::ImposeBC(double* inout) {
    for (int i = 0; i < loc_endx; ++i) {
        if(Drank == MPI_PROC_NULL){
            inout[IDX_mini(i, 0)] = 0.0;  // Bottom wall boundary 
        }
        if(Urank == MPI_PROC_NULL){
            inout[IDX_mini(i, loc_endy-1)] = 0.0; // Top wall boundary 
        }
    }
    // Continue from here
    for (int j = 0; j < loc_endy; ++j) {
        if(Lrank == MPI_PROC_NULL){
            inout[IDX_mini(0, j)] = 0.0; // Left wall boundary
        }    
        if(Rrank == MPI_PROC_NULL){
            inout[IDX_mini(loc_endx-1, j)] = 0.0; // Right wall boundary
        }      
    }
}