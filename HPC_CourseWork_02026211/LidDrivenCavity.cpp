#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <cmath>
#include <mpi.h>
#include <omp.h>

using namespace std;

#include <cblas.h>

#define IDX(I,J) ((J)*Nx + (I)) // This allows each element in the grid to be accessed column by column 
#define IDX_mini(I,J) ((J)*Ex + (I)) // This is for the submaticies 

#include "LidDrivenCavity.h"
#include "SolverCG.h"

/**
 * @class LidDrivenCavity
 * @brief This class is responsible for calculating and updating vorticity for a given stream function that is performed
 * by the Solver CG class
 * @note This class takes in all the information about ranks including their coordinates and rank numbers to discretise 
 * the working domain on the cartesian grid accordingly
 * 
*/



LidDrivenCavity::LidDrivenCavity()
{
}

LidDrivenCavity::~LidDrivenCavity()
{
    CleanUp();
}

void LidDrivenCavity::RecieveDataForMPI(MPI_Comm mycart_grid, int size,int myrank, int mycoords[2], int l_rank, int r_rank, 
    int u_rank, int d_rank, int b_x, int b_y, int e_x, int e_y ){
        this->Mycart_grid = mycart_grid;
        this->AllSize = size;
        this->Myrank = myrank;
        this->Mycoords = mycoords;
        this->L_rank = l_rank;
        this->R_rank = r_rank;
        this->D_rank = d_rank;
        this->U_rank = u_rank;
        this->B_x = b_x;
        this->B_y = b_y;
        this->E_x = e_x;
        this->E_y = e_y;
}

void LidDrivenCavity::SetDomainSize(double xlen, double ylen)
{
    this->Lx = xlen;
    this->Ly = ylen;
    UpdateDxDy();
}

void LidDrivenCavity::SetGridSize(int nx, int ny)
{
    this->Nx = nx;
    this->Ny = ny;
    UpdateDxDy();
}

void LidDrivenCavity::SetTimeStep(double deltat)
{
    this->dt = deltat;
}

void LidDrivenCavity::SetFinalTime(double finalt)
{
    this->T = finalt;
}

void LidDrivenCavity::SetReynoldsNumber(double re)
{
    this->Re = re;
    this->nu = 1.0/re;
}

void LidDrivenCavity::Initialise()
{
    CleanUp();
    beginx  = 0;
    beginy = 0;  
    
    if(B_x == 0){ beginx = 1;} // Start of global boundary
    if(B_y == 0){ beginy =1;} //  Start of global boundary 
    Ex = E_x-B_x;
    Ey = E_y-B_y;
    endx = Ex;
    endy = Ey;
    if(E_x== Nx){ endx = Ex -1;}  // End of Global boundary
    if(E_y == Ny){ endy = Ey -1;} // End of global boundary

    v_gather = new double[Nx*Ny]();
    s_gather = new double[Nx*Ny]();
    v   = new double[(Ex)*(Ey)]();
    vnew = new double[(Ex)*(Ey)]();
    s   = new double[(Ex)*(Ey)]();
    tmp = new double[(Ex)*(Ey)]();
    
    cg  = new SolverCG(Ex,Ey, dx, dy, Nx,Ny);
    cg->RecieveCG(beginx, beginy, endx, endy, L_rank , R_rank, D_rank, U_rank , E_x, E_y , B_x, B_y , Mycart_grid); // Rank info

    
}

/**
 * @brief The Integrate function allows iteration of time i.e. advancing to the next time step
 * Each time, the advance function is called which preforms all the equations
*/
void LidDrivenCavity::Integrate()
{
    int NSteps = ceil(T/dt);
    for (int t = 0; t < NSteps; ++t)
    {
        std::cout << "Step: " << setw(8) << t
                  << "  Time: " << setw(8) << t*dt
                  << std::endl;        

        Advance();   
    }
    

    /**
     * @brief This part of the integrate function allows the code to be written back onto a the Global matrix so it can be written into
     * the file in the correct format 
    */
    double v_loc[Nx*Ny];
    double s_loc [Nx*Ny];

    for (int i = 0; i < Nx*Ny; ++i){
        v_loc[i] = 0;
        s_loc[i] = 0;
    }
    

    for (int i = 0; i < Ex; ++i) {
        for (int j = 0; j < Ey; ++j) {
            v_loc[IDX((i+B_x),(j+B_y))] = vnew[IDX_mini(i,j)];
            s_loc[IDX((i+B_x),(j+B_y))] = s[IDX_mini(i,j)];
        } 
    }
    
    MPI_Reduce(v_loc, v_gather, Ny*Nx, MPI_DOUBLE, MPI_SUM,0,Mycart_grid);
    MPI_Reduce(s_loc, s_gather, Ny*Nx, MPI_DOUBLE, MPI_SUM,0,Mycart_grid);
    
}

/**
 * @note WriteSolution is only preformed on rank zeros and is used to out put into the file 
*/
void LidDrivenCavity::WriteSolution(std::string file)
{
    
    if(Myrank == 0){
        double* u0 = new double[Nx*Ny]();
        double* u1 = new double[Nx*Ny]();
        for (int i = 1; i < Nx - 1; ++i) {
            for (int j = 1; j < Ny - 1; ++j) {
                u0[IDX(i,j)] =  (s_gather[IDX(i,j+1)] - s_gather[IDX(i,j)]) / dy;
                u1[IDX(i,j)] = -(s_gather[IDX(i+1,j)] - s_gather[IDX(i,j)]) / dx;
            }
        }
        for (int i = 0; i < Nx; ++i) {
            u0[IDX(i,Ny-1)] = U;
        }
        std::ofstream f(file.c_str());
        std::cout << "Writing file " << file << std::endl;
        int k = 0;
        for (int i = 0; i < Nx; ++i){
            for (int j = 0; j < Ny; ++j){
                k = IDX(i, j);
                f << i * dx << " " << j * dy << " " << v_gather[k] <<  " " << s_gather[k] 
                << " " << u0[k] << " " << u1[k] << std::endl;
            }
            f << std::endl;
        }
        f.close();
        delete[] u0;
        delete[] u1;

    }   
}


void LidDrivenCavity::PrintConfiguration()
{
    cout << "Grid size: " << Nx << " x " << Ny << endl;
    cout << "Spacing:   " << dx << " x " << dy << endl;
    cout << "Length:    " << Lx << " x " << Ly << endl;
    cout << "Grid pts:  " << Npts << endl;
    cout << "Timestep:  " << dt << endl;
    cout << "Steps:     " << ceil(T/dt) << endl;
    cout << "Reynolds number: " << Re << endl;
    cout << "Linear solver: preconditioned conjugate gradient" << endl;
    cout << endl;
    if (nu * dt / dx / dy > 0.25) {
        cout << "ERROR: Time-step restriction not satisfied!" << endl;
        cout << "Maximum time-step is " << 0.25 * dx * dy / nu << endl;
        exit(-1);
    }
}


void LidDrivenCavity::CleanUp()
{
    if (v) {
        delete[] v;
        delete[] vnew;
        delete[] s;
        delete[] tmp;
        delete[] v_gather ;
        delete[] s_gather ;
        delete cg;
    }
}

void LidDrivenCavity::UpdateDxDy()
{
    dx = Lx / (Nx-1);
    dy = Ly / (Ny-1);
    Npts = Nx * Ny;
}


/**
 * @brief Advance allows the calculation of vorticity for a given timestep and then solves for a new vorticity by calling solve
 * 
*/
void LidDrivenCavity::Advance()
{

    double dxi  = pow(dx,-1);
    double dyi  = pow(dy,-1);
    double dx2i = pow(dx,-2);
    double dy2i = pow(dy,-2);
    

   


    /**
     * The follwing boundary conditions(two for loops) are only to be preformed by the boundary ranks
     * So the idea here is, instead of going from 1 to Nx-1 it will go from begin_x to end_x 
    */
    
   
    

    double LColSend_s[Ey], LColRecv_s[Ey], RColSend_s[Ey], RColRecv_s[Ey];
    double BRowSend_s[Ex], BRowRecv_s[Ex], TRowSend_s[Ex], TRowRecv_s[Ex];
// --------------------------------------------------------------------------------------------------------------------
    if(L_rank != MPI_PROC_NULL){  
          
        for (int j =0; j < Ey; ++j) {
            LColSend_s[j] = s[IDX_mini(0, j)];
        }
        MPI_Sendrecv(&LColSend_s, Ey, MPI_DOUBLE, L_rank, 0, &LColRecv_s, Ey, MPI_DOUBLE, L_rank, 0, Mycart_grid, MPI_STATUS_IGNORE);
    } 

    if(R_rank != MPI_PROC_NULL){
        
        for (int j =0; j < Ey; ++j) {
            RColSend_s[j] = s[IDX_mini(Ex-1, j)];
        }
        MPI_Sendrecv(&RColSend_s, Ey, MPI_DOUBLE, R_rank, 0, &RColRecv_s, Ey, MPI_DOUBLE, R_rank, 0, Mycart_grid, MPI_STATUS_IGNORE);
    }  

    // --------------------------------------------------------------------------------------------------------------------

    if (U_rank != MPI_PROC_NULL){
        
        for (int i =0; i < Ex; ++i){
            TRowSend_s[i] = s[IDX_mini(i, Ey-1)];
        }
        MPI_Sendrecv(&TRowSend_s, Ex, MPI_DOUBLE, U_rank, 0, &TRowRecv_s, Ex, MPI_DOUBLE, U_rank, 0, Mycart_grid, MPI_STATUS_IGNORE);
    }

    if (D_rank != MPI_PROC_NULL){
        
        for (int i =0; i < Ex; ++i){
            BRowSend_s[i] = s[IDX_mini(i, 0)];
            
        }
        MPI_Sendrecv(&BRowSend_s, Ex, MPI_DOUBLE, D_rank, 0, &BRowRecv_s, Ex, MPI_DOUBLE, D_rank, 0, Mycart_grid, MPI_STATUS_IGNORE);
    }


   
    // --------------------------------------------------------------------------------------------------------------------
    
    for (int i = beginx; i < endx; ++i) {
        // bottom        
        if (D_rank == MPI_PROC_NULL ){
            v[IDX_mini(i,0)]    = 2.0 * dy2i * (s[IDX_mini(i,0)]    - s[IDX_mini(i,1)]);  
        }
        // top        
        if (U_rank == MPI_PROC_NULL){
            v[IDX_mini(i,Ey-1)] = 2.0 * dy2i * (s[IDX_mini(i,Ey-1)] - s[IDX_mini(i,Ey-2)])
                       - 2.0 * dyi*U;
        }       
    }

    
    for (int j = beginy; j < endy; ++j) {
        // left       
        if (L_rank == MPI_PROC_NULL ){
            v[IDX_mini(0,j)]    = 2.0 * dx2i * (s[IDX_mini(0,j)]    - s[IDX_mini(1,j)]);
        }       
        // right        
        if (R_rank == MPI_PROC_NULL){
            v[IDX_mini(Ex-1,j)] = 2.0 * dx2i * (s[IDX_mini(Ex-1,j)] - s[IDX_mini(Ex-2,j)]);
        }            
    }
    // --------------------------------------------------------------------------------------------------------------------
 
    // Compute interior vorticity
    #pragma omp parallel for schedule(static) firstprivate(v,s) collapse(2)
    for (int j = beginy; j < endy; ++j) {
        for (int i = beginx; i < endx; ++i) {

            double sLmin = (i == 0)? LColRecv_s[j] : s[IDX_mini(i-1,j)];
            double sRmax = (i == Ex-1 )? RColRecv_s[j] : s[IDX_mini(i+1,j)];
            double sBmin = (j == 0)? BRowRecv_s[i] : s[IDX_mini(i,j-1)] ;
            double sTmax = (j == Ey-1 )? TRowRecv_s[i] : s[IDX_mini(i,j+1)] ;

            v[IDX_mini(i,j)] = dx2i*(
                    2.0 * s[IDX_mini(i,j)] - sRmax - sLmin)
                        + dy2i*(
                    2.0 * s[IDX_mini(i,j)] - sTmax - sBmin);
        }
    }

  
    // --------------------------------------------------------------------------------------------------------------------

    double LColSend_v[Ey], LColRecv_v[Ey], RColSend_v[Ey], RColRecv_v[Ey];
    double BRowSend_v[Ex], BRowRecv_v[Ex], TRowSend_v[Ex], TRowRecv_v[Ex];

    if(L_rank != MPI_PROC_NULL){
        
        for (int j =0; j < Ey; ++j) {
            LColSend_v[j] = v[IDX_mini(0, j)];
        }
        MPI_Sendrecv(&LColSend_v, Ey, MPI_DOUBLE, L_rank, 0, &LColRecv_v, Ey, MPI_DOUBLE, L_rank, 0, Mycart_grid, MPI_STATUS_IGNORE);
    }  // This Sends the left coloumn to the left rank and recieves from the left rank

    if(R_rank != MPI_PROC_NULL){
        
        for (int j =0; j < Ey; ++j) {
            RColSend_v[j] = v[IDX_mini(Ex-1, j)];
        }
        MPI_Sendrecv(&RColSend_v, Ey, MPI_DOUBLE, R_rank, 0, &RColRecv_v, Ey, MPI_DOUBLE, R_rank, 0, Mycart_grid, MPI_STATUS_IGNORE);
    }  // This Sends the right coloumn to the right rank and recieves from the right rank

    // --------------------------------------------------------------------------------------------------------------------

    if (U_rank != MPI_PROC_NULL){
        
        for (int i =0; i < Ex; ++i){
            TRowSend_v[i] = v[IDX_mini(i, Ey-1)];
        }
        MPI_Sendrecv(&TRowSend_v,Ex, MPI_DOUBLE, U_rank, 0, &TRowRecv_v, Ex, MPI_DOUBLE, U_rank, 0, Mycart_grid, MPI_STATUS_IGNORE);
    }

    if (D_rank != MPI_PROC_NULL){
        
        for (int i =0; i < Ex; ++i){
            BRowSend_v[i] = v[IDX_mini(i, 0)];
        }
        MPI_Sendrecv(&BRowSend_v, Ex, MPI_DOUBLE, D_rank, 0, &BRowRecv_v, Ex, MPI_DOUBLE, D_rank, 0, Mycart_grid, MPI_STATUS_IGNORE);
    }

    // --------------------------------------------------------------------------------------------------------------------


    

    /** Time advance vorticity */
    #pragma omp parallel for schedule(static) firstprivate(v,s, vnew) collapse(2)
    for (int j = beginy; j < endy; ++j) {
        for (int i = beginx; i < endx; ++i) {
            double vLmin = (i == 0)? LColRecv_v[j] : v[IDX_mini(i-1,j)];
            double vRmax = (i == Ex-1)? RColRecv_v[j] : v[IDX_mini(i+1,j)];

            double vBmin = (j == 0)? BRowRecv_v[i] : v[IDX_mini(i,j-1)] ;
            double vTmax = (j == Ey-1)? TRowRecv_v[i] : v[IDX_mini(i,j+1)] ;

            double sLmin = (i == 0)? LColRecv_s[j] : s[IDX_mini(i-1,j)];
            double sRmax = (i == Ex-1 )? RColRecv_s[j] : s[IDX_mini(i+1,j)];

            double sBmin = (j == 0)? BRowRecv_s[i] : s[IDX_mini(i,j-1)] ;
            double sTmax = (j == Ey-1 )? TRowRecv_s[i] : s[IDX_mini(i,j+1)] ;

            vnew[IDX_mini(i,j)] = v[IDX_mini(i,j)] + dt*(
                ( (sRmax - sLmin) * 0.5 * dxi
                    *(vTmax - vBmin) * 0.5 * dyi)
                - ( (sTmax - sBmin) * 0.5 * dyi
                    *(vRmax - vLmin) * 0.5 * dxi)
                + nu * (vRmax - 2.0 * v[IDX_mini(i,j)] + vLmin)*dx2i
                + nu * (vTmax - 2.0 * v[IDX_mini(i,j)] + vBmin)*dy2i);          
        }
    }
    cg->Solve(vnew, s);           
}
