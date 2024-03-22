#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <cmath>
#include <mpi.h>

using namespace std;

#include <cblas.h>

#define IDX(I,J) ((J)*Nx + (I)) // This allows each element in the grid to be accessed column by column 

#include "LidDrivenCavity.h"
#include "SolverCG.h"

LidDrivenCavity::LidDrivenCavity()
{
}

LidDrivenCavity::~LidDrivenCavity()
{
    CleanUp();
}

void LidDrivenCavity::RecieveDataForMPI(MPI_Comm mycart_grid, int myrank, int mycoords[2], int l_rank, int r_rank, 
    int u_rank, int d_rank, int b_x, int b_y, int e_x, int e_y ){
        this->Mycart_grid = mycart_grid;
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

    v   = new double[Npts]();
    vnew = new double[Npts]();
    vgather = new double[Npts]();
    s   = new double[Npts]();
    tmp = new double[Npts]();
    cg  = new SolverCG(Nx, Ny, dx, dy);
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
}

void LidDrivenCavity::WriteSolution(std::string file)
{
    double* u0 = new double[Nx*Ny]();
    double* u1 = new double[Nx*Ny]();
    for (int i = 1; i < Nx - 1; ++i) {
        for (int j = 1; j < Ny - 1; ++j) {
            u0[IDX(i,j)] =  (s[IDX(i,j+1)] - s[IDX(i,j)]) / dy;
            u1[IDX(i,j)] = -(s[IDX(i+1,j)] - s[IDX(i,j)]) / dx;
        }
    }
    for (int i = 0; i < Nx; ++i) {
        u0[IDX(i,Ny-1)] = U;
    }

    std::ofstream f(file.c_str());
    std::cout << "Writing file " << file << std::endl;
    int k = 0;
    for (int i = 0; i < Nx; ++i)
    {
        for (int j = 0; j < Ny; ++j)
        {
            k = IDX(i, j);
            f << i * dx << " " << j * dy << " " << v[k] <<  " " << s[k] 
              << " " << u0[k] << " " << u1[k] << std::endl;
        }
        f << std::endl;
    }
    f.close();

    delete[] u0;
    delete[] u1;
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
        delete[] vgather;
        delete[] s;
        delete[] tmp;
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

    double dxi  = 1.0/dx;
    double dyi  = 1.0/dy;
    double dx2i = 1.0/dx/dx;
    double dy2i = 1.0/dy/dy;
    

    // Only do these ones if == mpi_proc_null


    /**
     * @brief The follwing boundary conditions(two for loops) are only to be preformed by the boundary ranks
     * So the idea here is, instead of going from 1 to Nx-1 it will go from begin_x to end_x 
    */
    if(B_x == 0){B_x =1;}
    if(B_y == 0){B_y =1;}
    if(E_x == Nx){--E_x;}
    if(E_y == Ny){--E_y;}




    for (int i = 1; i < Nx-1; ++i) {
        // bottom
        v[IDX(i,0)]    = 2.0 * dy2i * (s[IDX(i,0)]    - s[IDX(i,1)]); // Keeping this for now so I can run without mpi 
        

        // top
        v[IDX(i,Ny-1)] = 2.0 * dy2i * (s[IDX(i,Ny-1)] - s[IDX(i,Ny-2)])
                      - 2.0 * dyi*U; // Keeping this for now so I can run without mpi 
        // cout << v[IDX(i,Ny-1)] << endl;
    }



    for (int j = 1; j < Ny-1; ++j) {
        // left
         v[IDX(0,j)]    = 2.0 * dx2i * (s[IDX(0,j)]    - s[IDX(1,j)]);  // Keeping this for now so I can run without mpi 

        // right
         v[IDX(Nx-1,j)] = 2.0 * dx2i * (s[IDX(Nx-1,j)] - s[IDX(Nx-2,j)]);  // Keeping this for now so I can run without mpi 
            
    }
 

    // Compute interior vorticity
    for (int i = 1; i < Nx-1; ++i) {
        for (int j = 1; j < Ny-1; ++j) {
            v[IDX(i,j)] = dx2i*(
                    2.0 * s[IDX(i,j)] - s[IDX(i+1,j)] - s[IDX(i-1,j)])
                        + 1.0/dy/dy*(
                    2.0 * s[IDX(i,j)] - s[IDX(i,j+1)] - s[IDX(i,j-1)]);
            // cout << v[IDX(i,j)] << endl;
            // cout << s[IDX(i,j)] << endl;
            
        }
    }

     
    

    // Time advance vorticity
    for (int i = 1; i < Nx-1; ++i) {
        for (int j = 1; j < Ny-1; ++j) {
            vnew[IDX(i,j)] = v[IDX(i,j)] + dt*(
                ( (s[IDX(i+1,j)] - s[IDX(i-1,j)]) * 0.5 * dxi
                 *(v[IDX(i,j+1)] - v[IDX(i,j-1)]) * 0.5 * dyi)
              - ( (s[IDX(i,j+1)] - s[IDX(i,j-1)]) * 0.5 * dyi
                 *(v[IDX(i+1,j)] - v[IDX(i-1,j)]) * 0.5 * dxi)
              + nu * (v[IDX(i+1,j)] - 2.0 * v[IDX(i,j)] + v[IDX(i-1,j)])*dx2i
              + nu * (v[IDX(i,j+1)] - 2.0 * v[IDX(i,j)] + v[IDX(i,j-1)])*dy2i);
              MPI_Gather(&vnew[IDX(i,j)],1, MPI_DOUBLE, &vgather[IDX(i,j)], (Nx-2)*(Ny-2), MPI_DOUBLE,0,Mycart_grid);
        }

    }

//    if(Myrank == 0){
//        MPI_Gather(&vnew[IDX(i,j)],Nx*Ny, MPI_DOUBLE, &vgather[IDX(i,j)], (Nx-1)*(Ny-1), MPI_DOUBLE,0,Mycart_grid);
//    }
 

    // Sinusoidal test case with analytical solution, which can be used to test
    // the Poisson solver
    /*
    const int k = 3;
    const int l = 3;
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            vnew[IDX(i,j)] = -M_PI * M_PI * (k * k + l * l)
                                       * sin(M_PI * k * i * dx)
                                       * sin(M_PI * l * j * dy);
        }
    }
    */
    if(Myrank==0){cg->Solve(vgather, s);}
    
}
