#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <cmath>
#include <mpi.h>
#include <vector>

using namespace std;

#include <cblas.h>

#define IDX(I,J) ((J)*Nx + (I)) // This allows each element in the grid to be accessed column by column 
#define IDX_mini(I,J) ((J)*endx + (I))


#include "LidDrivenCavity.h"
#include "SolverCG.h"

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
    if (AllSize >1){
        if(B_x == 0){B_x =1;}
        if(B_y == 0){B_y =1;}

        int Ex = E_x-B_x+2;
        int Ey = E_y-B_y+2;

        v   = new double[(Ex)*(Ey)]();
        vnew = new double[(Ex)*(Ey)]();
        vgather = new double[Npts]();
        sgather = new double[Npts]();
        s   = new double[(Ex)*(Ey)]();
        // cout << "This is what it is at the top  " << (E_x-B_x)<< endl; 
        tmp = new double[(Ex)*(Ey)]();
        cg  = new SolverCG(Ex-2,Ey-2, dx, dy);

    }
    else{
        v   = new double[Npts]();
        vnew   = new double[Npts]();
        s   = new double[Npts]();
        tmp = new double[Npts]();
        cg  = new SolverCG(Nx, Ny, dx, dy);
    }

    
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
            f << i * dx << " " << j * dy << " " << vnew[k] <<  " " << s[k] 
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
        delete[] sgather;
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
    int beginx  = 1;
    int beginy = 1; 
    int endx, endy, Ey, Ex;
    if (AllSize>1){
        if(B_x == 0){ beginx = 2;}
        if(B_y == 0){ beginy = 2;}
        Ex = E_x-B_x+2;
        Ey = E_y-B_y+2;
        endx = Ex-1;
        endy = Ey-1;
        if(E_x== Nx){ endx = Ex -2;}
        if(E_y == Ny){ endy = Ey -2;}

    }
    else{
        endx = Nx-1;
        endy =Ny-1;
    }
    


    // cout << endx  <<  "   " <<endy << endl;
   

    
    //  cout <<" This is Ex:  " <<Ex << endl;
    // if(Myrank){
    //     cout << "My rank is =  " << Myrank << " and my coordnates are :  " << Mycoords[0] << " , " << Mycoords[1] << endl ;
    //     cout << "My x position begins at: " << B_x << "  and ends at : " << E_x << endl;
    //     cout << "My y position begins at: " << B_y << "  and ends at : " << E_y << endl;
    // }
    
    
 

    for (int i = beginx; i < endx; ++i) {
        // bottom

        
        if (D_rank == MPI_PROC_NULL && AllSize>1){
            v[IDX_mini(i,1)]    = 2.0 * dy2i * (s[IDX_mini(i,1)]    - s[IDX_mini(i,2)]);  //done
        }
        else{
            v[IDX(i,0)]    = 2.0 * dy2i * (s[IDX(i,0)]    - s[IDX(i,1)]); // Keeping this for now so I can run without mpi 

        }
        

        // top
        
        if (U_rank == MPI_PROC_NULL && AllSize>1){
            v[IDX_mini(i,endy-1)] = 2.0 * dy2i * (s[IDX_mini(i,endy-1)] - s[IDX_mini(i,endy-2)])
                       - 2.0 * dyi*U;
            // cout << v[IDX_mini(i,Ey)] << endl; 
        }
        else{
            v[IDX(i,Ny-1)] = 2.0 * dy2i * (s[IDX(i,Ny-1)] - s[IDX(i,Ny-2)])
                       - 2.0 * dyi*U; // Keeping this for now so I can run without mpi 
        }
        
    }



    for (int j = beginx; j < endy; ++j) {
        // left
        
        if (L_rank == MPI_PROC_NULL && AllSize>1){
            v[IDX_mini(1,j)]    = 2.0 * dx2i * (s[IDX_mini(1,j)]    - s[IDX_mini(2,j)]);
        }
        else{
            v[IDX(0,j)]    = 2.0 * dx2i * (s[IDX(0,j)]    - s[IDX(1,j)]);  // Keeping this for now so I can run without mpi 
        }
        
        // right
        
        if (R_rank == MPI_PROC_NULL && AllSize>1){
            v[IDX_mini(endx-1,j)] = 2.0 * dx2i * (s[IDX_mini(endx-1,j)] - s[IDX_mini(endx-2,j)]);
        }
        else{
            v[IDX(Nx-1,j)] = 2.0 * dx2i * (s[IDX(Nx-1,j)] - s[IDX(Nx-2,j)]);  // Keeping this for now so I can run without mpi 
        }       
    }
 

    // Compute interior vorticity
    for (int i = beginx; i < endx; ++i) {
        for (int j = beginy; j < endy; ++j) {
            v[IDX_mini(i,j)] = dx2i*(
                    2.0 * s[IDX_mini(i,j)] - s[IDX_mini(i+1,j)] - s[IDX_mini(i-1,j)])
                        + 1.0/dy/dy*(
                    2.0 * s[IDX_mini(+i,j)] - s[IDX_mini(i,j+1)] - s[IDX_mini(i,j-1)]);
            // cout << v[IDX(i,j)] << endl;
            // cout << s[IDX(i,j)] << endl;
            
        }
    }

    int tag_send = 0;
    int tag_recv = tag_send;
    MPI_Request recvrequest;
    MPI_Request sendrequest;
    int requestcount = 0; 
    

    for (int i =beginx; i < endx-1; ++i) {
        if (D_rank != MPI_PROC_NULL ){

            MPI_Isend(&v[IDX_mini(i, 1)], endx, MPI_DOUBLE, D_rank, tag_send,Mycart_grid, &sendrequest);
            MPI_Isend(&s[IDX_mini(i, 1)], endx, MPI_DOUBLE, D_rank, tag_send,Mycart_grid, &sendrequest);
            MPI_Irecv(&v[IDX_mini(i,0)], endx,MPI_DOUBLE, D_rank, tag_recv, Mycart_grid, &recvrequest);
            MPI_Irecv(&s[IDX_mini(i,0)], endx,MPI_DOUBLE, D_rank, tag_recv, Mycart_grid, &recvrequest);
            // cout << "I just Sent this from the rank below: " << v[IDX_mini(i,1)] << endl;
            
            // cout << "I just Recieved this from the rank above: " << v[IDX_mini(i,Ey)] << endl; 

            // MPI_Sendrecv(&v[IDX_mini(i, 1)], 1, MPI_DOUBLE, D_rank, tag_send, &v[IDX_mini(i,Ey)], 1,
            //         MPI_DOUBLE, U_rank, tag_recv, Mycart_grid, MPI_STATUS_IGNORE ); // Sending stuff down 
        }
        
        if (U_rank != MPI_PROC_NULL){
            MPI_Isend(&v[IDX_mini(i, endy-1)], endx, MPI_DOUBLE, U_rank, tag_send,Mycart_grid, &sendrequest);
            MPI_Isend(&s[IDX_mini(i, endy-1)],endx, MPI_DOUBLE, U_rank, tag_send,Mycart_grid, &sendrequest);
            
            MPI_Irecv(&v[IDX_mini(i,endy)],endx, MPI_DOUBLE, U_rank, tag_recv, Mycart_grid, &recvrequest);
            MPI_Irecv(&s[IDX_mini(i,endy)],endx, MPI_DOUBLE, U_rank, tag_recv, Mycart_grid, &recvrequest);
        }
        
    }
    for (int j = beginy; j < endy-1; ++j) {
        if (L_rank != MPI_PROC_NULL ){

            MPI_Isend(&v[IDX_mini(1, j)], endy, MPI_DOUBLE, L_rank, tag_send,Mycart_grid, &sendrequest);
            MPI_Isend(&s[IDX_mini(1, j)], endy, MPI_DOUBLE, L_rank, tag_send,Mycart_grid, &sendrequest);
            MPI_Irecv( &v[IDX_mini(0,j)], endy, MPI_DOUBLE, L_rank, tag_recv, Mycart_grid, &recvrequest);
            MPI_Irecv( &s[IDX_mini(0,j)], endy, MPI_DOUBLE, L_rank, tag_recv, Mycart_grid, &recvrequest); 
        }

        if (R_rank != MPI_PROC_NULL ){
            MPI_Isend(&v[IDX_mini(endx-1, j)], endy, MPI_DOUBLE, R_rank, tag_send,Mycart_grid, &sendrequest);
            MPI_Isend(&s[IDX_mini(endx-1, j)], endy, MPI_DOUBLE, R_rank, tag_send,Mycart_grid, &sendrequest);


            
            MPI_Irecv(&v[IDX_mini(endx,j)], endy, MPI_DOUBLE, R_rank, tag_recv, Mycart_grid, &recvrequest);
            MPI_Irecv(&s[IDX_mini(endx,j)], endy, MPI_DOUBLE, R_rank, tag_recv, Mycart_grid, &recvrequest);
        }       
    }


    

    // for (int i = 0; i < requestcount; ++i){
    //     MPI_Wait(&sendrequest[i], MPI_STATUS_IGNORE);
    //     MPI_Wait(&recvrequest[i], MPI_STATUS_IGNORE);
    // }
    
  

    
    // int RecCount [1] = {E_x-B_x};
    // int disp[2] = {B_x,B_y};
    

    // Time advance vorticity
    for (int i = beginx; i < endx; ++i) {
        for (int j = beginy; j < endy; ++j) {
            vnew[IDX_mini(i,j)] = v[IDX_mini(i,j)] + dt*(
                ( (s[IDX_mini(i+1,j)] - s[IDX_mini(i-1,j)]) * 0.5 * dxi
                 *(v[IDX_mini(i,j+1)] - v[IDX_mini(i,j-1)]) * 0.5 * dyi)
              - ( (s[IDX_mini(i,j+1)] - s[IDX_mini(i,j-1)]) * 0.5 * dyi
                 *(v[IDX_mini(i+1,j)] - v[IDX_mini(i-1,j)]) * 0.5 * dxi)
              + nu * (v[IDX_mini(i+1,j)] - 2.0 * v[IDX_mini(i,j)] + v[IDX_mini(i-1,j)])*dx2i
              + nu * (v[IDX_mini(i,j+1)] - 2.0 * v[IDX_mini(i,j)] + v[IDX_mini(i,j-1)])*dy2i);

              cout << vnew[IDX_mini(i,j)] << "   " ;

            // MPI_Igather(&vnew[IDX_mini(i,j)],1, MPI_DOUBLE, &vgather[IDX(i+(B_x-1),j+(B_y-1))], (endx)*(endy), MPI_DOUBLE,0,Mycart_grid,&sendrequest);
            // MPI_Igather(&s[IDX_mini(i,j)],1, MPI_DOUBLE, &sgather[IDX(i+(B_x-1),j+(B_y-1))], (endx)*(endy), MPI_DOUBLE,0,Mycart_grid,&sendrequest);
            
        }
        cout << endl;

    }
    

        for (int i =beginx; i < endx-1; ++i) {
        if (D_rank != MPI_PROC_NULL ){

            MPI_Isend(&vnew[IDX_mini(i, 1)], endx, MPI_DOUBLE, D_rank, tag_send,Mycart_grid, &sendrequest);
            MPI_Irecv(&vnew[IDX_mini(i,0)], endx,MPI_DOUBLE, D_rank, tag_recv, Mycart_grid, &recvrequest);
            // cout << "I just Sent this from the rank below: " << v[IDX_mini(i,1)] << endl;
            
            // cout << "I just Recieved this from the rank above: " << v[IDX_mini(i,Ey)] << endl; 

            // MPI_Sendrecv(&v[IDX_mini(i, 1)], 1, MPI_DOUBLE, D_rank, tag_send, &v[IDX_mini(i,Ey)], 1,
            //         MPI_DOUBLE, U_rank, tag_recv, Mycart_grid, MPI_STATUS_IGNORE ); // Sending stuff down 
        }
        
        if (U_rank != MPI_PROC_NULL){
            MPI_Isend(&vnew[IDX_mini(i, endy-1)], endx, MPI_DOUBLE, U_rank, tag_send,Mycart_grid, &sendrequest);
            MPI_Irecv(&vnew[IDX_mini(i,endy)],endx, MPI_DOUBLE, U_rank, tag_recv, Mycart_grid, &recvrequest);
        }
        
        }
        for (int j = beginy; j < endy-1; ++j) {
            if (L_rank != MPI_PROC_NULL){

                MPI_Isend(&vnew[IDX_mini(1, j)], endy, MPI_DOUBLE, L_rank, tag_send,Mycart_grid, &sendrequest);
                MPI_Irecv(&vnew[IDX_mini(0,j)], endy, MPI_DOUBLE, L_rank, tag_recv, Mycart_grid, &recvrequest); 
            }
    
            if (R_rank != MPI_PROC_NULL){
                MPI_Isend(&vnew[IDX_mini(endx-1, j)], endy, MPI_DOUBLE, R_rank, tag_send,Mycart_grid, &sendrequest);
                MPI_Irecv(&vnew[IDX_mini(endx,j)], endy, MPI_DOUBLE, R_rank, tag_recv, Mycart_grid, &recvrequest);
            }       
        }



    
    /**
     * @brief Here the things that need to be sent to the other ranks 
    */

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

    cg->Solve(vnew, s);
    // // MPI_Scatter(&s,Nx*Ny, MPI_DOUBLE, &s, Nx*Ny, MPI_DOUBLE,0,Mycart_grid);
    // if (Myrank == 0 ){
    //     cg->Solve(vgather, s);
    //     // MPI_Scatter(&vgather,Nx*Ny, MPI_DOUBLE, &v, Nx*Ny, MPI_DOUBLE,0,Mycart_grid);
    //     MPI_Iscatter(&s,Nx*Ny, MPI_DOUBLE, &s, Nx*Ny, MPI_DOUBLE,0,Mycart_grid,&sendrequest);
    //     // MPI_Barrier(MPI_COMM_WORLD);
    // }
//     MPI_Scatter(&vgather,Nx*Ny, MPI_DOUBLE, &v, Nx*Ny, MPI_DOUBLE,0,Mycart_grid);
//     MPI_Scatter(&s,Nx*Ny, MPI_DOUBLE, &s, Nx*Ny, MPI_DOUBLE,0,Mycart_grid);
//     MPI_Barrier(MPI_COMM_WORLD);
}
