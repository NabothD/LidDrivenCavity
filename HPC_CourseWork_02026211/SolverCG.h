#pragma once
#include <mpi.h>

class SolverCG
{
public:
    SolverCG(int pNx, int pNy, double pdx, double pdy, int g_Nx, int g_Ny);
    ~SolverCG();
    void RecieveCG(int bx, int by, int ex, int ey,int l_rank ,int r_rank, int d_rank, int u_rank ,int g_ex, int g_ey ,int g_bx, int g_by , MPI_Comm mygrid);
    void Solve(double* b, double* x);
    

private:
    int Bx, G_ex, G_bx;
    int By, G_ey, G_by;
    int Ex, G_Nx;
    int Ey, G_Ny; 
    int Lrank;
    int Rrank;
    int  Drank;
    int Urank; 
    MPI_Comm Mygrid;
    double dx;
    double dy;
    int Nx;
    int Ny;
    int loc_endx ;
    int loc_endy ;
    double* r;
    double* p;
    double* z;
    double* t;

    void ApplyOperator(double* p, double* t);
    void Precondition(double* p, double* t);
    void ImposeBC(double* p);

};

