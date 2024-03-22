#include <string>
#include <mpi.h>
using namespace std;


class SolverCG;

class LidDrivenCavity
{
public:
    LidDrivenCavity();
    ~LidDrivenCavity();
    void RecieveDataForMPI(MPI_Comm mycart_grid,int size ,int myrank, int mycoords[2], int l_rank, int r_rank, 
    int u_rank, int d_rank, int b_x, int b_y, int e_x, int e_y );
    void SetDomainSize(double xlen, double ylen);
    void SetGridSize(int nx, int ny);
    void SetTimeStep(double deltat);
    void SetFinalTime(double finalt);
    void SetReynoldsNumber(double Re);

    void Initialise();
    void Integrate();
    void WriteSolution(std::string file);
    void PrintConfiguration();

private:
    int Myrank,AllSize, L_rank, R_rank, D_rank, U_rank, B_x, B_y, E_x, E_y;
    MPI_Comm Mycart_grid;
    int* Mycoords = nullptr;
    double* v   = nullptr;
    double* u0_gather = nullptr;
    double* u1_gather = nullptr;
    double* v_gather = nullptr;
    double* s_gather = nullptr;
    double* vsum   = nullptr;
    double* s   = nullptr;
    double* tmp = nullptr;
    double* vnew = nullptr;
    int Ex, Ey, beginx, beginy, endx, endy;

    double dt   = 0.01;
    double T    = 1.0;
    double dx;
    double dy;
    int    Nx   = 9;
    int    Ny   = 9;
    int    Npts = 81;
    double Lx   = 1.0;
    double Ly   = 1.0;
    double Re   = 10;
    double U    = 1.0;
    double nu   = 0.1;

    SolverCG* cg = nullptr;

    void CleanUp();
    void UpdateDxDy();
    void Advance();
};

