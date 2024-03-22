#include <iostream>
#include <fstream>
#define BOOST_TEST_MAIN
#include <sstream>
#include <vector>
using namespace std;
#define IDX(I,J) ((J)*9 + (I)) 
#include "LidDrivenCavity.h"
#include "SolverCG.h"

#define BOOST_TEST_MODULE SolverLidTester
#include <boost/test/included/unit_test.hpp>

/**
 * @file UnitTest.cpp 
 * @brief This .cpp file checks to ensure that both the Lid Driven Cavity and Solver CG class works corrrectly 
*/

struct MPIFixture {
    public:
        explicit MPIFixture() {
            argc = boost::unit_test::framework::master_test_suite().argc;
            argv = boost::unit_test::framework::master_test_suite().argv;
            cout << "Initialising MPI" << endl;
            MPI_Init(&argc, &argv);
        }

        ~MPIFixture() {
            cout << "Finalising MPI" << endl;
            MPI_Finalize();
        }

        int argc;
        char **argv;
};
BOOST_GLOBAL_FIXTURE(MPIFixture);


vector<double> splitLine(const string& line) {
    vector<double> columns;
    istringstream iss(line);
    double value;
    while (iss >> value) {
        columns.push_back(value);
    }
    return columns;
}

BOOST_AUTO_TEST_CASE(LidDrivenCavityTester){

    int rank = 0;
    int cart_rank = 0;
    int worldsize = 0;

    // Defining dome parameters that the MPI will depend on 
    int dims[2] = {0,0};
    int periods[2] = {0,0}; 
    int reorder =false; 
    int left_rank, right_rank, up_rank, down_rank, nrows, ncols;
    int coords[2];

    MPI_Comm_size(MPI_COMM_WORLD, &worldsize);  
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    
    // Beginning of descritisation of Nx and Ny
    // Defining the beginning and the ending of grid points as zeros for ch processors
    int begin_x = 0;  
    int begin_y = 0; 
    int end_x = 0; 
    int end_y = 0;

    nrows = (int)sqrt(worldsize);   // Number of rows in MPI grid
    ncols = (int)sqrt(worldsize);   // Number of columns in MPI grid

    int remain_x = 9 % ncols;  //Remainder if the Nx grid doesn't fit onto the mpi grid perefectly
    int remain_y = 9 % nrows;  //Remainder if the Ny grid doesn't fit onto the mpi grid perefectly
    int min_size_x  = (9 - remain_x)/ ncols;   // Minimum size of a chunk 
    int min_size_y  = (9 - remain_y)/ nrows;   // Minimum size of a chunk 
    
    
    MPI_Dims_create(worldsize, 2, dims);

    MPI_Comm cart_grid;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart_grid); // From this you get the new 
    
    MPI_Cart_coords(cart_grid, rank, 2, coords); // Getting coordinates of the current rank
    
    MPI_Cart_shift(cart_grid, 1, 1, &left_rank, &right_rank);    
    MPI_Cart_shift(cart_grid, 0, 1, &down_rank, &up_rank); 

    MPI_Cart_rank(cart_grid, coords, &cart_rank);
 
    /**
     * @brief The task of these if statements are to distribute the grid in an even way 
     * @param begin_x, begin_y tell each rank where their tarting point is in the x and y direction respectively 
     * @param end_x, end_y tell each rank where their ending point is in the x and y direction respectively   
    */


    if ( coords[1] < 9 % ncols) {
        min_size_x ++;
        begin_x = min_size_x * coords[1];
        end_x = min_size_x * (coords[1]+ 1);

    }
    else{
        begin_x = (min_size_x + 1) * remain_x + min_size_x * (coords[1] -remain_x);
        end_x = (min_size_x + 1) * remain_x + min_size_x * (coords[1] - remain_x +1);
    }


    if ( coords[0] < 9 % nrows) {
        min_size_y ++;
        begin_y = min_size_y * coords[0];
        end_y = min_size_y * (coords[0]+ 1);

    }
    else{
        begin_y = (min_size_y + 1) * remain_y + min_size_y * (coords[0] -remain_y);
        end_y = (min_size_y + 1) * remain_y + min_size_y * (coords[0] - remain_y +1);
    }

    LidDrivenCavity* solver1 = new LidDrivenCavity();
    solver1->RecieveDataForMPI(cart_grid, worldsize,cart_rank, coords, left_rank, right_rank, up_rank, down_rank, begin_x, begin_y, end_x, end_y );
    solver1->SetDomainSize(1.0, 1.0);
    solver1->SetGridSize(9, 9);
    solver1->SetTimeStep(0.01);
    solver1->SetFinalTime(1.0);
    solver1->SetReynoldsNumber(10);
    solver1->Initialise();
    solver1->Integrate();
    solver1->WriteSolution("finaltest.txt");
   
    
    ifstream file1("finaltest.txt", ios::binary); // This is the file produced by the test
    ifstream file2("ProperOne.txt", ios::binary); // This is the file the test compares to

    file1 >> noskipws;
    file2 >> noskipws;
    
   
    string line1, line2;
    while (getline(file1, line1) && getline(file2, line2)) {
        // Split lines into columns
        vector<double> columns1 = splitLine(line1);
        vector<double> columns2 = splitLine(line2);

        // Compare corresponding values with tolerance
        for (size_t i = 0; i < columns1.size(); ++i) {
            BOOST_CHECK_SMALL(columns1[i] - columns2[i], 1e-3); 
        }
    }
  
}

BOOST_AUTO_TEST_CASE(testSolverCGSolve) {
    int rank; 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == rank) {
        SolverCG* solver2 = nullptr;
        solver2 = new SolverCG(9, 9, 0.125, 0.125, 9, 9);
        solver2->RecieveCG(1, 1, 8, 8, MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL, 9, 9 , 0, 0 , MPI_COMM_WORLD); // Rank info
        

        double* v = new double[81]; // Assuming Nx*Ny = 81
        double* s = new double[81];
        
        const int k = 3;
        const int l = 3;
        for (int i = 0; i < 9; ++i) {
            for (int j = 0; j < 9; ++j) {
                v[IDX(i,j)] = -M_PI * M_PI * (k * k + l * l)
                                        * sin(M_PI * k * i * 0.125)
                                        * sin(M_PI * l * j * 0.125);
            }
        }
        
        solver2->Solve(v, s);
        
        ifstream filesol2("outputfile.txt", ios::binary); // This is the file the test compares to 

        string line2;
        
        vector<double> columns2 = splitLine(line2);


        // Compare corresponding values with tolerance
        for (size_t i = 0; i < columns2.size(); ++i) {
            BOOST_CHECK_SMALL(columns2[i] - v[i], 1e-9); 
            
        } 
    
    }      
}



