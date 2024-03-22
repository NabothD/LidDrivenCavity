#include <iostream>
#include <mpi.h>
#include <cmath>
using namespace std;
#include <boost/program_options.hpp>
namespace po = boost::program_options;
#include "LidDrivenCavity.h"

/**
 * @authors Naboth Dirirsa, Chris Cantwell
 * @date 22/03/2024
 * @brief This code solves the Lid Driven Cavity problem using parallel programming
 * I takes input from the user optianally, otherwise is set to default values
*/

int main(int argc, char **argv)
{
    int rank = 0;
    int cart_rank = 0;
    int worldsize = 0;
    
    po::options_description opts(
        "Solver for the 2D lid-driven cavity incompressible flow problem");
    opts.add_options()
        ("Lx",  po::value<double>()->default_value(1.0),
                 "Length of the domain in the x-direction.")
        ("Ly",  po::value<double>()->default_value(1.0),
                 "Length of the domain in the y-direction.")
        ("Nx",  po::value<int>()->default_value(9),
                 "Number of grid points in x-direction.")
        ("Ny",  po::value<int>()->default_value(9),
                 "Number of grid points in y-direction.")
        ("dt",  po::value<double>()->default_value(0.01),
                 "Time step size.")
        ("T",   po::value<double>()->default_value(1.0),
                 "Final time.")
        ("Re",  po::value<double>()->default_value(10),
                 "Reynolds number.")
        ("verbose",    "Be more verbose.")
        ("help",       "Print help message.");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, opts), vm);
    po::notify(vm);

    if (vm.count("help")) {
        cout << opts << endl;
        return 0;
    }
    // Defining dome parameters that the MPI will depend on 
    int dims[2] = {0,0};
    int periods[2] = {0,0}; 
    int reorder =false; 
    int left_rank, right_rank, up_rank, down_rank, nrows, ncols;
    int coords[2];

    // Initialise MPI.
    int err = MPI_Init(&argc, &argv);
    if (err != MPI_SUCCESS) {
        cout << "Failed to initialise MPI" << endl;
        return -1;
    }

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

    /**
     * @brief This if statement ensures that the number of proccesors inputted is a square number
    */


    if( nrows * nrows != worldsize){
        if (rank == 0){
            cout << "Error in Number of Processors" << endl 
            << "Number of proccessors must be a square number !" << endl
            << "Terminating the program .... " << endl;
        }        
        MPI_Finalize();
        return -1; 
    }

    int remain_x = vm["Nx"].as<int>() % ncols;  //Remainder if the Nx grid doesn't fit onto the mpi grid perefectly
    int remain_y = vm["Ny"].as<int>() % nrows;  //Remainder if the Ny grid doesn't fit onto the mpi grid perefectly
    int min_size_x  = (vm["Nx"].as<int>() - remain_x)/ ncols;   // Minimum size of a chunk 
    int min_size_y  = (vm["Ny"].as<int>() - remain_y)/ nrows;   // Minimum size of a chunk 
    
    
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


    if ( coords[1] < vm["Nx"].as<int>() % ncols) {
        min_size_x ++;
        begin_x = min_size_x * coords[1];
        end_x = min_size_x * (coords[1]+ 1);

    }
    else{
        begin_x = (min_size_x + 1) * remain_x + min_size_x * (coords[1] -remain_x);
        end_x = (min_size_x + 1) * remain_x + min_size_x * (coords[1] - remain_x +1);
    }


    if ( coords[0] < vm["Ny"].as<int>() % nrows) {
        min_size_y ++;
        begin_y = min_size_y * coords[0];
        end_y = min_size_y * (coords[0]+ 1);

    }
    else{
        begin_y = (min_size_y + 1) * remain_y + min_size_y * (coords[0] -remain_y);
        end_y = (min_size_y + 1) * remain_y + min_size_y * (coords[0] - remain_y +1);
    }
 
    LidDrivenCavity* solver = new LidDrivenCavity();
    solver->RecieveDataForMPI(cart_grid, worldsize,cart_rank, coords, left_rank, right_rank, up_rank, down_rank, begin_x, begin_y, end_x, end_y );
    solver->SetDomainSize(vm["Lx"].as<double>(), vm["Ly"].as<double>());
    solver->SetGridSize(vm["Nx"].as<int>(),vm["Ny"].as<int>());  
    solver->SetTimeStep(vm["dt"].as<double>());
    solver->SetFinalTime(vm["T"].as<double>());
    solver->SetReynoldsNumber(vm["Re"].as<double>());
    solver->PrintConfiguration();
    solver->Initialise();
    solver->WriteSolution("ic.txt");
    solver->Integrate();
    solver->WriteSolution("final.txt");   
    MPI_Finalize();
	return 0;
    
    
}
