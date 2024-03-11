#include <iostream>
#include <algorithm>
#include <cstring>
#include <fstream>
#define BOOST_TEST_MAIN
#include <sstream>
#include <vector>
using namespace std;

#include "LidDrivenCavity.h"

#define BOOST_TEST_MODULE SolverLidTester
#include <boost/test/included/unit_test.hpp>



std::vector<double> splitLine(const std::string& line) {
    std::vector<double> columns;
    std::istringstream iss(line);
    double value;
    while (iss >> value) {
        columns.push_back(value);
    }
    return columns;
}

BOOST_AUTO_TEST_CASE(Solver){
    LidDrivenCavity* solver1 = new LidDrivenCavity();
    solver1->SetDomainSize(1.0, 1.0);
    solver1->SetGridSize(9, 9);
    solver1->SetTimeStep(0.01);
    solver1->SetFinalTime(1.0);
    solver1->SetReynoldsNumber(10);

    solver1->PrintConfiguration();
    solver1->Initialise();
    solver1->WriteSolution("ictest.txt");
    solver1->Integrate();
    solver1->WriteSolution("finaltest.txt");
   
    
    ifstream file1("finaltest.txt", ios::binary);
    ifstream file2("finalTrue.txt", ios::binary);

    file1 >> noskipws;
    file2 >> noskipws;
    
   
    std::string line1, line2;
    while (std::getline(file1, line1) && std::getline(file2, line2)) {
        // Split lines into columns
        std::vector<double> columns1 = splitLine(line1);
        std::vector<double> columns2 = splitLine(line2);

        // Compare corresponding values with tolerance
        for (size_t i = 0; i < columns1.size(); ++i) {
            BOOST_CHECK_SMALL(columns1[i] - columns2[i], 1e-3); 
        }
    }
  
}

