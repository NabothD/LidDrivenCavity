CXX = mpicxx -fopenmp
CXXFLAGS= -std=c++20 -Wall -O3
HDRS= LidDrivenCavity.h SolverCG.h mpi.h omp.h 
OBJS = LidDrivenCavity.o SolverCG.o LidDrivenCavitySolver.o
CONDOC = Doxyfile
LDLIBS = -lblas -lboost_program_options -lboost_timer -fopenmp


%.o : %.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -o $@ -c $<

unittests: LidDrivenCavity.o SolverCG.o UnitTest.o
	$(CXX) -o $@ $^ $(LDLIBS)

solver: $(OBJS)
	$(CXX) -o $@ $^ $(LDLIBS)

doc: $(CONDOC)
	@doxygen


all:solver


.PHONY: clean

clean:
	@echo "Clearing..."
	rm -f *.o solver unittest doc