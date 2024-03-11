CXX = g++ 
CXXFLAGS= -std=c++20 -Wall -O2
HDRS= LidDrivenCavity.h SolverCG.h
OBJS = LidDrivenCavity.o SolverCG.o LidDrivenCavitySolver.o
LDLIBS = -lblas -lboost_program_options

%.o : %.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -o $@ -c $<

unittest: LidDrivenCavity.o SolverCG.o UnitTest.o
	$(CXX) -o $@ $^ $(LDLIBS)

solver: $(OBJS)
	$(CXX) -o $@ $^ $(LDLIBS)


all:solver


doc: 
	@doxygen ./Doxyfile

.PHONY: clean

clean:
	@echo "Clearing..."
	rm -f *.o solver unittest