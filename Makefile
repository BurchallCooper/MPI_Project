
CXX      = ~/openmpi/bin/mpic++
EXX	 = ~/openmpi/bin/mpiexec
IFILE 	 =  test.32x32.txt
NPROC	 = 4	
#CXX      = mpic++ 
CXXFLAGS = -Wall -g

all:	main.cpp  Complex.cc InputImage.cc
	$(CXX)  -g -o fft2d main.cpp Complex.cc InputImage.cc

run: 	
	$(EXX) -n $(NPROC) fft2d $(IFILE)

clean:
	@rm  fft2d *.wbctxt
	

