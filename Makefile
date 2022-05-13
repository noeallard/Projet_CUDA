CXX=g++
CXXFLAGS=-O3 -march=native
LDLIBS=`pkg-config --libs opencv`


.PHONY: clean

convolution-cpp: convolution.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

convolution-cu: convolution.cu
	nvcc -o $@ $< $(LDLIBS)

clean:
	rm -f convolution-cu
