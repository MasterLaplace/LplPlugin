CC  ?= gcc
CXX ?= g++

SHARED   := ../../shared
ENGINE   := ../../engine

INCLUDES := -I$(SHARED) -I$(ENGINE)
CXXFLAGS := -O3 -march=native -std=c++20 -Wall -pthread $(INCLUDES)

ENGINE_HEADERS := $(wildcard $(ENGINE)/*.hpp) $(ENGINE)/PhysicsGPU.cuh
SHARED_HEADERS := $(SHARED)/lpl_protocol.h $(SHARED)/Math.hpp

benchmark: benchmark.cpp $(ENGINE_HEADERS) $(SHARED_HEADERS)
	$(CXX) $(CXXFLAGS) -o benchmark benchmark.cpp

clean:
	rm -f benchmark *.o

.PHONY: benchmark clean
