CC           ?= gcc
CXX          ?= g++
NVCC         ?= nvcc
NVCC_FLAGS   ?= -O3 -std=c++20 -ccbin $(CC) -Xcompiler "-Wall -pthread"
NVCC_AVAILABLE ?=

SHARED   := ../../shared
ENGINE   := ../../engine
BCI      := ../../plugins/bci/include

INCLUDES := -I$(SHARED) -I$(ENGINE) -I$(BCI)
CXXFLAGS := -O3 -std=c++20 -Wall -pthread $(INCLUDES)

ENGINE_HEADERS := $(wildcard $(ENGINE)/*.hpp) $(ENGINE)/PhysicsGPU.cuh
SHARED_HEADERS := $(SHARED)/lpl_protocol.h $(SHARED)/Math.hpp

ifeq ($(NVCC_AVAILABLE),yes)
server: main.o PhysicsGPU.o
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o server main.o PhysicsGPU.o -lpthread -lm

main.o: main.cpp $(ENGINE_HEADERS) $(SHARED_HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -x cu -c main.cpp -o main.o

PhysicsGPU.o: $(ENGINE)/PhysicsGPU.cu $(ENGINE)/PhysicsGPU.cuh $(SHARED)/Math.hpp
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $(ENGINE)/PhysicsGPU.cu -o PhysicsGPU.o
else
server: main.cpp $(ENGINE_HEADERS) $(SHARED_HEADERS)
	$(CXX) $(CXXFLAGS) -o server main.cpp -lm
endif

clean:
	rm -f server *.o

.PHONY: server clean
