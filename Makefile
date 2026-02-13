obj-m += lpl_kmod.o

KDIR := /lib/modules/$(shell uname -r)/build

CC = gcc
CXX = g++
NVCC = nvcc
NVCC_FLAGS = -O3 -std=c++20 -ccbin $(CC) -Xcompiler "-Wall -pthread"
NVCC_AVAILABLE := $(shell command -v $(NVCC) 2>/dev/null)

# ============================================================
#  Common header dependencies
# ============================================================

WORLD_HEADERS = WorldPartition.hpp Partition.hpp FlatAtomicsHashMap.hpp \
                EntityRegistry.hpp Math.hpp Morton.hpp SpinLock.hpp \
                PinnedAllocator.hpp FlatDynamicOctree.hpp

# ============================================================
#  Targets
# ============================================================

all: driver engine

# --- Kernel Module ---
driver:
	@mkdir -p /tmp/lpl_kernel_build
	@cp lpl_kmod.c lpl_protocol.h /tmp/lpl_kernel_build/
	@echo 'obj-m += lpl_kmod.o' > /tmp/lpl_kernel_build/Makefile
	@$(MAKE) -C $(KDIR) M=/tmp/lpl_kernel_build modules
	@cp /tmp/lpl_kernel_build/lpl_kmod.ko .
	@rm -rf /tmp/lpl_kernel_build

# --- Engine (NIC → GPU pipeline with ECS SystemScheduler) ---
ifeq ($(NVCC_AVAILABLE),)
engine: main.o NetworkDispatch.o
	$(CXX) -O3 -std=c++20 -Wall -pthread -o engine main.o NetworkDispatch.o -lpthread -lm

main.o: main.cpp PhysicsGPU.cuh NetworkDispatch.hpp SystemScheduler.hpp \
        lpl_protocol.h $(WORLD_HEADERS)
	$(CXX) -O3 -std=c++20 -Wall -pthread -c main.cpp -o main.o

NetworkDispatch.o: NetworkDispatch.cpp NetworkDispatch.hpp lpl_protocol.h \
        $(WORLD_HEADERS) Math.hpp PhysicsGPU.cuh
	$(CXX) -O3 -std=c++20 -Wall -pthread -c NetworkDispatch.cpp -o NetworkDispatch.o
else
engine: main.o PhysicsGPU.o NetworkDispatch.o
	$(NVCC) $(NVCC_FLAGS) -o engine main.o PhysicsGPU.o NetworkDispatch.o -lpthread -lm

main.o: main.cpp PhysicsGPU.cuh NetworkDispatch.hpp SystemScheduler.hpp \
        lpl_protocol.h $(WORLD_HEADERS)
	$(NVCC) $(NVCC_FLAGS) -x cu -c main.cpp -o main.o

PhysicsGPU.o: PhysicsGPU.cu PhysicsGPU.cuh Math.hpp
	$(NVCC) $(NVCC_FLAGS) -c PhysicsGPU.cu -o PhysicsGPU.o

NetworkDispatch.o: NetworkDispatch.cpp NetworkDispatch.hpp lpl_protocol.h \
        $(WORLD_HEADERS) Math.hpp PhysicsGPU.cuh
	$(NVCC) $(NVCC_FLAGS) -x cu -c NetworkDispatch.cpp -o NetworkDispatch.o
endif

# --- WorldPartition standalone demo (CPU-only, no CUDA) ---
server_demo: server.cpp $(WORLD_HEADERS)
	$(CXX) -O3 -std=c++20 -Wall -pthread -o server_demo server.cpp

# --- Benchmark (CPU performance testing) ---
benchmark: benchmark.cpp $(WORLD_HEADERS)
	$(CXX) -O3 -march=native -std=c++20 -Wall -pthread -o benchmark benchmark.cpp

# --- Visual Demo (terminal-based, no dependencies) ---
visual: visual.cpp $(WORLD_HEADERS)
	$(CXX) -O3 -std=c++20 -Wall -pthread -o visual visual.cpp

# --- Visual 3D Client (connects to engine server via UDP) ---
visual3d: visual3d.cpp lpl_protocol.h $(WORLD_HEADERS)
	$(CXX) -O3 -std=c++20 -Wall -pthread -o visual3d visual3d.cpp -lglfw -lGLEW -lGL -lm

client: visual3d

# ============================================================
#  Utils
# ============================================================

clean:
	rm -f /tmp/lpl_kernel_build 2>/dev/null; true
	rm -f *.o engine server_demo benchmark visual visual3d *.ko

install:
	sudo insmod lpl_kmod.ko
	sudo chmod 666 /dev/lpl_driver

uninstall:
	sudo rmmod lpl_kmod

run:
	./engine

logs:
	dmesg | tail -20

.PHONY: all driver engine server_demo client clean install uninstall run logs
