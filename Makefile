obj-m += lpl_kmod.o

KDIR := /lib/modules/$(shell uname -r)/build

CC = gcc
CXX = g++
NVCC = nvcc
NVCC_FLAGS = -O3 -std=c++20 -ccbin $(CC) -Xcompiler "-Wall -pthread"

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

# --- Engine (NIC → GPU pipeline) ---
engine: main.o Engine.o
	$(NVCC) $(NVCC_FLAGS) -o engine main.o Engine.o -lpthread -lm

main.o: main.cpp Engine.cuh lpl_protocol.h WorldPartition.hpp Partition.hpp \
        FlatAtomicsHashMap.hpp EntityRegistry.hpp Math.hpp Morton.hpp SpinLock.hpp \
        PinnedAllocator.hpp FlatDynamicOctree.hpp
	$(NVCC) $(NVCC_FLAGS) -x cu -c main.cpp -o main.o

Engine.o: Engine.cu Engine.cuh lpl_protocol.h WorldPartition.hpp Partition.hpp \
        FlatAtomicsHashMap.hpp EntityRegistry.hpp Math.hpp Morton.hpp SpinLock.hpp \
        PinnedAllocator.hpp FlatDynamicOctree.hpp
	$(NVCC) $(NVCC_FLAGS) -c Engine.cu -o Engine.o

# --- WorldPartition standalone demo (CPU-only, no CUDA) ---
server: server.cpp WorldPartition.hpp Partition.hpp FlatAtomicsHashMap.hpp \
        Math.hpp Morton.hpp SpinLock.hpp PinnedAllocator.hpp FlatDynamicOctree.hpp \
        EntityRegistry.hpp
	$(CXX) -O3 -std=c++20 -Wall -pthread -o server server.cpp

# ============================================================
#  Utils
# ============================================================

clean:
	rm -f /tmp/lpl_kernel_build 2>/dev/null; true
	rm -f *.o engine server *.ko

install:
	sudo insmod lpl_kmod.ko
	sudo chmod 666 /dev/lpl_driver

uninstall:
	sudo rmmod lpl_kmod

run:
	./engine

logs:
	dmesg | tail -20

.PHONY: all driver engine server clean install uninstall run logs
