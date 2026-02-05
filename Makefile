obj-m += lpl_kmod.o

KDIR := /lib/modules/$(shell uname -r)/build
PWD := $(shell pwd)

CC = gcc-12
NVCC = nvcc

CFLAGS = -O3 -pthread -Wall
NVCC_FLAGS = -O3 -ccbin $(CC)

all: driver app

driver:
	@mkdir -p /tmp/lpl_kernel_build
	@cp lpl_kmod.c plugin.h /tmp/lpl_kernel_build/
	@echo 'obj-m += lpl_kmod.o' > /tmp/lpl_kernel_build/Makefile
	@$(MAKE) -C $(KDIR) M=/tmp/lpl_kernel_build modules
	@cp /tmp/lpl_kernel_build/lpl_kmod.ko .
	@rm -rf /tmp/lpl_kernel_build

app: main.o plugin.o
	$(NVCC) $(NVCC_FLAGS) -o engine main.o plugin.o -lpthread -lm

main.o: main.c plugin.h
	$(CC) $(CFLAGS) -c main.c -o main.o

plugin.o: plugin.cu plugin.h
	$(NVCC) $(NVCC_FLAGS) -c plugin.cu -o plugin.o

clean:
	rm -f /tmp/lpl_kernel_build 2>/dev/null; true
	rm -f *.o engine *.ko

install:
	sudo insmod lpl_kmod.ko

uninstall:
	sudo rmmod lpl_kmod

logs:
	dmesg | tail -20
