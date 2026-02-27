# =============================================================
#  LAPLACE PLUGIN — Root Orchestrator
#  Structure: shared/ kernel/ engine/ plugins/ apps/
# =============================================================

CC   = gcc
CXX  = g++
NVCC = nvcc
NVCC_FLAGS = -O3 -std=c++20 -ccbin $(CC) -Xcompiler "-Wall -pthread"

# --- Detect NVCC ---
NVCC_PATH := $(shell command -v $(NVCC) 2>/dev/null)
NVCC_AVAILABLE :=
ifneq ($(NVCC_PATH),)
    NVCC_TEST_SRC := $(shell echo 'int main(){return 0;}' > .nvcc_test.cu)
    NVCC_TEST_OUT := $(shell $(NVCC) -ccbin $(CC) -x cu .nvcc_test.cu -o .nvcc_test.o >/dev/null 2>&1 && echo "yes")
    NVCC_CLEANUP  := $(shell rm -f .nvcc_test.cu .nvcc_test.o)
    ifeq ($(NVCC_TEST_OUT),yes)
        NVCC_AVAILABLE := yes
    endif
endif

export CC CXX NVCC NVCC_FLAGS NVCC_AVAILABLE

# =============================================================
#  Targets
# =============================================================

.PHONY: all driver server benchmark visual client test clean install uninstall run logs

all: driver server

driver:
	$(MAKE) -C kernel

server:
	$(MAKE) -C apps/server

benchmark:
	$(MAKE) -C apps/benchmark

visual client:
	$(MAKE) -C apps/client visual

android:
	$(MAKE) -C apps/client android

test:
	cd plugins/bci && xmake config --with_brainflow=n && xmake run lpl-bci-tests

clean:
	$(MAKE) -C kernel clean
	$(MAKE) -C apps/server clean
	$(MAKE) -C apps/benchmark clean
	$(MAKE) -C apps/client clean
	cd plugins/bci && xmake clean

install:
	sudo insmod kernel/lpl_kmod.ko
	sudo chmod 666 /dev/lpl_driver

uninstall:
	sudo rmmod lpl_kmod

run:
	./apps/server/server

logs:
	sudo dmesg | tail -20
