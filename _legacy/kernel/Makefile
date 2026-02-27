KDIR   := /lib/modules/$(shell uname -r)/build
SHARED := ../shared

all: driver

driver:
	@mkdir -p /tmp/lpl_kernel_build
	@cp lpl_kmod.c $(SHARED)/lpl_protocol.h /tmp/lpl_kernel_build/
	@echo 'obj-m += lpl_kmod.o' > /tmp/lpl_kernel_build/Makefile
	@$(MAKE) -C $(KDIR) M=/tmp/lpl_kernel_build modules
	@cp /tmp/lpl_kernel_build/lpl_kmod.ko .
	@rm -rf /tmp/lpl_kernel_build

clean:
	rm -f lpl_kmod.ko

.PHONY: all driver clean
