KDIR   := /lib/modules/$(shell uname -r)/build

all: driver

driver:
	@mkdir -p /tmp/lpl_kernel_build
	@cp lpl_kmod.c Kbuild lpl_protocol.h /tmp/lpl_kernel_build/
	@$(MAKE) -C $(KDIR) M=/tmp/lpl_kernel_build modules
	@cp /tmp/lpl_kernel_build/lpl_kmod.ko .
	@rm -rf /tmp/lpl_kernel_build

clean:
	@rm -f lpl_kmod.ko
	@rm -rf /tmp/lpl_kernel_build

.PHONY: all driver clean
