# Makefile
include sources.mk

# defaults
SIM ?= verilator



TOPLEVEL_LANG ?=verilog
TEST_RUNNER   ?=runner
MODULE        ?=runner
TOPLEVEL      ?=top_wrapper_tb
INCLUDE_DIR = $(WORKAREA)/hw/build/ip/include/
# VERILATOR_FLAGS += $(addprefix -I, $(VERILOG_INCLUDE_DIRS)) -I$(INCLUDE_DIR)
# VERILATOR_FLAGS += -I$(INCLUDE_DIR)
EXTRA_ARGS +=  -I$(INCLUDE_DIR)
EXTRA_ARGS += +define+SIMULATION



include $(shell cocotb-config --makefiles)/Makefile.sim
# source $(WORKAREA)/scripts/add_hash.sh

