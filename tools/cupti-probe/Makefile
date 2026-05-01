# cupti-probe — Makefile
#
# Build:
#   make
#
# Build with a specific CUDA toolkit (default looks at /usr/local/cuda):
#   make CUDA_HOME=/usr/local/cuda-13.0
#
# Clean:
#   make clean

CUDA_HOME ?= /usr/local/cuda
NVCC      ?= $(CUDA_HOME)/bin/nvcc
CUPTI_LIB ?= $(CUDA_HOME)/extras/CUPTI/lib64

CXXSTD    := c++17
OPT       := -O2
ARCHFLAG  ?=

# Sources
SRC_DIR   := src
SRCS      := $(SRC_DIR)/main.cu $(SRC_DIR)/probes.cu $(SRC_DIR)/profiler_probe.cu $(SRC_DIR)/range_profiler_probe.cu
HEADERS   := $(SRC_DIR)/probe_common.h $(SRC_DIR)/probes.h

BIN       := cupti-probe

# CUPTI lib for Activity API + nvperf libs for Profiler API.
# nvperf libs ship under $CUDA_HOME/extras/CUPTI/lib64 alongside libcupti.
# nvperf_target needed for NVPW_CounterData_* host-side queries.
LIBS      := -L$(CUPTI_LIB) -lcudart -lcuda -lcupti -lnvperf_host -lnvperf_target

.PHONY: all clean

all: $(BIN)

$(BIN): $(SRCS) $(HEADERS)
	$(NVCC) $(OPT) -std=$(CXXSTD) $(ARCHFLAG) $(SRCS) -o $(BIN) $(LIBS)

clean:
	rm -f $(BIN) cupti_probe_*_results.json
