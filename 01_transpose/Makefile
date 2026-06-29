# NanoKernels 01_transpose
# Build CPU + CUDA, run all versions (CPU, CUDA, Triton vs torch).
#
# Usage (on remote A100 node):
#   make            # build cpu + cuda
#   make run        # build + run cpu, cuda, triton
#
COMMON := ../common
NVCC   ?= nvcc
ARCH   ?= sm_80
CXX    ?= g++

CUDA_BIN := cuda/transpose_cuda
CPU_BIN  := cpu/transpose_cpu

ROWS ?= 4096
COLS ?= 4096

# Workaround: Triton parses `/sbin/ldconfig -p` with .decode() (utf-8); on this
# node the cache has non-utf8 bytes. Point Triton straight at libcuda to skip it.
TRITON_LIBCUDA_PATH ?= /usr/lib/x86_64-linux-gnu
export TRITON_LIBCUDA_PATH

.PHONY: all run cpu cuda triton clean

all: $(CPU_BIN) $(CUDA_BIN)

$(CPU_BIN): cpu/transpose_cpu.cpp
	$(CXX) -O3 -fopenmp -march=native $< -o $@

$(CUDA_BIN): cuda/transpose.cu
	$(NVCC) -O3 -arch=$(ARCH) -I$(COMMON) $< -o $@

cpu: $(CPU_BIN)
	./$(CPU_BIN) $(ROWS) $(COLS)

cuda: $(CUDA_BIN)
	./$(CUDA_BIN) $(ROWS) $(COLS)

triton:
	python triton/transpose_triton.py

run: all
	@echo "==== CPU ===="    && ./$(CPU_BIN) $(ROWS) $(COLS)
	@echo "==== CUDA ===="   && ./$(CUDA_BIN) $(ROWS) $(COLS)
	@echo "==== Triton ====" && python triton/transpose_triton.py

clean:
	rm -f $(CPU_BIN) $(CUDA_BIN) *.ncu-rep *.nsys-rep
