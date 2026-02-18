#===============
# options
#===============
# path to cuda
CUDA_PATH ?= /usr/local/cuda

# path to hip
HIP_PATH ?= /opt/rocm

# GPU Backend selection (cuda, hip, or auto)
BACKEND ?= auto

# Compute Capability (e.g., 90 for NVIDIA H200, gfx940 for AMD MI300X, or auto)
GPU_ARCH ?= auto

# "ozIMMU_EF := yes" if you use the ozIMMU_EF (accelerated Ozaki scheme 1) in a sample code.
ozIMMU_EF ?= no

# path to ozIMMU_EF (Set this if "ozIMMU_EF := yes")
ozIMMU_EF_DIR ?= ../../ozIMMU_EF

# "cuMpSGEMM := yes" if you use the cuMpSGEMM in a sample code and compute_cap < 100.
cuMpSGEMM ?= no

# path to cuMpSGEMM (Set this if "cuMpSGEMM := yes")
cuMpSGEMM_DIR ?= ../../cuMpSGEMM

# "HIJACK := yes" if you are hijacking cublasGemmEx with emulation
HIJACK ?= no


#===============
# Auto-detect backend (CUDA or HIP)
#===============
ifeq ($(BACKEND),auto)
ifneq ($(shell command -v nvidia-smi 2>/dev/null),)
BACKEND := cuda
else ifneq ($(shell command -v rocminfo 2>/dev/null),)
BACKEND := hip
else
$(error Neither NVIDIA (CUDA) nor AMD (ROCm) GPU environment detected!)
endif
endif


#===============
# CUDA setup
#===============
ifeq ($(BACKEND),cuda)

# Auto-detect arch
ifeq ($(GPU_ARCH),auto)
GPU_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '.')
endif

# get memory size (fallback to 0 when NVML query is unavailable)
GPU_MEM_MB := $(shell nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -dc '0-9')
ifeq ($(GPU_MEM_MB),)
GPU_MEM_MB := 0
endif

# path
export PATH := $(CUDA_PATH)/bin:$(PATH)
export LD_LIBRARY_PATH := $(CUDA_PATH)/lib64:$(LD_LIBRARY_PATH)

# compiler & flags
COMPILER := nvcc
LIBS := -lcublas -lcudart -lcuda -lnvidia-ml -ldl ../lib/libgemmul8.a
FLAGS := -std=c++20 -O3 -I../include
FLAGS += -DGPU_MEM_MB=$(GPU_MEM_MB)
ARCH := -gencode arch=compute_$(GPU_ARCH),code=sm_$(GPU_ARCH)

# external library
LD :=
ifneq ($(filter $(GPU_ARCH), 80 86 87 89 90),)
else
    cuMpSGEMM := no
endif
ifeq ($(ozIMMU_EF),yes)
FLAGS += -DozIMMU_EF_FLAG -I$(ozIMMU_EF_DIR)/include
LIBS += -L$(ozIMMU_EF_DIR)/build -lozimmu
LD := $(ozIMMU_EF_DIR)/build/libozimmu.so:$(LD)
endif
ifeq ($(cuMpSGEMM),yes)
FLAGS += -DcuMpSGEMM_FLAG -I$(cuMpSGEMM_DIR)/include
LIBS += -L$(cuMpSGEMM_DIR)/build -lcumpsgemm
ifneq ($(LD),)
LD := $(LD):$(cuMpSGEMM_DIR)/build/libcumpsgemm.so
else
LD := $(cuMpSGEMM_DIR)/build/libcumpsgemm.so
endif
endif

endif


#===============
# HIP setup
#===============
ifeq ($(BACKEND),hip)

# Auto-detect arch
ifeq ($(GPU_ARCH),auto)
GPU_ARCH := $(shell amd-smi static --asic --csv | grep -o 'gfx[0-9]\+' | head -n 1)
endif

# path
export PATH := $(HIP_PATH)/bin:$(PATH)
export LD_LIBRARY_PATH := $(HIP_PATH)/lib:$(LD_LIBRARY_PATH)
GPU_MEM_MB := $(shell amd-smi static --vram --csv | awk -F, "NR==2 {print \$$4}")

# compiler & flags
COMPILER := hipcc
LIBS := -lamd_smi -lamdhip64 -lhipblas -ldl -Wl,../lib/libgemmul8.a
FLAGS := -std=c++20 -O3 -I../include
FLAGS += -ffp-contract=off
FLAGS += -Wno-unused-result -Wno-unused-command-line-argument
FLAGS += -DOCML_BASIC_ROUNDED_OPERATIONS
FLAGS += -DGPU_MEM_MB=$(GPU_MEM_MB)
ARCH := --offload-arch=$(GPU_ARCH)

# external library
cuMpSGEMM := no
ozIMMU_EF := no
LD :=

endif


#===============
# Hijack
#===============
ifeq ($(HIJACK),yes)
LD+=../lib/libgemmul8.so
endif


#===============
# Compile
#===============
TARGET_d := test_double
TARGET_f := test_float
TARGET_z := test_double_complex
TARGET_c := test_float_complex
TARGET_i := test_int32
TARGET_ib := test_int32_bench

TARGETS := $(TARGET_d) $(TARGET_f) $(TARGET_z) $(TARGET_c)
ifeq ($(BACKEND),cuda)
TARGETS += $(TARGET_i) $(TARGET_ib)
endif

all: INFO $(TARGETS) VERSION

INFO:
	$(info BACKEND      : $(BACKEND))
ifeq ($(BACKEND),cuda)
	$(info CUDA_PATH    : $(CUDA_PATH))
endif
ifeq ($(BACKEND),hip)
	$(info HIP_PATH     : $(HIP_PATH))
endif
	$(info GPU_ARCH     : $(GPU_ARCH))
	$(info COMPILER     : $(COMPILER))
	$(info GPU_MEM [MB] : $(GPU_MEM_MB))
	$(info ozIMMU_EF    : $(ozIMMU_EF))
	$(info ozIMMU_EF_DIR: $(ozIMMU_EF_DIR))
	$(info cuMpSGEMM    : $(cuMpSGEMM))
	$(info cuMpSGEMM_DIR: $(cuMpSGEMM_DIR))
	$(info HIJACK       : $(HIJACK))

$(TARGET_d): $(TARGET_d).cu
	$(COMPILER) $< $(FLAGS) $(ARCH) -o $@ $(LIBS)

$(TARGET_f): $(TARGET_f).cu
	$(COMPILER) $< $(FLAGS) $(ARCH) -o $@ $(LIBS)

$(TARGET_z): $(TARGET_z).cu
	$(COMPILER) $< $(FLAGS) $(ARCH) -o $@ $(LIBS)

$(TARGET_c): $(TARGET_c).cu
	$(COMPILER) $< $(FLAGS) $(ARCH) -o $@ $(LIBS)

ifeq ($(BACKEND),cuda)
$(TARGET_i): $(TARGET_i).cu
	$(COMPILER) $< $(FLAGS) $(ARCH) -o $@ $(LIBS)

$(TARGET_ib): $(TARGET_ib).cu
	$(COMPILER) $< $(FLAGS) $(ARCH) -o $@ $(LIBS)
endif

VERSION:
	$(COMPILER) --version 2>&1 | tee $(COMPILER)_version

test_s:
	LD_PRELOAD=$(LD) ./$(TARGET_f) $(MODE)

test_d:
	LD_PRELOAD=$(LD) ./$(TARGET_d) $(MODE)

test_z:
	LD_PRELOAD=$(LD) ./$(TARGET_z) $(MODE)

test_c:
	LD_PRELOAD=$(LD) ./$(TARGET_c) $(MODE)

ifeq ($(BACKEND),cuda)
test_i32:
	LD_PRELOAD=$(LD) ./$(TARGET_i)

bench_i32:
	LD_PRELOAD=$(LD) ./$(TARGET_ib) $(MODE)
endif

clean:
	rm -f *.o
	rm -f $(TARGET_d) $(TARGET_f) $(TARGET_z) $(TARGET_c) $(TARGET_i) $(TARGET_ib)
