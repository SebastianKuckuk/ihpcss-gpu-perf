# helper function

GPU   = ${shell (nvidia-smi --query-gpu=name --format=csv,noheader | cut -d " " -f 2)}


# configuration

TEST_CASE   = fma
BUILD_DIR   = ../../build/$(TEST_CASE)
PROFILE_DIR = ../../profiles/$(GPU)/$(TEST_CASE)


# default benchmark parameters

NX    = $$((1024 * 1024))
NWARM = 2
NIT   = 10

PARAMETERS = ${NX} ${NWARM} ${NIT}

TAG   = ${NX}-${NIT}

METRICS = gpu__time_duration.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_second


# all

targets = \
    $(BUILD_DIR)/$(TEST_CASE)-base \
    $(BUILD_DIR)/$(TEST_CASE)-omp-host \
    $(BUILD_DIR)/$(TEST_CASE)-cuda \
    $(BUILD_DIR)/$(TEST_CASE)-omp-target \


.PHONY: all
all: mk-target-dir $(targets)


mk-target-dir:
	@mkdir -p $(BUILD_DIR)


mk-profile-dir:
	@mkdir -p $(PROFILE_DIR)


# build rules

$(BUILD_DIR)/$(TEST_CASE)-base: $(TEST_CASE)-base.cpp $(TEST_CASE)-util.h ../util.h
	g++ -O3 -march=native -std=c++17 -o $(BUILD_DIR)/$(TEST_CASE)-base $(TEST_CASE)-base.cpp

$(BUILD_DIR)/$(TEST_CASE)-omp-host: $(TEST_CASE)-omp-host.cpp $(TEST_CASE)-util.h ../util.h
	g++ -O3 -march=native -std=c++17 -fopenmp -o $(BUILD_DIR)/$(TEST_CASE)-omp-host $(TEST_CASE)-omp-host.cpp

$(BUILD_DIR)/$(TEST_CASE)-cuda: $(TEST_CASE)-cuda.cu $(TEST_CASE)-util.h ../util.h ../cuda-util.h
	nvc++ -O3 -std=c++17 -o $(BUILD_DIR)/$(TEST_CASE)-cuda $(TEST_CASE)-cuda.cu

$(BUILD_DIR)/$(TEST_CASE)-omp-target: $(TEST_CASE)-omp-target.cpp $(TEST_CASE)-util.h ../util.h
	nvc++ -O3 -mp=gpu -o $(BUILD_DIR)/$(TEST_CASE)-omp-target $(TEST_CASE)-omp-target.cpp # -Minfo=mp


# aliases without build directory

.PHONY: $(TEST_CASE)-base
$(TEST_CASE)-base: $(BUILD_DIR)/$(TEST_CASE)-base

.PHONY: $(TEST_CASE)-omp-host
$(TEST_CASE)-omp-host: $(BUILD_DIR)/$(TEST_CASE)-omp-host

.PHONY: $(TEST_CASE)-cuda
$(TEST_CASE)-cuda: $(BUILD_DIR)/$(TEST_CASE)-cuda

.PHONY: $(TEST_CASE)-omp-target
$(TEST_CASE)-omp-target: $(BUILD_DIR)/$(TEST_CASE)-omp-target


# automated benchmark target

.PHONY: bench
bench: all
# 	@echo "Base:"
# 	$(BUILD_DIR)/$(TEST_CASE)-base $(PARAMETERS)
# 	@echo ""
# 	@echo "OpenMP Host:"
# 	$(BUILD_DIR)/$(TEST_CASE)-omp-host $(PARAMETERS)
# 	@echo ""
	@echo "CUDA:"
	$(BUILD_DIR)/$(TEST_CASE)-cuda $(PARAMETERS)
	@echo ""
	@echo "OpenMP Target:"
	$(BUILD_DIR)/$(TEST_CASE)-omp-target $(PARAMETERS)
	@echo ""


# profiling target - nsight systems

.PHONY: nsys
nsys: all mk-profile-dir
	@echo "CUDA:"
	nsys profile -o $(PROFILE_DIR)/$(TEST_CASE)-cuda --force-overwrite=true $(BUILD_DIR)/$(TEST_CASE)-cuda $(PARAMETERS)
	@echo ""
	@echo "OpenMP Target:"
	nsys profile -o $(PROFILE_DIR)/$(TEST_CASE)-omp-target --force-overwrite=true $(BUILD_DIR)/$(TEST_CASE)-omp-target $(PARAMETERS)
	@echo ""


.PHONY: nsys-stats
nsys-stats: all mk-profile-dir
	@echo "CUDA:"
	nsys profile -o /tmp/$(TEST_CASE)-cuda --force-overwrite=true --stats=true $(BUILD_DIR)/$(TEST_CASE)-cuda $(PARAMETERS) > $(PROFILE_DIR)/${TEST_CASE}-cuda-nsys-stats
	cat $(PROFILE_DIR)/${TEST_CASE}-cuda-nsys-stats
	@echo ""
	@echo "OpenMP Target:"
	nsys profile -o /tmp/$(TEST_CASE)-omp-target --force-overwrite=true --stats=true $(BUILD_DIR)/$(TEST_CASE)-omp-target $(PARAMETERS) > $(PROFILE_DIR)/${TEST_CASE}-omp-target-nsys-stats
	cat $(PROFILE_DIR)/${TEST_CASE}-omp-target-nsys-stats
	@echo ""


# profiling target - nsight compute

.PHONY: ncu
ncu: all mk-profile-dir
	@echo "CUDA:"
	ncu -o $(PROFILE_DIR)/$(TEST_CASE)-cuda --force-overwrite --launch-skip=$(NWARM) --launch-count=1 --set=full $(BUILD_DIR)/$(TEST_CASE)-cuda $(PARAMETERS)
	@echo ""
	@echo "OpenMP Target:"
	ncu -o $(PROFILE_DIR)/$(TEST_CASE)-omp-target --force-overwrite --launch-skip=$(NWARM) --launch-count=1 --set=full $(BUILD_DIR)/$(TEST_CASE)-omp-target $(PARAMETERS)
	@echo ""


.PHONY: ncu-cl
ncu-cl: all mk-profile-dir
	@echo "CUDA:"
	ncu --launch-skip=$(NWARM) --launch-count=1 $(BUILD_DIR)/$(TEST_CASE)-cuda $(PARAMETERS) > $(PROFILE_DIR)/${TEST_CASE}-cuda-ncu-cl
	cat $(PROFILE_DIR)/${TEST_CASE}-cuda-ncu-cl
	@echo ""
	@echo "OpenMP Target:"
	ncu --launch-skip=$(NWARM) --launch-count=1 $(BUILD_DIR)/$(TEST_CASE)-omp-target $(PARAMETERS) > $(PROFILE_DIR)/${TEST_CASE}-omp-target-ncu-cl
	cat $(PROFILE_DIR)/${TEST_CASE}-omp-target-ncu-cl
	@echo ""


.PHONY: ncu-metrics
ncu-metrics: all mk-profile-dir
	@echo "CUDA:"
	ncu --launch-skip=$(NWARM) --launch-count=1 --metrics=$(METRICS) $(BUILD_DIR)/$(TEST_CASE)-cuda $(PARAMETERS) > $(PROFILE_DIR)/${TEST_CASE}-cuda-ncu-metrics
	cat $(PROFILE_DIR)/${TEST_CASE}-cuda-ncu-metrics
	@echo ""
	@echo "OpenMP Target:"
	ncu --launch-skip=$(NWARM) --launch-count=1 --metrics=$(METRICS) $(BUILD_DIR)/$(TEST_CASE)-omp-target $(PARAMETERS) > $(PROFILE_DIR)/${TEST_CASE}-omp-target-ncu-metrics
	cat $(PROFILE_DIR)/${TEST_CASE}-omp-target-ncu-metrics
	@echo ""


# helper function

.PHONY: echo-gpu
echo-gpu:
	echo $(GPU)


# clean target

.PHONY: clean
clean:
	rm $(targets)
