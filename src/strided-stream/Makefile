# configuration

TEST_CASE = strided-stream
BUILD_DIR = ../../build/$(TEST_CASE)


# default benchmark parameters

NX    = $$((64 * 1024 * 1024))
NWARM = 2
NIT   = 10

PARAMETERS = ${NX} ${NWARM} ${NIT}


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
	@for stride in 1 2 ; do \
		echo "Base, stride $$stride:" ; \
		$(BUILD_DIR)/$(TEST_CASE)-base $(PARAMETERS) $$stride ; \
		echo "" ; \
	done
	@for stride in 1 2 ; do \
		echo "OpenMP Host, stride $$stride:" ; \
		$(BUILD_DIR)/$(TEST_CASE)-omp-host $(PARAMETERS) $$stride ; \
		echo "" ; \
	done
	@for stride in 1 2 ; do \
		echo "CUDA, stride $$stride:" ; \
		$(BUILD_DIR)/$(TEST_CASE)-cuda $(PARAMETERS) $$stride ; \
		echo "" ; \
	done
	@for stride in 1 2 ; do \
		echo "OpenMP Target, stride $$stride:" ; \
		$(BUILD_DIR)/$(TEST_CASE)-omp-target $(PARAMETERS) $$stride ; \
		echo "" ; \
	done


# clean target

.PHONY: clean
clean:
	rm $(targets)
