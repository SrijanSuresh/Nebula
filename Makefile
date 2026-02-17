# Paths
CUDA_PATH = /usr/local/cuda
SRC_DIR = src
EXT_DIR = external
BUILD_DIR = build
BIN_DIR = bin

# Compilers
CXX = g++
CC = gcc
NVCC = $(CUDA_PATH)/bin/nvcc

# Flags - Added -ldl and ensured include paths cover glad
CXXFLAGS = -Iinclude -I$(CUDA_PATH)/include -lGL -lglfw -ldl
NVCCFLAGS = -Iinclude -L$(CUDA_PATH)/lib64 -lcudart

# Targets - Ensure glad.o is linked into the final binary
all: $(BIN_DIR)/titan_nebula

$(BIN_DIR)/titan_nebula: $(BUILD_DIR)/main.o $(BUILD_DIR)/nebula_math.o $(BUILD_DIR)/glad.o
	mkdir -p $(BIN_DIR)
	$(NVCC) $^ -o $@ $(CXXFLAGS)

# Compile glad.c from the external folder
$(BUILD_DIR)/glad.o: $(EXT_DIR)/glad.c
	mkdir -p $(BUILD_DIR)
	$(CC) -c $< -o $@ -Iinclude

$(BUILD_DIR)/main.o: $(SRC_DIR)/main.cpp
	mkdir -p $(BUILD_DIR)
	$(CXX) -c $< -o $@ $(CXXFLAGS)

$(BUILD_DIR)/nebula_math.o: $(SRC_DIR)/nebula_math.cu
	mkdir -p $(BUILD_DIR)
	$(NVCC) -c $< -o $@ $(NVCCFLAGS)

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)
