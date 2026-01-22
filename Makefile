NVCC = nvcc
INC = -I./include
SRC = src/main.cu src/kernels.cu
TARGET = edge_detector

# Build the executable
all:
	$(NVCC) $(INC) $(SRC) -o $(TARGET)

# Clean build files
clean:
	del $(TARGET).exe $(TARGET).exp $(TARGET).lib