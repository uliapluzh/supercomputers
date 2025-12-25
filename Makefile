CXX = mpicxx
NVCC = nvcc

CXXFLAGS  = -std=c++17 -O2

# RTX 3060 → compute capability 8.6
NVCCFLAGS = -O2 -Xcompiler -fPIC \
            -gencode arch=compute_86,code=sm_86 \
            -gencode arch=compute_86,code=compute_86

OBJS = main.o reader.o compute.o logging.o redistribute.o reduce.o

TARGET = mytask
PLUGIN = libmin_delta_cuda.so

all: $(PLUGIN) $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ -ldl

$(PLUGIN): compute_cuda.cu
	$(NVCC) $(NVCCFLAGS) -shared $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f *.o $(TARGET) $(PLUGIN)