CXX = mpicxx
CXXFLAGS = -std=c++17 -O2

TARGET = mytask
SRCS = main.cpp reader.cpp compute.cpp logging.cpp redistribute.cpp

all:
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET)

clean:
	rm -f $(TARGET)
