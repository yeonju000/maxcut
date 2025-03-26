# Compiler and flags
CXX = g++
CXXFLAGS = -Wall -std=c++11 -O3

# Target executable name
TARGET = maxcut

# Default target
all: $(TARGET)

# Rule to link the program
$(TARGET): maxcut.cpp
   $(CXX) $(CXXFLAGS) -o $(TARGET) maxcut.cpp

# Rule to run the program
run:
   ./$(TARGET)

# Rule to clean old builds
clean:
   rm -f $(TARGET)
