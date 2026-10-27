CXX ?= c++
CXXFLAGS ?= -O3 -march=native -std=c++17 -Wall -Wextra -Wpedantic
all: build/kfnnue
build/kfnnue: src/main.cpp
	mkdir -p build
	$(CXX) $(CXXFLAGS) $< -o $@
test: build/kfnnue
	./build/kfnnue selftest
clean:
	rm -rf build
.PHONY: all test clean
