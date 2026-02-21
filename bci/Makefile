CXX ?= g++

SHARED   := ../shared
INCLUDES := -I$(SHARED) -Iinclude
CXXFLAGS := -O3 -std=c++20 -Wall $(INCLUDES)

test: tests/test_metrics.cpp tests/test_riemannian.cpp \
      include/SignalMetrics.hpp include/OpenBCIDriver.hpp include/RiemannianGeometry.hpp
	$(CXX) $(CXXFLAGS) -o tests/test_metrics    tests/test_metrics.cpp
	$(CXX) $(CXXFLAGS) -o tests/test_riemannian tests/test_riemannian.cpp
	./tests/test_metrics
	./tests/test_riemannian

clean:
	rm -f tests/test_metrics tests/test_riemannian

.PHONY: test clean
