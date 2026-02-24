CXX ?= g++

SHARED   := ../../shared
INCLUDES := -I$(SHARED) -Iinclude
CXXFLAGS := -O3 -std=c++20 -Wall $(INCLUDES)

# ─── Source headers ────────────────────────────────────────────
BCI_HEADERS := $(wildcard include/*.hpp) $(wildcard include/sim/*.hpp)

# ─── Targets ──────────────────────────────────────────────────

test: test-unit test-synthetic

test-unit: tests/test_metrics.cpp tests/test_riemannian.cpp $(BCI_HEADERS)
	$(CXX) $(CXXFLAGS) -o tests/test_metrics    tests/test_metrics.cpp
	$(CXX) $(CXXFLAGS) -o tests/test_riemannian tests/test_riemannian.cpp
	./tests/test_metrics
	./tests/test_riemannian

test-synthetic: tests/test_synthetic.cpp $(BCI_HEADERS)
	$(CXX) $(CXXFLAGS) -o tests/test_synthetic tests/test_synthetic.cpp
	./tests/test_synthetic

clean:
	rm -f tests/test_metrics tests/test_riemannian tests/test_synthetic

.PHONY: test test-unit test-synthetic clean
