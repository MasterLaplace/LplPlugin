CXX ?= g++

SHARED   := ../../shared
INCLUDES := -I$(SHARED) -Iinclude
CXXFLAGS := -O3 -std=c++20 -Wall $(INCLUDES)

# ─── Optional third-party detection ──────────────────────────
# Eigen3
ifneq ($(shell pkg-config --exists eigen3 2>/dev/null && echo yes),)
    CXXFLAGS += -DLPL_USE_EIGEN $(shell pkg-config --cflags eigen3)
else ifneq ($(wildcard /usr/include/eigen3/Eigen/Dense),)
    CXXFLAGS += -DLPL_USE_EIGEN -I/usr/include/eigen3
endif

# Boost.Lockfree (header-only)
ifneq ($(wildcard /usr/include/boost/lockfree/spsc_queue.hpp),)
    CXXFLAGS += -DLPL_USE_BOOST
endif

# liblsl
ifneq ($(shell pkg-config --exists lsl 2>/dev/null && echo yes),)
    CXXFLAGS += -DLPL_USE_LSL $(shell pkg-config --cflags lsl)
    LDFLAGS  += $(shell pkg-config --libs lsl)
else ifneq ($(wildcard /usr/include/lsl_cpp.h),)
    CXXFLAGS += -DLPL_USE_LSL
    LDFLAGS  += -llsl
endif

# ─── Source headers ────────────────────────────────────────────
BCI_HEADERS := $(wildcard include/*.hpp) $(wildcard include/sim/*.hpp) $(wildcard openvibe/*.hpp)

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
