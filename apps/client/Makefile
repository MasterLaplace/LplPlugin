CC              ?= gcc
CXX             ?= g++
ANDROID_NDK_HOME ?=
ANDROID_API     = 24

SHARED   := ../../shared
ENGINE   := ../../engine
BCI      := ../../plugins/bci/include

INCLUDES := -I$(SHARED) -I$(ENGINE) -I$(BCI)
CXXFLAGS := -O3 -std=c++20 -Wall -pthread -DLPL_USE_SOCKET $(INCLUDES)

ENGINE_HEADERS := $(wildcard $(ENGINE)/*.hpp)
SHARED_HEADERS := $(SHARED)/lpl_protocol.h $(SHARED)/Math.hpp
BCI_HEADERS    := $(wildcard $(BCI)/*.hpp) $(wildcard $(BCI)/*/*.hpp)

# BCI Library from xmake
BCI_LIB_DIR := ../../plugins/bci/build/linux/x86_64/release
BCI_LIB     := $(BCI_LIB_DIR)/liblpl-bci.a

visual: bci_lib visual.cpp $(ENGINE_HEADERS) $(SHARED_HEADERS) $(BCI_HEADERS)
	$(CXX) $(CXXFLAGS) -o visual visual.cpp $(BCI_LIB) -lglfw -lGLEW -lGL -lm

android: bci_lib
	@if [ -z "$(ANDROID_NDK_HOME)" ]; then \
		echo "ERREUR: NDK introuvable — définir ANDROID_NDK_HOME."; exit 1; \
	fi
	$(eval TOOLCHAIN := $(ANDROID_NDK_HOME)/toolchains/llvm/prebuilt/linux-x86_64/bin)
	$(TOOLCHAIN)/aarch64-linux-android$(ANDROID_API)-clang \
		-c $(ANDROID_NDK_HOME)/sources/android/native_app_glue/android_native_app_glue.c \
		-I$(ANDROID_NDK_HOME)/sources/android/native_app_glue -fPIC -o android_glue.o
	$(TOOLCHAIN)/aarch64-linux-android$(ANDROID_API)-clang++ \
		-O3 -std=c++20 -shared -u ANativeActivity_onCreate \
		$(INCLUDES) -I$(ANDROID_NDK_HOME)/sources/android/native_app_glue \
		-fPIC -DLPL_USE_SOCKET -D__ANDROID__ -static-libstdc++ \
		-o liblpl_visual.so visual_android.cpp android_glue.o \
		-landroid -llog -lEGL -lGLESv2 -Wl,-soname,liblpl_visual.so
	@rm android_glue.o

bci_lib:
	cd ../../plugins/bci && xmake f -c && xmake config --with_brainflow=n && xmake build lpl-bci

clean:
	rm -f visual liblpl_visual.so *.o
	cd ../../plugins/bci && xmake clean

.PHONY: visual android clean bci_lib
