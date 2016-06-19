OUT             = MacCormackFluid

NVCC            = nvcc
CPP             = g++
CFG             = release

CFLAGS.release  = -Wall -std=c++11 -O2 -ffast-math
CFLAGS.debug    = -g -Wall -std=c++11 -O0
CFLAGS = $(CFLAGS.$(CFG))

CUFLAGS.release = -arch=sm_30 -O3 -D_FORCE_INLINES -use_fast_math -Xcompiler="-fPIC" -Xcompiler="-fexpensive-optimizations" -Xcompiler="-funroll-loops" -Xcompiler="-ffast-math" -std=c++11 -Xcompiler="-D__STRICT_ANSI__" -Xcompiler="-D_MWAITXINTRIN_H_INCLUDED"
CUFLAGS.debug   = -arch=sm_30 -Xcompiler="-fPIC" -Xcompiler="-O0 -g" -std=c++11 -Xcompiler="-D__STRICT_ANSI__" -Xcompiler="-D_MWAITXINTRIN_H_INCLUDED"
CUFLAGS = $(CUFLAGS.$(CFG))

LDFLAGS         = -lm -ldl -lrt

CUSRC           = $(wildcard *.cu)
CUOBJ           = $(CUSRC:.cu=.o)

CPPSRC          = $(wildcard *.cpp)
CPPOBJ          = $(CPPSRC:.cpp=.o)

HEADERS         = $(wildcard *.hpp)

all: $(CPPOBJ) $(CUOBJ)
    $(NVCC) -o $(OUT) $(CPPOBJ) $(CUOBJ) $(LDFLAGS)

release:
    $(MAKE) $(MAKEFILE) CFG=release

debug:
    $(MAKE) $(MAKEFILE) CFG=debug

%.o: %.cpp $(HEADERS)
    $(CPP) $(CFLAGS) -c -o $@ $<

%.o: %.cu
    $(NVCC) $(CUFLAGS) -c $< -o $@

clean:
    rm -rf $(OUT) $(CPPOBJ) $(CUOBJ)

.PHONY : all clean